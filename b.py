# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from PIL import Image
from data.data_utils import pil_img2rgb
import os
import gc

from data.data_utils import (
    create_sparse_mask, 
    get_flattened_position_ids_extrapolate, 
    get_flattened_position_ids_interpolate,
    patchify, 
)
from .qwen2_navit import NaiveCache
from .modeling_utils import MLPconnector, TimestepEmbedder, PositionEmbedding
from modeling.cache_utils.taylorseer import cache_init

from tqdm import tqdm
from modeling.longclip_b.model import longclip

import time

class BagelConfig(PretrainedConfig):
    def __init__(
        self,
        visual_gen=True,
        visual_und=True,
        llm_config=None,
        vit_config=None,
        vae_config=None,
        latent_patch_size=2,
        max_latent_size=32,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        interpolate_pos=False,
        timestep_shift=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.visual_gen = visual_gen
        self.visual_und = visual_und
        self.llm_config = llm_config
        self.vit_config = vit_config
        self.vae_config = vae_config
        self.latent_patch_size = latent_patch_size
        self.max_latent_size = max_latent_size
        self.vit_max_num_patch_per_side = vit_max_num_patch_per_side
        self.connector_act = connector_act
        self.interpolate_pos = interpolate_pos
        self.timestep_shift = timestep_shift


class Bagel(PreTrainedModel):
    config_class = BagelConfig
    base_model_prefix = 'bagel'

    def __init__(self, language_model, vit_model, config: BagelConfig):
        super().__init__(config)    
        self.language_model = language_model
        self.hidden_size = config.llm_config.hidden_size
        self.use_moe = "Mo" in config.llm_config.layer_module
        self.num_heads = config.llm_config.num_attention_heads

        if config.visual_gen:
            self.latent_patch_size = config.latent_patch_size
            self.timestep_shift = config.timestep_shift
            self.latent_downsample = config.vae_config.downsample * config.latent_patch_size
            self.max_latent_size = config.max_latent_size
            self.latent_channel = config.vae_config.z_channels
            self.patch_latent_dim = self.latent_patch_size ** 2 * self.latent_channel
            self.time_embedder = TimestepEmbedder(self.hidden_size)
            self.vae2llm = nn.Linear(self.patch_latent_dim, self.hidden_size)
            self.llm2vae = nn.Linear(self.hidden_size, self.patch_latent_dim)
            self.latent_pos_embed = PositionEmbedding(self.max_latent_size, self.hidden_size)

        if config.visual_und:
            self.vit_model = vit_model
            self.vit_patch_size = config.vit_config.patch_size
            self.vit_max_num_patch_per_side = config.vit_max_num_patch_per_side
            self.vit_hidden_size = config.vit_config.hidden_size
            self.connector = MLPconnector(self.vit_hidden_size, self.hidden_size, config.connector_act)
            self.vit_pos_embed = PositionEmbedding(self.vit_max_num_patch_per_side, self.hidden_size)

        if config.interpolate_pos:
            self.get_flattened_position_ids = get_flattened_position_ids_interpolate
        else:
            self.get_flattened_position_ids = get_flattened_position_ids_extrapolate

        self.config = config
        self._init_weights()
        self.self_prompt = None
        self.prompt_clip_feature = None

    def _init_weights(self):
        if self.config.visual_gen:
            nn.init.constant_(self.llm2vae.weight, 0)
            nn.init.constant_(self.llm2vae.bias, 0)

    def forward(
        self,
        sequence_length: int,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        sample_lens: List[int],
        packed_position_ids: torch.LongTensor,
        nested_attention_masks: List[torch.Tensor] = None,
        split_lens: List[int] = None,
        attn_modes: List[str] = None,
        # for visual understanding
        ce_loss_indexes: Optional[torch.BoolTensor] = None,
        packed_label_ids: Optional[torch.LongTensor] = None,
        packed_vit_tokens: Optional[torch.Tensor] = None,
        packed_vit_token_indexes: Optional[torch.LongTensor] = None,
        packed_vit_position_ids: Optional[torch.LongTensor] = None,
        vit_token_seqlens: Optional[torch.IntTensor] = None,
        # for visual generation
        padded_latent: Optional[torch.Tensor] = None,
        patchified_vae_latent_shapes: Optional[List[Tuple[int, int]]] = None,
        packed_latent_position_ids: Optional[torch.LongTensor] = None,
        packed_vae_token_indexes: Optional[torch.LongTensor] = None,
        packed_timesteps: Optional[torch.LongTensor] = None,
        mse_loss_indexes: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            sequence_length: length of sequence.
            packed_text_ids: 1-D int tensor, packed text token ids.
            packed_text_indexes: 1-D int tensor, packed text token indexes in sequence.
            sample_lens: A list of N ints, length of each sample in packed_sequence.
            nested_attention_masks: A list of N 2-D float tensor,  where 0.0 means attention and 
                -inf means ignore.
            packed_position_ids: packed 1-D positions, an image has only one global position shared
                by all latent tokens.

            packed_vit_tokens: packed patchified image tokens for vit model.
            packed_vit_position_ids: 1-D int tensor, the position of each token for vit model.
            packed_vit_token_indexes: 1-D int tensor, packed vit token indexes in sequence.
            vit_token_seqlens: 1-D int tensor, the length of each image tokens for vit model.
            packed_label_ids: 1-D int tensor, packed label token ids.
            ce_loss_indexes: 1-D bool tensor, where to compute ce loss.

            padded_latent: padded latent from VAE encoder.
            patchified_vae_latent_shapes: A list of (h, w) tuples, patchfied latent shapes of each image.
            packed_latent_position_ids: 1-D int tensor, the position of each token for latent.
            packed_vae_token_indexes: 1-D int tensor, padded image token indexes in sequence.
            packed_timesteps: 1-D float tensor, flow timesteps. 0 indicates use clean image.
            mse_loss_indexes: 1-D bool tensor, where to compute mse loss.
        """
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros(size=(sequence_length, self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        if nested_attention_masks is None:
            sparse_mask = create_sparse_mask(sample_lens, split_lens, attn_modes, packed_text_embedding.device)
            seqlen = sum(sample_lens)
            block_mask = create_block_mask(
                sparse_mask, B=1, H=self.num_heads, Q_LEN=seqlen, KV_LEN=seqlen, 
                device=packed_text_embedding.device, BLOCK_SIZE=128, _compile=True
            )
            attention_mask = block_mask
        else:
            attention_mask = nested_attention_masks

        if self.config.visual_und:
            cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
            cu_seqlens = cu_seqlens.to(torch.int32)
            max_seqlen = torch.max(vit_token_seqlens).item()
            packed_vit_token_embed = self.vit_model(
                packed_pixel_values=packed_vit_tokens, 
                packed_flattened_position_ids=packed_vit_position_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            packed_vit_token_embed = self.connector(packed_vit_token_embed)
            vit_token_pos_emb = self.vit_pos_embed(packed_vit_position_ids)
            packed_vit_token_embed = packed_vit_token_embed + vit_token_pos_emb
            packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed

        if self.config.visual_gen:
            p = self.latent_patch_size
            packed_latent = []
            for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
                latent = latent[:, :h * p, :w * p].reshape(self.latent_channel, h, p, w, p)
                latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)
                packed_latent.append(latent)
            packed_latent_clean = torch.cat(packed_latent, dim=0)

            noise = torch.randn_like(packed_latent_clean)
            packed_timesteps = torch.sigmoid(packed_timesteps)
            packed_timesteps = self.timestep_shift * packed_timesteps / (1 + (self.timestep_shift - 1) * packed_timesteps)
            packed_latent = (1 - packed_timesteps[:, None]) * packed_latent_clean + packed_timesteps[:, None] * noise
            packed_timestep_embeds = self.time_embedder(packed_timesteps)
            latent_token_pos_emb = self.latent_pos_embed(packed_latent_position_ids)
            packed_latent = self.vae2llm(packed_latent) + packed_timestep_embeds + latent_token_pos_emb
            packed_sequence[packed_vae_token_indexes] = packed_latent

        extra_inputs = {}
        if self.use_moe:
            packed_und_token_indexes = packed_text_indexes
            if packed_vit_token_indexes is not None:
                packed_und_token_indexes=torch.cat([packed_text_indexes, packed_vit_token_indexes], dim=0)
            extra_inputs.update(
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_vae_token_indexes,
            )

        last_hidden_state = self.language_model(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_ids=packed_position_ids,
            **extra_inputs,
        )

        mse = None
        if self.config.visual_gen:
            packed_mse_preds = self.llm2vae(last_hidden_state[mse_loss_indexes])
            target = noise - packed_latent_clean # NOTE: v_t=dx_t/dt=x_1-x_0, pointing from data to noise
            has_mse = packed_timesteps > 0
            mse = (packed_mse_preds - target[has_mse]) ** 2

        ce = None
        if ce_loss_indexes is not None:
            packed_ce_preds = self.language_model.lm_head(last_hidden_state[ce_loss_indexes])
            ce = F.cross_entropy(packed_ce_preds, packed_label_ids, reduction="none")

        return dict(mse=mse, ce=ce)


    def prepare_prompts(self, curr_kvlens, curr_rope, prompts, tokenizer, new_token_ids):
        packed_text_ids = list()
        packed_text_position_ids = list()
        text_token_lens = list()
        packed_text_indexes = list()
        packed_key_value_indexes = list()

        curr = 0
        newlens, new_rope = list(), list()
        for prompt, curr_kvlen, curr_position_id in zip(prompts, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            text_ids = tokenizer.encode(prompt)
            text_ids = [new_token_ids['bos_token_id']] + text_ids + [new_token_ids['eos_token_id']]
            text_token_lens.append(len(text_ids))
            packed_text_ids.extend(text_ids)
            packed_text_position_ids.extend(range(curr_position_id, curr_position_id + len(text_ids)))
            packed_text_indexes.extend(range(curr, curr + len(text_ids)))
            newlens.append(curr_kvlen + len(text_ids))
            new_rope.append(curr_position_id + len(text_ids))
            curr += len(text_ids)

        generation_input = {
            "text_token_lens": torch.tensor(text_token_lens, dtype=torch.int),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_position_ids": torch.tensor(packed_text_position_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_text(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.IntTensor,
        packed_text_position_ids: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
    ):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {"mode": "und"}
        # breakpoint()
        output = self.language_model.forward_inference(
            packed_query_sequence=packed_text_embedding,
            query_lens=text_token_lens,
            packed_query_position_ids=packed_text_position_ids,
            packed_query_indexes=packed_text_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            is_causal=True,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_vit_images(self, curr_kvlens, curr_rope, images, transforms, new_token_ids):
        packed_vit_token_indexes = list()
        vit_token_seqlens, packed_vit_tokens, packed_vit_position_ids = list(), list(), list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        _curr = curr = 0
        newlens, new_rope = list(), list()
        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            image_tensor = transforms(image)
            vit_position_ids = self.get_flattened_position_ids(
                image_tensor.size(1), image_tensor.size(2), 
                self.vit_patch_size, 
                max_num_patches_per_side=self.vit_max_num_patch_per_side
            )
            vit_tokens = patchify(image_tensor, self.vit_patch_size)
            packed_vit_tokens.append(vit_tokens)
            num_img_tokens = vit_tokens.shape[0]
            packed_vit_position_ids.append(vit_position_ids)
            vit_token_seqlens.append(num_img_tokens)
            packed_vit_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            packed_position_ids.extend([curr_position_id] * (num_img_tokens + 2))
            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            new_rope.append(curr_position_id + 1)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "vit_token_seqlens": torch.tensor(vit_token_seqlens, dtype=torch.int),
            "packed_vit_tokens": torch.cat(packed_vit_tokens, dim=0),
            "packed_vit_position_ids": torch.cat(packed_vit_position_ids, dim=0),
            "packed_vit_token_indexes": torch.tensor(packed_vit_token_indexes, dtype=torch.long),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_vit(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_vit_tokens: torch.Tensor,
        packed_vit_token_indexes: torch.LongTensor,
        packed_vit_position_ids: torch.LongTensor,
        vit_token_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
    ):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
        cu_seqlens = cu_seqlens.to(torch.int32)
        max_seqlen = torch.max(vit_token_seqlens).item()
        packed_vit_token_embed = self.vit_model(
            packed_pixel_values=packed_vit_tokens, 
            packed_flattened_position_ids=packed_vit_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        packed_vit_token_embed = self.connector(packed_vit_token_embed)
        pos_emb = self.vit_pos_embed(packed_vit_position_ids)
        packed_vit_token_embed = packed_vit_token_embed + pos_emb
        if packed_vit_token_embed.dtype != packed_sequence.dtype:
            packed_vit_token_embed = packed_vit_token_embed.to(packed_sequence.dtype)
        packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {"mode": "und"}

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            is_causal=False,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_vae_images(self, curr_kvlens, curr_rope, images, transforms, new_token_ids, timestep=0):
        patchified_vae_latent_shapes, packed_vae_position_ids = list(), list()
        packed_vae_token_indexes = list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        _curr = curr = 0
        vae_image_tensors = list()
        newlens, new_rope = list(), list()
        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            image_tensor = transforms(image)
            vae_image_tensors.append(image_tensor)
            vae_posiiton_ids = self.get_flattened_position_ids(
                image_tensor.size(1), image_tensor.size(2),
                self.latent_downsample, 
                max_num_patches_per_side=self.max_latent_size
            )
            packed_vae_position_ids.append(vae_posiiton_ids)
            H, W = image_tensor.shape[1:]
            h = H // self.latent_downsample
            w = W // self.latent_downsample
            patchified_vae_latent_shapes.append((h, w))

            num_img_tokens = w * h
            packed_vae_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            packed_position_ids.extend([curr_position_id] * (num_img_tokens + 2))
            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            new_rope.append(curr_position_id + 1)

        image_sizes = [item.shape for item in vae_image_tensors]
        max_image_size = [max(item) for item in list(zip(*image_sizes))]
        padded_images = torch.zeros(size=(len(vae_image_tensors), *max_image_size))
        for i, image_tensor in enumerate(vae_image_tensors):
            padded_images[i, :, :image_tensor.shape[1], :image_tensor.shape[2]] = image_tensor

        generation_input = {
            "padded_images": padded_images,
            "patchified_vae_latent_shapes": patchified_vae_latent_shapes,
            "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0),
            "packed_timesteps": torch.tensor([timestep]),
            "packed_vae_token_indexes": torch.tensor(packed_vae_token_indexes, dtype=torch.long),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_vae(
        self,
        vae_model,
        past_key_values: NaiveCache,
        padded_images: torch.Tensor,
        patchified_vae_latent_shapes: List,
        packed_vae_position_ids: torch.LongTensor,
        packed_timesteps: torch.Tensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_key_value_indexes: torch.Tensor,
    ):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        padded_latent = vae_model.encode(padded_images.to(torch.bfloat16).to("cuda"))

        p = self.latent_patch_size
        packed_latent = list()
        for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
            latent = latent[:, :h * p, :w * p].reshape(self.latent_channel, h, p, w, p)
            latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)
            packed_latent.append(latent)
        packed_latent = torch.cat(packed_latent, dim=0)
        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
        packed_timestep_embeds = self.time_embedder(packed_timesteps.to(torch.bfloat16))
        packed_latent = self.vae2llm(packed_latent) + packed_timestep_embeds + packed_pos_embed
        if packed_latent.dtype != packed_sequence.dtype:
            packed_latent = packed_latent.to(packed_sequence.dtype)
        packed_sequence[packed_vae_token_indexes] = packed_latent

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {
                "mode": "gen",
                "packed_vae_token_indexes": packed_vae_token_indexes,
                "packed_text_indexes": packed_text_indexes
            }

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=True,
            is_causal=False,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_vae_latent(self, curr_kvlens, curr_rope, image_sizes, new_token_ids, model_dtype=None):
        packed_text_ids, packed_text_indexes = list(), list()
        packed_vae_position_ids, packed_vae_token_indexes, packed_init_noises = list(), list(), list()
        packed_position_ids, packed_seqlens, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        query_curr = curr = 0
        for (H, W), curr_kvlen, curr_position_id in zip(image_sizes, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            vae_posiiton_ids = self.get_flattened_position_ids(
                H, W,
                self.latent_downsample, 
                max_num_patches_per_side=self.max_latent_size
            )
            packed_vae_position_ids.append(vae_posiiton_ids)

            h, w = H // self.latent_downsample, W // self.latent_downsample
            num_image_tokens = h * w
            if model_dtype is None:
                packed_init_noises.append(
                    torch.randn(num_image_tokens, self.latent_channel * self.latent_patch_size ** 2)
                )
            else:
                packed_init_noises.append(
                    torch.randn(num_image_tokens, self.latent_channel * self.latent_patch_size ** 2).to(model_dtype)
                )
            packed_vae_token_indexes.extend(range(query_curr, query_curr + num_image_tokens))
            packed_indexes.extend(range(curr, curr + num_image_tokens))
            curr += num_image_tokens
            query_curr += num_image_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            packed_position_ids.extend([curr_position_id] * (num_image_tokens + 2))
            packed_seqlens.append(num_image_tokens + 2)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_init_noises": torch.cat(packed_init_noises, dim=0),
            "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0),
            "packed_vae_token_indexes": torch.tensor(packed_vae_token_indexes, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        return generation_input

    def prepare_vae_latent_cfg(self, curr_kvlens, curr_rope, image_sizes):
        packed_position_ids, packed_indexes, packed_key_value_indexes = list(), list(), list()

        query_curr = curr = 0
        for (H, W), curr_kvlen, curr_position_id in zip(image_sizes, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            h, w = H // self.latent_downsample, W // self.latent_downsample
            num_image_tokens = h * w
            packed_indexes.extend(range(curr, curr + num_image_tokens))
            curr += num_image_tokens
            query_curr += num_image_tokens

            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            packed_position_ids.extend([curr_position_id] * (num_image_tokens + 2))

        generation_input = {
            "cfg_packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "cfg_key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "cfg_packed_query_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "cfg_packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        return generation_input

    @torch.no_grad
    def generate_image(
        self,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_init_noises: torch.Tensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        past_key_values: NaiveCache,
        key_values_lens: torch.IntTensor,
        packed_key_value_indexes: torch.LongTensor,
        num_timesteps: int = 24,
        timestep_shift: float = 1.0,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        cfg_interval: Optional[Tuple[float, float]] = [0, 1],
        # cfg_text
        cfg_text_scale: float = 1.0,
        cfg_text_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_text_past_key_values: Optional[NaiveCache] = None,
        cfg_text_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_text_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        # cfg_img
        cfg_img_scale: float = 1.0,
        cfg_img_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_img_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_img_past_key_values: Optional[NaiveCache] = None,
        cfg_img_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_img_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_type: str = "parallel",
        # cache_args
        enable_taylorseer=False,
    ):
        if enable_taylorseer:
            self.language_model.model.enable_taylorseer = True
            model_pred_cache_dic, model_pred_current = cache_init(self, num_timesteps)
            model_pred_text_cache_dic, model_pred_text_current = cache_init(self, num_timesteps)
            model_pred_img_cache_dic, model_pred_img_current = cache_init(self, num_timesteps)
        else:
            self.language_model.model.enable_taylorseer = False
            model_pred_cache_dic, model_pred_current = None, None
            model_pred_text_cache_dic, model_pred_text_current = None, None
            model_pred_img_cache_dic, model_pred_img_current = None, None
    
        x_t = packed_init_noises

        timesteps = torch.linspace(1, 0, num_timesteps, device=x_t.device)
        timesteps = timestep_shift * timesteps / (1 + (timestep_shift - 1) * timesteps)
        dts =  timesteps[:-1] - timesteps[1:]
        timesteps = timesteps[:-1]
        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
            timestep = torch.tensor([t] * x_t.shape[0], device=x_t.device)
            if t > cfg_interval[0] and t <= cfg_interval[1]:
                cfg_text_scale_ = cfg_text_scale
                cfg_img_scale_ = cfg_img_scale
            else:
                cfg_text_scale_ = 1.0
                cfg_img_scale_ = 1.0
            v_t = self._forward_flow(
                x_t=x_t,
                timestep=timestep, 
                packed_vae_token_indexes=packed_vae_token_indexes,
                packed_vae_position_ids=packed_vae_position_ids,
                packed_text_ids=packed_text_ids,
                packed_text_indexes=packed_text_indexes,
                packed_position_ids=packed_position_ids,
                packed_indexes=packed_indexes,
                packed_seqlens=packed_seqlens,
                key_values_lens=key_values_lens,
                past_key_values=past_key_values,
                packed_key_value_indexes=packed_key_value_indexes,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                # cfg_text
                cfg_text_scale=cfg_text_scale_,
                cfg_text_packed_position_ids=cfg_text_packed_position_ids,
                cfg_text_packed_query_indexes=cfg_text_packed_query_indexes,
                cfg_text_key_values_lens=cfg_text_key_values_lens,
                cfg_text_past_key_values=cfg_text_past_key_values,
                cfg_text_packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                # cfg_img
                cfg_img_scale=cfg_img_scale_,
                cfg_img_packed_position_ids=cfg_img_packed_position_ids,
                cfg_img_packed_query_indexes=cfg_img_packed_query_indexes,
                cfg_img_key_values_lens=cfg_img_key_values_lens,
                cfg_img_past_key_values=cfg_img_past_key_values,
                cfg_img_packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                cfg_type=cfg_type,
                # cache
                model_pred_cache_dic=model_pred_cache_dic,
                model_pred_current=model_pred_current,
                model_pred_text_cache_dic=model_pred_text_cache_dic,
                model_pred_text_current=model_pred_text_current,
                model_pred_img_cache_dic=model_pred_img_cache_dic,
                model_pred_img_current=model_pred_img_current,
            )
            # breakpoint()
            x_t = x_t - v_t.to(x_t.device) * dts[i] # velocity pointing from data to noise
        
        if enable_taylorseer:
            del model_pred_cache_dic, model_pred_current
            del model_pred_text_cache_dic, model_pred_text_current
            del model_pred_img_cache_dic, model_pred_img_current

        unpacked_latent = x_t.split((packed_seqlens - 2).tolist())
        return unpacked_latent

    def generate_image_reca(
        self,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_init_noises: torch.Tensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        past_key_values: NaiveCache,
        key_values_lens: torch.IntTensor,
        packed_key_value_indexes: torch.LongTensor,
        num_timesteps: int = 24,
        timestep_shift: float = 1.0,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        cfg_interval: Optional[Tuple[float, float]] = [0, 1],
        # cfg_text
        cfg_text_scale: float = 1.0,
        cfg_text_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_text_past_key_values: Optional[NaiveCache] = None,
        cfg_text_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_text_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        # cfg_img
        cfg_img_scale: float = 1.0,
        cfg_img_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_img_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_img_past_key_values: Optional[NaiveCache] = None,
        cfg_img_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_img_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_type: str = "parallel",
        # cache_args
        enable_taylorseer=False,
        clip_model=None,
        clip_processor=None,
        re_update_num = 1,
        update_lr=[(0,49)],
        use_longclip=True,
        use_fgclip=False,
        prompt=None,
        vae_model=None,
        image_shapes=None,
        tokenizer=None, 
        new_token_ids=None,
        image_transform=None,
        use_save_pic=False,
        update_scale=100.0,
        model_dtype=None,
        save_grad=False,
        use_lookback=False,
        lookback_steps=0,
        use_understanding_ce_loss=False,
        ce_max_tokens=32,
        ce_aux_weight=0.1,
        ce_ema_beta=0.9,
        **unused_profile_kwargs,
    ):
        # breakpoint()
        score_list = []
        self.eval()
        # This file is an isolated Native-CE profiling implementation. The
        # production package selects it only through BAGEL_MODEL_IMPL, and CE
        # itself must also be explicitly enabled so default runs stay intact.
        use_understanding_ce_loss = (
            use_understanding_ce_loss
            or os.environ.get("BAGEL_RECA_NATIVE_CE", "0") == "1"
        )
        internal_feature_guidance = (
            use_understanding_ce_loss
            and os.environ.get("BAGEL_RECA_INTERNAL_FEATURE", "0") == "1"
        )
        hybrid_reca_guidance = (
            use_understanding_ce_loss
            and not internal_feature_guidance
            and os.environ.get("BAGEL_RECA_HYBRID", "0") == "1"
        )
        dual_ce_guidance = (
            use_understanding_ce_loss
            and not internal_feature_guidance
            and not hybrid_reca_guidance
            and os.environ.get("BAGEL_RECA_DUAL_CE", "0") == "1"
        )
        if internal_feature_guidance:
            loss_mode = "internal_feature"
        elif hybrid_reca_guidance:
            loss_mode = "ce_clip_joint"
        elif dual_ce_guidance:
            loss_mode = "dual_view_ce"
        elif use_understanding_ce_loss:
            loss_mode = "understanding_ce"
        else:
            loss_mode = "clip"
        if use_understanding_ce_loss:
            ce_max_tokens = int(
                os.environ.get("BAGEL_CE_MAX_TOKENS", str(ce_max_tokens))
            )
            update_scale = float(
                os.environ.get("BAGEL_RECA_UPDATE_SCALE", str(update_scale))
            )
            re_update_num = int(
                os.environ.get("BAGEL_RECA_RE_UPDATE_NUM", str(re_update_num))
            )
        ce_vit_max_side = int(os.environ.get("BAGEL_CE_VIT_MAX_SIDE", "168"))
        ce_target_mode = os.environ.get("BAGEL_CE_TARGET_MODE", "caption")
        if ce_target_mode not in {"caption", "binary_user"}:
            raise ValueError("BAGEL_CE_TARGET_MODE must be caption or binary_user")
        if ce_target_mode != "caption":
            print(f"Native CE target mode: {ce_target_mode}")
        ce_instruction_mode = os.environ.get("BAGEL_CE_INSTRUCTION_MODE", "default")
        if ce_instruction_mode not in {"default", "target_terms"}:
            raise ValueError("BAGEL_CE_INSTRUCTION_MODE must be default or target_terms")
        if ce_instruction_mode != "default":
            print(f"Native CE instruction mode: {ce_instruction_mode}")
        profile_native_ce = os.environ.get("BAGEL_PROFILE_NATIVE_CE", "0") == "1"
        native_ce_anchor_stride = max(1, int(os.environ.get("BAGEL_NATIVE_CE_ANCHOR_STRIDE", "1")))
        first_update_step = min((start for start, _ in update_lr), default=0)
        last_update_step = max((end for _, end in update_lr), default=0)
        use_ce_selection = bool(unused_profile_kwargs.get("use_ce_selection", False))
        use_ce_selection = (
            use_ce_selection
            or os.environ.get("BAGEL_USE_CE_SELECTION", "0") == "1"
        )
        ce_selection_prompt = unused_profile_kwargs.get(
            "ce_selection_prompt", prompt
        ) or prompt
        ce_selection_vit_max_side = int(unused_profile_kwargs.get(
            "ce_selection_vit_max_side",
            os.environ.get("BAGEL_CE_SELECTION_VIT_MAX_SIDE", "168"),
        ))
        ce_selection_every_step = bool(unused_profile_kwargs.get(
            "ce_selection_every_step", False
        )) or os.environ.get("BAGEL_CE_SELECTION_EVERY_STEP", "0") == "1"
        in_step_user_ce_gss = (
            use_understanding_ce_loss
            and use_ce_selection
            and ce_selection_every_step
        )
        final_ideal_ce_select = (
            use_understanding_ce_loss
            and (
                os.environ.get("BAGEL_FINAL_T_IDEAL_CE_SELECT", "0") == "1"
                or (use_ce_selection and not ce_selection_every_step)
            )
        )
        final_ce_select_mode = os.environ.get(
            "BAGEL_FINAL_T_CE_SELECT_MODE", "ideal"
        )
        if final_ce_select_mode not in {"ideal", "user_binary"}:
            raise ValueError(
                "BAGEL_FINAL_T_CE_SELECT_MODE must be ideal or user_binary"
            )
        print(f"ReCA loss mode: {loss_mode}")
        if final_ideal_ce_select:
            print(
                "Final-timestep ideal CE selection: compare unmodified and "
                "fully rectified candidates; CE ties keep the latter"
            )
        if (
            use_understanding_ce_loss
            and not internal_feature_guidance
            and os.environ.get("BAGEL_ENABLE_DROPBP", "0") == "1"
        ):
            print("UMM DropBP: sensitivity calibration enabled")
        if internal_feature_guidance:
            print(
                "Intrinsic feature guidance: reused generation hidden vs contextual text hidden"
            )
        joint_clip_weight = float(os.environ.get("BAGEL_JOINT_CLIP_WEIGHT", "50"))
        dual_ce_binary_weight = float(
            os.environ.get("BAGEL_DUAL_CE_BINARY_WEIGHT", "0.5")
        )
        if not 0.0 <= dual_ce_binary_weight <= 1.0:
            raise ValueError("BAGEL_DUAL_CE_BINARY_WEIGHT must be between 0 and 1")
        dual_ce_user_mode = os.environ.get("BAGEL_DUAL_CE_USER_MODE", "binary")
        if dual_ce_user_mode not in {"binary", "caption_user"}:
            raise ValueError("BAGEL_DUAL_CE_USER_MODE must be binary or caption_user")
        dual_ce_anchor = os.environ.get("BAGEL_DUAL_CE_ANCHOR", "ideal")
        if dual_ce_anchor not in {"ideal", "user"}:
            raise ValueError("BAGEL_DUAL_CE_ANCHOR must be ideal or user")
        if dual_ce_guidance:
            print(
                "ReCA dual-view native CE guidance: "
                f"user_mode={dual_ce_user_mode}, user_weight={dual_ce_binary_weight:.3f}, "
                f"anchor={dual_ce_anchor}"
            )
        if hybrid_reca_guidance:
            print(
                "ReCA joint guidance: one same-point CE+CLIP update per timestep, "
                f"clip_weight={joint_clip_weight:g}"
            )
        if loss_mode == "clip" and clip_model is None:
            if not use_longclip:
                raise RuntimeError(
                    "CLIP-only ReCA requires LongCLIP when clip_model is not "
                    "supplied by the inferencer"
                )
            cached_clip = self.__dict__.get("_reca_longclip_runtime")
            if cached_clip is None:
                if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                    clip_device = "cuda:1"
                elif torch.cuda.is_available():
                    clip_device = "cuda"
                else:
                    clip_device = "cpu"
                longclip_path = os.environ.get(
                    "BAGEL_LONGCLIP_PATH",
                    "./pretrained_models/LongCLIP-B/longclip-B.pt",
                )
                print(f"Loading LongCLIP for CLIP-only guidance on {clip_device}")
                clip_model, clip_processor = longclip.load(
                    longclip_path,
                    device=clip_device,
                )
                clip_model.eval()
                for parameter in clip_model.parameters():
                    parameter.requires_grad_(False)
                self.__dict__["_reca_longclip_runtime"] = (
                    clip_model,
                    clip_processor,
                )
            else:
                clip_model, clip_processor = cached_clip
        if enable_taylorseer:
            self.language_model.model.enable_taylorseer = True
            model_pred_cache_dic, model_pred_current = cache_init(self, num_timesteps)
            model_pred_text_cache_dic, model_pred_text_current = cache_init(self, num_timesteps)
            model_pred_img_cache_dic, model_pred_img_current = cache_init(self, num_timesteps)
        else:
            self.language_model.model.enable_taylorseer = False
            model_pred_cache_dic, model_pred_current = None, None
            model_pred_text_cache_dic, model_pred_text_current = None, None
            model_pred_img_cache_dic, model_pred_img_current = None, None
    
        x_t = packed_init_noises.clone().requires_grad_().cpu()

        timesteps = torch.linspace(1, 0, num_timesteps, device=x_t.device)
        timesteps = timestep_shift * timesteps / (1 + (timestep_shift - 1) * timesteps)
        dts =  timesteps[:-1] - timesteps[1:]
        timesteps = timesteps[:-1]
        # breakpoint()
        def infer_one_step(init_x_t, t):
            timestep = torch.tensor([t] * init_x_t.shape[0], device=init_x_t.device)
            if t > cfg_interval[0] and t <= cfg_interval[1]:
                cfg_text_scale_ = cfg_text_scale
                cfg_img_scale_ = cfg_img_scale
            else:
                cfg_text_scale_ = 1.0
                cfg_img_scale_ = 1.0

            v_t = self._forward_flow(
                x_t=init_x_t,
                timestep=timestep, 
                packed_vae_token_indexes=packed_vae_token_indexes,
                packed_vae_position_ids=packed_vae_position_ids,
                packed_text_ids=packed_text_ids,
                packed_text_indexes=packed_text_indexes,
                packed_position_ids=packed_position_ids,
                packed_indexes=packed_indexes,
                packed_seqlens=packed_seqlens,
                key_values_lens=key_values_lens,
                past_key_values=past_key_values,
                packed_key_value_indexes=packed_key_value_indexes,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                # cfg_text
                cfg_text_scale=cfg_text_scale_,
                cfg_text_packed_position_ids=cfg_text_packed_position_ids,
                cfg_text_packed_query_indexes=cfg_text_packed_query_indexes,
                cfg_text_key_values_lens=cfg_text_key_values_lens,
                cfg_text_past_key_values=cfg_text_past_key_values,
                cfg_text_packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                # cfg_img
                cfg_img_scale=cfg_img_scale_,
                cfg_img_packed_position_ids=cfg_img_packed_position_ids,
                cfg_img_packed_query_indexes=cfg_img_packed_query_indexes,
                cfg_img_key_values_lens=cfg_img_key_values_lens,
                cfg_img_past_key_values=cfg_img_past_key_values,
                cfg_img_packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                cfg_type=cfg_type,
                # cache
                model_pred_cache_dic=model_pred_cache_dic,
                model_pred_current=model_pred_current,
                model_pred_text_cache_dic=model_pred_text_cache_dic,
                model_pred_text_current=model_pred_text_current,
                model_pred_img_cache_dic=model_pred_img_cache_dic,
                model_pred_img_current=model_pred_img_current
            )
            return v_t

        extra_p = None
        dropbp_profile_path = "./debug/umm_dropbp_profile.json"
        recalibrate_dropbp = os.environ.get(
            "BAGEL_RECALIBRATE_DROPBP",
            "0",
        ) == "1"
        reuse_reca_grad = os.environ.get(
            "BAGEL_REUSE_RECA_GRAD",
            "0",
        ) == "1" and not hybrid_reca_guidance
        enable_dropbp = os.environ.get("BAGEL_ENABLE_DROPBP", "0") == "1"
        if reuse_reca_grad:
            print("ReCA gradient reuse: enabled within each timestep")
        umm_dropbp_calibrated = (
            use_understanding_ce_loss
            and not internal_feature_guidance
            and enable_dropbp
            and not recalibrate_dropbp
        )
        umm_dropbp_layers = (0, 1, 2, 3) if umm_dropbp_calibrated else ()
        umm_dropbp_grad_scale = 1.37271 if umm_dropbp_calibrated else 1.0
        if umm_dropbp_calibrated and os.path.exists(dropbp_profile_path):
            try:
                with open(dropbp_profile_path, "r", encoding="utf-8") as profile_file:
                    dropbp_profile = json.load(profile_file)
                umm_dropbp_layers = tuple(dropbp_profile["layers"])
                umm_dropbp_grad_scale = float(dropbp_profile["grad_scale"])
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                print("DropBP profile invalid; using built-in calibrated profile")
        if umm_dropbp_calibrated:
            print(
                f"DropBP loaded layers={umm_dropbp_layers}, "
                f"grad_scale={umm_dropbp_grad_scale:.6g}"
            )
        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
            # breakpoint()
            timestep = torch.tensor([t] * x_t.shape[0], device=x_t.device)
            if t > cfg_interval[0] and t <= cfg_interval[1]:
                cfg_text_scale_ = cfg_text_scale
                cfg_img_scale_ = cfg_img_scale
            else:
                cfg_text_scale_ = 1.0
                cfg_img_scale_ = 1.0

            update_flag = False
            for item in update_lr:
                if i >= item[0] and i <= item[1]:
                    update_flag = True
                    break
            
            re_update_num_ = re_update_num if update_flag else 0
            if (
                use_understanding_ce_loss
                and update_flag
                and (i - first_update_step) % native_ce_anchor_stride != 0
            ):
                re_update_num_ = 0
            if hybrid_reca_guidance and update_flag:
                re_update_num_ = 1
            loss_list = []
            flag = False

            x_t_list = []
            x_t_score = []
            x_t_loss= []
            final_select_unmodified_next = None
            in_step_gss_candidates_next = []
            for re_update in range(re_update_num_ + 1):
                final_flag = True
                if update_flag and re_update < re_update_num_:
                    final_flag = False

                # The first guidance gradient is reused at the same timestep.
                # Those updates need neither another flow forward nor another loss pass.
                if reuse_reca_grad and not final_flag and re_update > 0:
                    continue

                track_generator_grad = not final_flag
                with torch.set_grad_enabled(track_generator_grad):
                    tt = time.time()
                    v_t = self._forward_flow_reca(
                        x_t=x_t,
                        timestep=timestep, 
                        packed_vae_token_indexes=packed_vae_token_indexes,
                        packed_vae_position_ids=packed_vae_position_ids,
                        packed_text_ids=packed_text_ids,
                        packed_text_indexes=packed_text_indexes,
                        packed_position_ids=packed_position_ids,
                        packed_indexes=packed_indexes,
                        packed_seqlens=packed_seqlens,
                        key_values_lens=key_values_lens,
                        past_key_values=past_key_values,
                        packed_key_value_indexes=packed_key_value_indexes,
                        cfg_renorm_min=cfg_renorm_min,
                        cfg_renorm_type=cfg_renorm_type,
                        # cfg_text
                        cfg_text_scale=cfg_text_scale_,
                        cfg_text_packed_position_ids=cfg_text_packed_position_ids,
                        cfg_text_packed_query_indexes=cfg_text_packed_query_indexes,
                        cfg_text_key_values_lens=cfg_text_key_values_lens,
                        cfg_text_past_key_values=cfg_text_past_key_values,
                        cfg_text_packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                        # cfg_img
                        cfg_img_scale=cfg_img_scale_,
                        cfg_img_packed_position_ids=cfg_img_packed_position_ids,
                        cfg_img_packed_query_indexes=cfg_img_packed_query_indexes,
                        cfg_img_key_values_lens=cfg_img_key_values_lens,
                        cfg_img_past_key_values=cfg_img_past_key_values,
                        cfg_img_packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                        cfg_type=cfg_type,
                        # cache
                        model_pred_cache_dic=model_pred_cache_dic,
                        model_pred_current=model_pred_current,
                        model_pred_text_cache_dic=model_pred_text_cache_dic,
                        model_pred_text_current=model_pred_text_current,
                        model_pred_img_cache_dic=model_pred_img_cache_dic,
                        model_pred_img_current=model_pred_img_current,
                        final_flag=final_flag,
                    )
                    ttt = time.time()
                # breakpoint()
                x_t_1 = x_t - v_t.to(x_t.device) * dts[i].cpu()
                if in_step_user_ce_gss and update_flag:
                    in_step_gss_candidates_next.append(x_t_1.detach())
                if (
                    final_ideal_ce_select
                    and i == last_update_step
                    and re_update == 0
                ):
                    # Candidate 0 is the ordinary denoising transition before
                    # applying the rectification gradient at the final update
                    # timestep. Candidate K is x_t_1 after all updates below.
                    final_select_unmodified_next = x_t_1.detach()

                if use_lookback and update_flag:
                    with torch.no_grad():
                        x_t_lookback = x_t.detach()
                        x_t_0_lookback = x_t_lookback - v_t.to(x_t.device) * t
                        for ii, tt in tqdm(enumerate(timesteps[i:lookback_steps]), total=len(timesteps[i:lookback_steps])):
                            v_t_lookback = infer_one_step(x_t_lookback, tt)
                            x_t_lookback = x_t_lookback - v_t_lookback.to(x_t.device) * dts[i + ii].cpu()
                            x_t_0_lookback = x_t_lookback - v_t_lookback.to(x_t.device) * tt
                        x_t_0_latent = x_t_0_lookback.split((packed_seqlens - 2).tolist())
                    with torch.no_grad():
                        if use_understanding_ce_loss:
                            l, v = self.UnderstandingCELoss(
                                x_t_0=x_t_0_lookback,
                                target_text=prompt,
                                vae_model=vae_model,
                                image_shape=image_shapes,
                                image_transform=image_transform,
                                tokenizer=tokenizer,
                                new_token_ids=new_token_ids,
                                ce_max_tokens=ce_max_tokens,
                                ce_vit_max_side=168,
                                umm_dropbp_layers=(),
                            )
                        else:
                            l, v = self.calc_clip_with_prompt_nograd(
                                prompt,
                                x_t_0_latent,
                                vae_model,
                                image_shapes,
                                tokenizer,
                                new_token_ids,
                                image_transform,
                                clip_model,
                                clip_processor,
                                use_longclip,
                                use_fgclip,
                            )

                    x_t_list.append(x_t_1.detach().requires_grad_())
                    x_t_loss.append(l.item())
                    # x_t_score.append(vlm_output_text)
                
                if not final_flag:
                    x_t_0 = x_t - v_t.to(x_t.device) * t
                    x_t_0_latent = x_t_0.split((packed_seqlens - 2).tolist())
                    t1 = time.time()
                    refresh_ideal_each_timestep = os.environ.get(
                        "BAGEL_REFRESH_IDEAL_EACH_TIMESTEP", "0"
                    ) == "1"
                    refresh_ideal_each_update = os.environ.get(
                        "BAGEL_REFRESH_IDEAL_EACH_UPDATE", "0"
                    ) == "1"
                    if refresh_ideal_each_update or (
                        refresh_ideal_each_timestep and re_update == 0
                    ):
                        extra_p = None
                    if extra_p is None:
                        ideal_img = self._decode_reca_latent_to_pil(
                            x_t_0_latent[0].detach(),
                            image_shapes,
                            vae_model,
                        )
                        if os.environ.get("BAGEL_SAVE_RECA_DEBUG_IMAGE", "0") == "1":
                            os.makedirs("./debug", exist_ok=True)
                            ideal_img.save("./debug/debug_middle.png")
                        ideal_prompt_style = os.environ.get(
                            "BAGEL_IDEAL_PROMPT_STYLE", "legacy"
                        )
                        if ideal_prompt_style == "user_only":
                            extra_p = prompt
                        elif ideal_prompt_style == "delta_stable_target_v14":
                            ideal_instruction = (
                                f"User request: '{prompt}'. Look at the intermediate image and produce a "
                                "stable corrective target for Native CE guidance. Output exactly one line "
                                "starting with 'TARGET:'. After 'TARGET:', write one positive visual target "
                                "sentence. It should keep all user-required objects, counts, colors, "
                                "attributes, and spatial relations. If a required part is weak or ambiguous "
                                "in the image, make that requirement more visually explicit: counts should "
                                "be clearly separated; requested colors should cover the main visible body "
                                "or surface; requested relations should be obvious and unambiguous; objects "
                                "should have recognizable shape or object-defining visible parts. You may "
                                "keep harmless scene context from the image if it helps visual grounding. "
                                "Do not add new salient objects or contradict the user request. Do not list "
                                "problems. Do not use words like no, not, missing, unclear, wrong, issue, or "
                                "mistake. Output only 'TARGET: ...' under 40 words."
                            )
                            ideal_max_length = 96
                        elif ideal_prompt_style == "delta_unclear_terms_general_v13":
                            ideal_instruction = (
                                f"User request: '{prompt}'. Inspect the intermediate image and identify "
                                "only the user-required visual constraints that are missing, ambiguous, "
                                "weak, or not clearly visible. Output positive corrective terms for those "
                                "unclear requirements, not a full image caption. Use only concepts from "
                                "the user request plus generic visual evidence needed to make the required "
                                "objects recognizable. For object identity, mention characteristic visible "
                                "shape, main body, or object-defining parts without naming category-specific "
                                "examples. For counts, require exactly the requested number of clearly "
                                "separated distinct instances. For colors or attributes, require the "
                                "specified object to show the requested attribute clearly on its main "
                                "visible body or surface, not merely a tiny region. For spatial relations, "
                                "require the specified relation to be visually obvious and unambiguous. "
                                "If all user requirements are already clear, output the most important "
                                "requested constraint in concise positive form. Do not add unrequested "
                                "objects, counts, colors, attributes, materials, relations, background, "
                                "lighting, camera, style, critique, mistakes, or negation. Output only "
                                "semicolon-separated terms under 18 words."
                            )
                            ideal_max_length = 56
                        elif ideal_prompt_style == "delta_unclear_terms_v12":
                            ideal_instruction = (
                                f"User request: '{prompt}'. Inspect the intermediate image and identify "
                                "only the user-required visual constraints that are missing, ambiguous, "
                                "weak, or not clearly visible. Output positive corrective description "
                                "terms for those unclear requirements, not a full image caption and not a "
                                "critique. Use only concepts from the user request, plus minimal object-"
                                "recognition parts when they make the requested object clearer. For count "
                                "issues, write exactly N clearly separated distinct [objects]. For color "
                                "issues, write predominantly [color] [object] main body/surface. For "
                                "spatial issues, write [object A] clearly [left of/right of/above/below] "
                                "[object B] with unambiguous placement. For object identity issues, add "
                                "essential parts only if helpful, such as wheels for skateboard, wings for "
                                "airplane, handle for handbag, buttons for remote, faucet/basin for sink, "
                                "bowl/tank for toilet, blade for knife, ring for donut, sign face for stop "
                                "sign, boards for skis or snowboards. Do not mention requirements already "
                                "clear unless all requirements are clear; then output the single most "
                                "important user constraint in concise positive form. Do not add unrequested "
                                "colors, numbers, objects, relations, materials, lighting, background, "
                                "style, critique, mistakes, or negation. Output only semicolon-separated "
                                "terms under 20 words."
                            )
                            ideal_max_length = 64
                        elif ideal_prompt_style == "delta_binding_checklist_v11":
                            ideal_instruction = (
                                f"User request: '{prompt}'. Look at the intermediate image and rewrite "
                                "the request into one compact positive visual checklist for CE guidance. "
                                "Do not write a scene caption. Use semicolon-separated constraints. Keep "
                                "only user-requested objects, exact counts, requested colors/attributes, "
                                "requested spatial relations, and minimal recognition parts. For multiple "
                                "objects with colors, bind each color to its object separately, e.g. "
                                "red skis; brown tie; both recognizable. For counts, write exactly N "
                                "separate recognizable objects. For relations, write object A clearly left "
                                "of/above/below object B. For object identity, add only essential parts if "
                                "useful: wings, wheels, handle, buttons, faucet, basin, bowl, tank, blade, "
                                "ring, sign face, boards. Do not add unrequested colors, numbers, objects, "
                                "materials, lighting, background, style, critique, mistakes, or negation. "
                                "Output only the checklist under 22 words."
                            )
                            ideal_max_length = 64
                        elif ideal_prompt_style == "delta_short_constraints_v9":
                            ideal_instruction = (
                                f"User request: '{prompt}'. Look at the intermediate image only to make "
                                "the request visually enforceable. Output one short positive constraint "
                                "phrase, not a full scene description. Use the user's object names and "
                                "relations as much as possible. For count prompts, write: exactly N clearly "
                                "separated [objects]. For color prompts, write: predominantly [color] "
                                "[object] with recognizable object parts. For relation prompts, write the "
                                "exact requested relation, e.g. [object A] clearly left of [object B], with "
                                "unambiguous placement. For two-object color prompts, write both requested "
                                "objects with their requested colors and key recognition parts if helpful. "
                                "Do not add unrequested colors, numbers, objects, materials, lighting, "
                                "background, camera, style, critique, mistakes, or negation. Output only "
                                "the phrase under 16 words."
                            )
                            ideal_max_length = 48
                        elif ideal_prompt_style == "delta_short_user_anchored_v10":
                            ideal_instruction = (
                                f"User request: '{prompt}'. Look at the intermediate image and write one "
                                "very short positive corrective phrase that reinforces only the user's "
                                "explicit visual requirements. Keep exact counts, requested colors, "
                                "attribute bindings, and spatial relations. Add object-recognition parts "
                                "only when useful, such as wheels for skateboard, wings for airplane, "
                                "handle for handbag, keys/buttons for remote, faucet/basin for sink, "
                                "bowl/tank for toilet, blade for knife, ring for donut, sign face for stop "
                                "sign, boards for skis or snowboards. Do not add unrequested colors, "
                                "numbers, objects, relations, materials, lighting, background, style, "
                                "critique, mistakes, or negation. Output only the phrase under 14 words."
                            )
                            ideal_max_length = 48
                        elif ideal_prompt_style == "delta_become_requested_v26":
                            ideal_instruction = (
                                f"User request: '{prompt}'. If the draft image should become the "
                                "requested image, what should be changed or made clearer? Write one "
                                "positive sentence describing the corrected image after those changes. "
                                "Keep the requested object identity, count, color, attribute, and spatial "
                                "relation. Use the draft only for harmless object parts or simple layout "
                                "that do not conflict with the request. Output only one natural sentence "
                                "under 35 words."
                            )
                            ideal_max_length = 96
                        elif ideal_prompt_style == "twostage_edit_target_v28":
                            stage1_instruction = (
                                f"User request: '{prompt}'. The image is a draft and may contain "
                                "incorrect details. If this draft should become the requested image, "
                                "what should be changed or made clearer? Answer with only the needed "
                                "visual changes. Keep the requested object identity, count, color, "
                                "attribute, and spatial relation from the user request. Use the draft "
                                "only to refer to harmless object parts or simple layout. Output one "
                                "concise sentence under 35 words."
                            )
                            edit_delta = self.chat(
                                tokenizer=tokenizer,
                                new_token_ids=new_token_ids,
                                image_transform=image_transform,
                                images=[ideal_img],
                                prompt=stage1_instruction,
                                max_length=96,
                                do_sample=False,
                            ).strip()
                            stage2_instruction = (
                                f"User request: '{prompt}'. Draft-to-final changes: '{edit_delta}'. "
                                "Now describe the final image after applying those changes. Describe "
                                "the final image itself, not the editing process. The description must "
                                "follow the user request and the listed changes. Keep only helpful "
                                "object parts or simple layout from the draft if they do not conflict. "
                                "Output only one positive natural sentence under 35 words."
                            )
                            print(f"twostage edit delta: {edit_delta}")
                            ideal_instruction = stage2_instruction
                            ideal_max_length = 96
                        elif ideal_prompt_style == "reflection_constraint_target_v30":
                            ideal_instruction = (
                                f"User request: '{prompt}'. Write a CE target, not a caption. Output one "
                                "positive sentence that states the exact visual constraints the final "
                                "image must satisfy. Begin with the requested object names. Preserve all "
                                "requested counts, colors, attributes, and spatial relations. Use the image "
                                "only to add object-recognition parts for requested objects, such as shape, "
                                "body, handle, wheels, faucet, tank, blade, leaves, or sign face. Do not "
                                "mention sharpness, focus, lighting, background, surface color, text labels, "
                                "mistakes, or negation. Do not add unrequested objects, colors, counts, or "
                                "relations. Output under 30 words."
                            )
                            ideal_max_length = 80
                        elif ideal_prompt_style == "reflection_user_rewrite_parts_v31":
                            ideal_instruction = (
                                f"User request: '{prompt}'. Rewrite the request as one concrete final-image "
                                "description for CE guidance. Keep the same requested objects, counts, "
                                "colors, attributes, and relations. If the image suggests useful parts of "
                                "a requested object, include only those parts to make the object recognizable. "
                                "For color requests, say the requested object's main body is the requested "
                                "color. For counts, say exactly N separate instances. For relations, say "
                                "the requested relation is obvious. Avoid all background, lighting, quality, "
                                "brand/text, critique, and negation. Output one sentence under 32 words."
                            )
                            ideal_max_length = 80
                        elif ideal_prompt_style == "reflection_evidence_target_v32":
                            ideal_instruction = (
                                f"User request: '{prompt}'. Look at the draft and infer what visual evidence "
                                "would make the user request unmistakably true. Output that evidence as one "
                                "positive target sentence. Include the requested object identities, exact "
                                "counts, requested colors or attributes, requested spatial relations, and "
                                "only necessary object-recognition parts. Do not describe the draft, do not "
                                "say what is wrong, and do not add background, lighting, text, new colors, "
                                "new objects, or new relations. Output under 34 words."
                            )
                            ideal_max_length = 88
                        elif ideal_prompt_style == "reflection_minimal_binding_v33":
                            ideal_instruction = (
                                f"User request: '{prompt}'. Output only a compact positive visual constraint "
                                "phrase for CE. Keep user-requested object names and bind each requested "
                                "color, count, attribute, and relation to the correct object. Add at most "
                                "one object-recognition part if needed. No background, no style, no critique, "
                                "no negation. Under 22 words."
                            )
                            ideal_max_length = 64
                        elif ideal_prompt_style == "reflection_twostage_constraints_v34":
                            stage1_instruction = (
                                f"User request: '{prompt}'. Look at the draft image only to identify "
                                "which user-requested visual constraints need reinforcement for the "
                                "final image. Think in terms of object identity, exact count, requested "
                                "color or attribute binding, and requested spatial relation. Do not "
                                "mention image quality, sharpness, focus, lighting, background, camera, "
                                "style, text labels, mistakes, or negation. Output only short positive "
                                "constraint notes using concepts from the user request plus necessary "
                                "object-recognition parts."
                            )
                            edit_delta = self.chat(
                                tokenizer=tokenizer,
                                new_token_ids=new_token_ids,
                                image_transform=image_transform,
                                images=[ideal_img],
                                prompt=stage1_instruction,
                                max_length=80,
                                do_sample=False,
                            ).strip()
                            stage2_instruction = (
                                f"User request: '{prompt}'. Constraint notes: '{edit_delta}'. "
                                "Convert these notes into one positive CE target sentence for the "
                                "correct final image. Start with the requested object names. Preserve "
                                "every requested count, color, attribute, and spatial relation. Include "
                                "only necessary object-recognition parts from the notes. Do not mention "
                                "quality, sharpness, focus, lighting, background, text labels, mistakes, "
                                "or negation. Do not add unrequested objects, colors, counts, attributes, "
                                "or relations. Output under 30 words."
                            )
                            print(f"twostage constraint notes: {edit_delta}")
                            ideal_instruction = stage2_instruction
                            ideal_max_length = 80
                        elif ideal_prompt_style == "reflection_count_anchor_v35":
                            ideal_instruction = (
                                f"User request: '{prompt}'. The draft image may be wrong. Write one "
                                "positive corrected-final-image sentence for Native CE. The user request "
                                "is absolute. If the request contains a number, the sentence must state "
                                "exactly that number using the form 'exactly N separate distinct [object]'. "
                                "Never mention any other number from the draft. Preserve requested object "
                                "identity, colors, attributes, and spatial relations. Use the draft only "
                                "for harmless layout or object-recognition parts that do not conflict. "
                                "Do not describe mistakes, do not say should, missing, wrong, no, or not. "
                                "Output only one natural sentence under 28 words."
                            )
                            ideal_max_length = 72
                        elif ideal_prompt_style == "reflection_count_anchor_plus_user_v36":
                            anchor_instruction = (
                                f"User request: '{prompt}'. The draft image may be wrong. Write one "
                                "positive corrected-final-image sentence for image updating. The user request "
                                "is absolute. If the request contains a number, the sentence must state "
                                "exactly that number using the form 'exactly N separate distinct [object]'. "
                                "Never mention any other number from the draft. Preserve requested object "
                                "identity, colors, attributes, and spatial relations. Use the draft only "
                                "for harmless layout or object-recognition parts that do not conflict. "
                                "Do not describe mistakes, do not say should, missing, wrong, no, or not. "
                                "Output only one natural sentence under 28 words."
                            )
                            anchor_target = self.chat(
                                tokenizer=tokenizer,
                                new_token_ids=new_token_ids,
                                image_transform=image_transform,
                                images=[ideal_img],
                                prompt=anchor_instruction,
                                max_length=72,
                                do_sample=False,
                            ).strip()
                            print(f"count anchor target: {anchor_target}")
                            extra_p = f"{prompt}. {anchor_target}"
                            ideal_instruction = None
                            ideal_max_length = 0
                        elif ideal_prompt_style == "reflection_general_target_v37":
                            ideal_instruction = (
                                f"User request: '{prompt}'. Look at the draft image and write one positive "
                                "description of the corrected final image. The user request is the authority. "
                                "Make weak requested constraints visually clear: exact counts as exactly N "
                                "separate distinct objects, requested colors on the object's main visible body, "
                                "requested relations with obvious placement, and recognizable object parts. "
                                "You may keep harmless visible layout/context only when it helps the requested "
                                "objects. Never mention a count, color, object, attribute, or relation that "
                                "contradicts the user request. Do not describe mistakes, editing, should, missing, "
                                "wrong, no, or not. Output exactly one natural sentence under 40 words."
                            )
                            ideal_max_length = 96
                        elif ideal_prompt_style == "reflection_general_target_plus_user_v38":
                            anchor_instruction = (
                                f"User request: '{prompt}'. Look at the draft image and write one positive "
                                "description of the corrected final image. The user request is the authority. "
                                "Make weak requested constraints visually clear: exact counts as exactly N "
                                "separate distinct objects, requested colors on the object's main visible body, "
                                "requested relations with obvious placement, and recognizable object parts. "
                                "You may keep harmless visible layout/context only when it helps the requested "
                                "objects. Never mention a count, color, object, attribute, or relation that "
                                "contradicts the user request. Do not describe mistakes, editing, should, missing, "
                                "wrong, no, or not. Output exactly one natural sentence under 40 words."
                            )
                            anchor_target = self.chat(
                                tokenizer=tokenizer,
                                new_token_ids=new_token_ids,
                                image_transform=image_transform,
                                images=[ideal_img],
                                prompt=anchor_instruction,
                                max_length=96,
                                do_sample=False,
                            ).strip()
                            print(f"general reflection target: {anchor_target}")
                            extra_p = f"{prompt}. {anchor_target}"
                            ideal_instruction = None
                            ideal_max_length = 0
                        elif ideal_prompt_style == "reflection_visibility_amplify_v39":
                            ideal_instruction = (
                                f"User request: '{prompt}'. The draft image may satisfy the request only weakly or unclearly. "
                                "Write one positive target description of the final image that makes the user request unmistakably visible. "
                                "Follow the user request exactly. If a count is requested, say exactly N clearly separated, non-overlapping, "
                                "recognizable instances of the requested object. If an object may be unclear, include generic object-defining "
                                "visible parts, such as body, handle, wheels, basin, faucet, keys, blade, bowl, tank, pole, sign face, or board, "
                                "only when they help recognition. If color, attribute, or spatial relation is requested, make it dominant, obvious, "
                                "and bound to the correct object. Use harmless visible context only if it helps grounding. Never contradict the user request. "
                                "Do not describe errors, editing, uncertainty, should, missing, wrong, no, or not. Output one natural sentence under 45 words."
                            )
                            ideal_max_length = 112
                        elif ideal_prompt_style == "reflection_visibility_amplify_plus_user_v40":
                            anchor_instruction = (
                                f"User request: '{prompt}'. The draft image may satisfy the request only weakly or unclearly. "
                                "Write one positive target description of the final image that makes the user request unmistakably visible. "
                                "Follow the user request exactly. If a count is requested, say exactly N clearly separated, non-overlapping, "
                                "recognizable instances of the requested object. If an object may be unclear, include generic object-defining "
                                "visible parts, such as body, handle, wheels, basin, faucet, keys, blade, bowl, tank, pole, sign face, or board, "
                                "only when they help recognition. If color, attribute, or spatial relation is requested, make it dominant, obvious, "
                                "and bound to the correct object. Use harmless visible context only if it helps grounding. Never contradict the user request. "
                                "Do not describe errors, editing, uncertainty, should, missing, wrong, no, or not. Output one natural sentence under 45 words."
                            )
                            anchor_target = self.chat(
                                tokenizer=tokenizer,
                                new_token_ids=new_token_ids,
                                image_transform=image_transform,
                                images=[ideal_img],
                                prompt=anchor_instruction,
                                max_length=112,
                                do_sample=False,
                            ).strip()
                            print(f"visibility reflection target: {anchor_target}")
                            extra_p = f"{prompt}. {anchor_target}"
                            ideal_instruction = None
                            ideal_max_length = 0
                        elif ideal_prompt_style == "fixed_env_target":
                            fixed_target = os.environ.get("BAGEL_FIXED_IDEAL_TARGET", "").strip()
                            if fixed_target:
                                extra_p = f"{prompt}. {fixed_target}"
                            else:
                                extra_p = prompt
                            print(f"fixed env ideal prompt: {extra_p}")
                            ideal_instruction = None
                            ideal_max_length = 0
                        elif ideal_prompt_style == "reflection_constraint_first_then_evidence_v45":
                            anchor_instruction = (
                                f"User request: '{prompt}'. Write one CE supervision target for the corrected final image. "
                                "The first clause must restate the user request exactly as a positive visual constraint, using the same requested object names, count words, color words, attributes, and spatial relation words. "
                                "Do not infer the relation from the draft image. Do not change below into on, above, inside, next to, or in front of. Do not change above, left of, right of, under, behind, or in front of into another relation. "
                                "After that first clause, you may add a short second clause with harmless visible evidence from the draft that makes only the requested objects recognizable, such as shape, body, handle, wheel, strap, blade, faucet, bowl, or sign face. "
                                "If the draft conflicts with the request, ignore the conflicting detail. Do not mention errors, editing, missing, wrong, should, no, or not. Output one natural sentence under 45 words."
                            )
                            anchor_target = self.chat(
                                tokenizer=tokenizer,
                                new_token_ids=new_token_ids,
                                image_transform=image_transform,
                                images=[ideal_img],
                                prompt=anchor_instruction,
                                max_length=112,
                                do_sample=False,
                            ).strip()
                            print(f"constraint-first reflection target: {anchor_target}")
                            extra_p = f"{prompt}. {anchor_target}"
                            ideal_instruction = None
                            ideal_max_length = 0
                        elif ideal_prompt_style == "reflection_bound_attributes_v47":
                            anchor_instruction = (
                                f"User request: '{prompt}'. Write one CE supervision target for the corrected final image. "
                                "First restate the requested objects and constraints from the user request as the main clause. "
                                "Bind every requested color or attribute to the requested object itself: say the object's main visible body or surface has that color or attribute. "
                                "Do not mention background, wall, floor, table, lighting, shadows, scene color, or surface color when the request contains a color. "
                                "If a count is requested, say exactly the requested number of clearly separated recognizable instances. "
                                "If a spatial relation is requested, use only the exact requested relation words and make the relation obvious; do not replace them with another relation. "
                                "If no spatial relation is requested, do not invent left, right, above, below, on, under, in front of, behind, beside, or next to. "
                                "Add only minimal object-recognition evidence for requested objects, such as shape, body, handle, wheel, strap, blade, faucet, bowl, leaves, sign face, screen, or keys. "
                                "Ignore draft details that conflict with the user request. Do not mention errors, editing, missing, wrong, should, no, or not. Output one natural sentence under 42 words."
                            )
                            anchor_target = self.chat(
                                tokenizer=tokenizer,
                                new_token_ids=new_token_ids,
                                image_transform=image_transform,
                                images=[ideal_img],
                                prompt=anchor_instruction,
                                max_length=112,
                                do_sample=False,
                            ).strip()
                            print(f"bound-attribute reflection target: {anchor_target}")
                            extra_p = f"{prompt}. {anchor_target}"
                            ideal_instruction = None
                            ideal_max_length = 0
                        elif ideal_prompt_style == "reflection_generic_evidence_v41":
                            ideal_instruction = (
                                f"User request: '{prompt}'. Look at the draft image as a weak attempt at the request. "
                                "Write one positive target description for the corrected final image. The target should make the user request "
                                "unmistakably true and visually easy to verify. Preserve all requested object identities, exact counts, colors, "
                                "attributes, and spatial relations. If a requested object is unclear, add the minimal object-defining visual "
                                "evidence that would make it recognizable, without naming unrelated objects. If a count is requested, make the "
                                "instances clearly separated and individually recognizable. If a color or attribute is requested, bind it to the "
                                "correct object's main visible region. If a spatial relation is requested, make the placement obvious. Use harmless "
                                "visible context only if it helps grounding. Do not mention errors, editing, uncertainty, should, missing, wrong, no, "
                                "or not. Do not contradict the user request. Output one natural sentence under 45 words."
                            )
                            ideal_max_length = 112
                        elif ideal_prompt_style == "reflection_generic_evidence_plus_user_v42":
                            anchor_instruction = (
                                f"User request: '{prompt}'. Look at the draft image as a weak attempt at the request. "
                                "Write one positive target description for the corrected final image. The target should make the user request "
                                "unmistakably true and visually easy to verify. Preserve all requested object identities, exact counts, colors, "
                                "attributes, and spatial relations. If a requested object is unclear, add the minimal object-defining visual "
                                "evidence that would make it recognizable, without naming unrelated objects. If a count is requested, make the "
                                "instances clearly separated and individually recognizable. If a color or attribute is requested, bind it to the "
                                "correct object's main visible region. If a spatial relation is requested, make the placement obvious. Use harmless "
                                "visible context only if it helps grounding. Do not mention errors, editing, uncertainty, should, missing, wrong, no, "
                                "or not. Do not contradict the user request. Output one natural sentence under 45 words."
                            )
                            anchor_target = self.chat(
                                tokenizer=tokenizer,
                                new_token_ids=new_token_ids,
                                image_transform=image_transform,
                                images=[ideal_img],
                                prompt=anchor_instruction,
                                max_length=112,
                                do_sample=False,
                            ).strip()
                            print(f"generic evidence reflection target: {anchor_target}")
                            extra_p = f"{prompt}. {anchor_target}"
                            ideal_instruction = None
                            ideal_max_length = 0
                        elif ideal_prompt_style == "reflection_constraint_bound_v29":
                            ideal_instruction = (
                                f"User request: '{prompt}'. Look at the intermediate draft image and "
                                "write one positive corrective target sentence for Native CE. The "
                                "sentence must be a corrected final-image target, not a critique and "
                                "not a general caption. Start with the user-requested objects and bind "
                                "every requested count, color, attribute, and spatial relation to the "
                                "correct object. Use the draft only to add object-recognition cues or "
                                "simple non-conflicting layout that help the requested objects remain "
                                "recognizable. For unusual colors, keep the object identity explicit by "
                                "mentioning shape or parts plus the requested color on the main body. "
                                "For counts, require exactly the requested number of separate distinct "
                                "instances. For relations, make the requested relation visually obvious. "
                                "Do not mention sharpness, focus, lighting, camera, mood, background, "
                                "text labels, brand words, mistakes, or negation. Do not introduce any "
                                "unrequested object, count, color, attribute, or relation. Output only "
                                "one natural sentence under 34 words."
                            )
                            ideal_max_length = 88
                        elif ideal_prompt_style == "delta_corrected_final_v21":
                            ideal_instruction = (
                                f"User request: '{prompt}'. The image may contain mistakes. Do not "
                                "describe the current image. Instead, write the corrected final image "
                                "that should replace it. The user request is the authority. If the "
                                "current image has too many requested objects, write only or exactly "
                                "the requested number. If the requested color is weak or wrong, write "
                                "that the requested object is clearly and mostly the requested color. "
                                "If a requested relation is weak or wrong, write that relation as clear "
                                "and unambiguous. Preserve every requested object, count, color, "
                                "attribute, and spatial relation. You may add harmless scene context "
                                "only if it supports the corrected target. Do not copy wrong counts, "
                                "wrong colors, wrong objects, or wrong relations from the image. Do not "
                                "explain, critique, or mention mistakes. Output one positive natural "
                                "sentence under 40 words."
                            )
                            ideal_max_length = 96
                        elif ideal_prompt_style == "delta_corrected_plain_v19":
                            ideal_instruction = (
                                f"User request: '{prompt}'. The image may be imperfect. Write one "
                                "sentence describing the corrected final image that should be shown. "
                                "Follow the user request over the image. Preserve every requested "
                                "object, exact count, color, attribute, and spatial relation. If the "
                                "image conflicts with the request, describe the requested version "
                                "instead. Use exactly or only for counts, mostly and clearly for "
                                "requested colors, obvious for spatial relations, and include simple "
                                "recognizable object parts when helpful. Add minimal scene context only "
                                "if useful. Output only one positive natural sentence under 40 words."
                            )
                            ideal_max_length = 96
                        elif ideal_prompt_style == "delta_strict_short_v20":
                            ideal_instruction = (
                                f"User request: '{prompt}'. The current image may be wrong. Write the "
                                "correct final image in one sentence. Start from the requested objects "
                                "and constraints, not from the current image. Keep exact counts, colors, "
                                "attributes, and relations from the request. Say exactly or only for "
                                "counts, clearly and mostly for colors, and obvious for relations. Add "
                                "only essential recognizable object parts. Output one positive sentence "
                                "under 32 words."
                            )
                            ideal_max_length = 80
                        elif ideal_prompt_style == "delta_visual_constraint_v8":
                            ideal_instruction = (
                                f"User request: '{prompt}'. Look at the intermediate image, but do not "
                                "freely describe it. Rewrite the user request as one visually enforceable "
                                "positive target sentence for correcting the final image. Include only "
                                "requested objects, exact counts, requested colors/attributes, requested "
                                "spatial relations, and essential object-recognition parts. For counts, "
                                "say exactly N clearly separated distinct instances. For colors, say the "
                                "requested object is predominantly the requested color over its main "
                                "visible body/surface, not just a small mark. For spatial relations, say "
                                "the relation is visually obvious and unambiguous. For object identity, "
                                "mention essential parts only if useful, e.g. skateboard wheels, airplane "
                                "wings, handbag handle, remote buttons, sink faucet/basin, toilet bowl/tank, "
                                "knife blade, donut ring, stop-sign face, skis/snowboard boards. You may "
                                "add only one harmless support phrase if already visible: on a surface, on "
                                "a table, in a room, outdoors, or on snow. Do not invent colors for objects "
                                "whose color is not requested. Do not add new objects, materials, lighting, "
                                "camera, style, texture, quality, mood, atmosphere, background, critique, "
                                "mistakes, or negation. Output only the sentence, under 26 words."
                            )
                            ideal_max_length = 80
                        elif ideal_prompt_style == "delta_visual_context_v7":
                            ideal_instruction = (
                                f"User request: '{prompt}'. Inspect the intermediate image and write "
                                "one positive corrective target sentence for the final image. Preserve "
                                "every explicit user requirement: object identities, exact counts, "
                                "user-specified colors, attribute bindings, and required spatial "
                                "relations. If the user specifies a color, state that the requested "
                                "object is predominantly that color across its main visible body/surface, "
                                "not merely a small part. If the user specifies a count, state that there "
                                "are exactly that many clearly separated, distinct instances. If the user "
                                "specifies a spatial relation, state that it is visually obvious, with "
                                "clear separation and unambiguous left/right/above/below ordering. Keep "
                                "object-defining parts visible when useful for recognition, such as "
                                "wheels for skateboard, wings for airplane, handle for handbag, keys for "
                                "remote, faucet/bowl/tank for sink or toilet, blade for knife, circular "
                                "ring for donut, sign face for stop sign, and poles or boards for skis or "
                                "snowboards. Use the current image only for harmless grounding of requested "
                                "objects and simple support context such as on a surface, on a table, in a "
                                "room, outdoors, or on snow. If the user does not specify a color/material "
                                "for an object, do not invent one. Do not add new salient objects, extra "
                                "numbers, colors, materials, or strong spatial relations beyond the user "
                                "request. Do not mention lighting, camera, style, texture, quality, mood, "
                                "atmosphere, background, mistakes, critique, or negation. Output only one "
                                "sentence under 28 words."
                            )
                            ideal_max_length = 80
                        elif ideal_prompt_style == "delta_visual_context_v6":
                            ideal_instruction = (
                                f"User request: '{prompt}'. Inspect the intermediate image and write "
                                "one positive corrective target sentence for the final image. Preserve "
                                "every explicit user requirement: object identities, exact counts, "
                                "user-specified colors, attribute bindings, and required spatial "
                                "relations. If the user specifies a color, state that the requested "
                                "object is predominantly that color across its main visible body/surface, "
                                "not merely a small part; name concrete parts only when helpful, such as "
                                "body, surface, top, handle, screen, petals, cheese, toppings, crust, "
                                "fabric, blade, bowl, or tank. If the user specifies a count, state that "
                                "there are exactly that many clearly separated, distinct instances of the "
                                "requested object. Use the current image only for harmless grounding of "
                                "requested objects: visible non-color parts and simple support context "
                                "such as on a surface, on a table, in a room, outdoors, or on snow. If "
                                "the user does not specify a color/material for an object, do not invent "
                                "one. Do not add new salient objects, extra numbers, colors, materials, "
                                "or strong spatial relations beyond the user request. Allowed harmless "
                                "layouts are only side by side, near, beside, on a surface/table, in a "
                                "room, outdoors, or on snow, unless the user explicitly requested another "
                                "relation. Do not mention lighting, camera, style, texture, quality, mood, "
                                "atmosphere, background, mistakes, critique, or negation. Output only one "
                                "sentence under 24 words."
                            )
                            ideal_max_length = 64
                        elif ideal_prompt_style == "delta_visual_context_v5":
                            ideal_instruction = (
                                f"User request: '{prompt}'. Inspect the intermediate image and write "
                                "one positive corrective target sentence for the final image. First "
                                "preserve every explicit user requirement: object identities, exact "
                                "counts, user-specified colors, attribute bindings, and required spatial "
                                "relations. Use the image only for harmless visual grounding of requested "
                                "objects: visible object parts, handles, faucets, wheels, screens, petals, "
                                "cheese, toppings, crust, fabric, blade, bowl, tank, or simple support "
                                "context such as on a surface, on a table, in a room, outdoors, or on "
                                "snow. If the user specifies a color or attribute, make it concrete by "
                                "naming the requested object's relevant part or region when helpful. If "
                                "the user does not specify a color/material for an object, do not invent "
                                "one; say only the object name or a visible non-color part. Do not add new "
                                "salient objects, exact numbers, colors, materials, or strong spatial "
                                "relations beyond the user request. Allowed harmless layouts are only "
                                "side by side, near, beside, on a surface/table, in a room, outdoors, or "
                                "on snow, unless the user explicitly requested another relation. Do not "
                                "mention lighting, camera, style, texture, quality, mood, atmosphere, "
                                "background, mistakes, critique, or negation. Output only one sentence "
                                "under 22 words."
                            )
                            ideal_max_length = 64
                        elif ideal_prompt_style == "delta_visual_context_v4":
                            ideal_instruction = (
                                f"User request: '{prompt}'. Inspect the intermediate image and write "
                                "one positive target sentence for the final image. The sentence must "
                                "preserve all objects, counts, colors, attributes, and spatial relations "
                                "explicitly required by the user. You may also keep non-conflicting "
                                "visual evidence from the current image: stable visible colors or parts "
                                "of the requested objects; harmless natural layout among requested "
                                "objects, such as next to, beside, near, or side by side; and simple "
                                "support context, such as on a table, on a surface, in a room, or "
                                "outdoors. For requested colors or attributes, make them visually "
                                "concrete by naming object parts or regions when helpful, such as body, "
                                "surface, top, handle, screen, petals, cheese, toppings, crust, fabric, "
                                "blade, bowl, or tank. Do not add new salient objects. Do not contradict "
                                "the user-requested color, count, attribute, or relation. Do not replace "
                                "a required relation with a weaker one. Do not mention lighting, camera, "
                                "style, texture, quality, mood, atmosphere, or background. Do not "
                                "critique the current image, mention mistakes, or use negation. Output "
                                "only one sentence under 24 words."
                            )
                            ideal_max_length = 64
                        elif ideal_prompt_style in {"delta_visual_evidence_v3", "delta_visual_evidence_guarded"}:
                            ideal_instruction = (
                                f"User request: '{prompt}'. Look at the intermediate image and write "
                                "one positive target sentence for the corrected final image. The target "
                                "must preserve all user-required objects, counts, colors, attribute "
                                "bindings, and spatial relations. Use the current image only to make "
                                "attributes visually concrete: for a colored or attributed object, name "
                                "the visible parts/regions that should carry that attribute, such as "
                                "body, surface, front, top, handle, wheels, petals, cheese, toppings, "
                                "crust, fabric, blade, or screen. If the user did not specify an object's "
                                "color or material, you may keep a stable visible color/material for that "
                                "same requested object. Do not add new objects or relations. Do not "
                                "contradict any user-specified color or attribute. Do not critique, use "
                                "negation, mention mistakes, lighting, camera, style, mood, or background. "
                                "Output only one sentence, under 24 words."
                            )
                            ideal_max_length = 64
                        elif ideal_prompt_style == "delta_visual_evidence_v2":
                            ideal_instruction = (
                                f"User request: '{prompt}'. Look at the intermediate image only to "
                                "ground the user's requested attributes onto visible parts of the "
                                "requested object(s). Output one positive target sentence for the final "
                                "image. Do not critique, compare, mention mistakes, or use negation. "
                                "First preserve the exact requested objects, counts, colors, attribute "
                                "bindings, and spatial relations. If a color or attribute is requested, "
                                "state which visible parts or regions of that requested object should "
                                "show it, for example body, surface, front, top, handle, wheels, petals, "
                                "cheese, toppings, crust, fabric, blade, or screen. Mention only colors "
                                "and attributes that appear in the user request. Do not assign any color "
                                "or attribute to an object unless the user requested it. You may add only "
                                "plain support context, such as on a table or on a surface, if it is "
                                "already visible and not a new object. No lighting, background, camera, "
                                "style, material, texture, quality, or mood. Output only one sentence, "
                                "under 24 words."
                            )
                            ideal_max_length = 64
                        elif ideal_prompt_style == "delta_visual_evidence":
                            ideal_instruction = (
                                f"User request: '{prompt}'. Inspect the intermediate image and write "
                                "one concise positive correction target for the final image. Make each "
                                "requested attribute visually actionable: when a requested color or "
                                "attribute applies to an object, name the visible object parts or "
                                "regions in the current image that should show it, such as surface, "
                                "body, top, front, handle, wheels, petals, toppings, cheese, crust, "
                                "or clothing. Use only parts that plausibly belong to the requested "
                                "object; if unsure, say the whole object. Preserve all requested "
                                "objects, counts, attribute bindings, and spatial relations. You may "
                                "keep simple visible support context only if harmless, e.g. on a table "
                                "or on a surface. Do not add extra salient objects, new colors, style, "
                                "lighting, camera, texture, mood, or background description. If the "
                                "image conflicts with the user request, follow the user request. Output "
                                "only one sentence, under 24 words."
                            )
                            ideal_max_length = 64
                        elif ideal_prompt_style == "delta_context_v2":
                            ideal_instruction = (
                                f"User request: '{prompt}'. Look at the intermediate image and output "
                                "one sentence that the final image should satisfy. The sentence must "
                                "start from the user's required objects/counts/colors/relations. You "
                                "may append only a simple place phrase if it is clearly visible and "
                                "harmless, chosen from: on a table, on a surface, in a room, outdoors, "
                                "on grass, on a street. Do not add adjectives to the place phrase. Do "
                                "not mention background color, material, lighting, texture, camera, "
                                "style, quality, mood, or any extra object. If uncertain, omit context. "
                                "If the image conflicts with the user request, follow the user request. "
                                "Output only the sentence, under 18 words."
                            )
                            ideal_max_length = 48
                        elif ideal_prompt_style == "delta_context":
                            ideal_instruction = (
                                f"User request: '{prompt}'. Inspect the intermediate image and write "
                                "one concise positive target caption for the corrected final image. "
                                "Always preserve all requested objects, exact counts, colors, attribute "
                                "bindings, and spatial relations. You may keep simple harmless context "
                                "already visible, such as on a table, on a surface, outdoors, or in a "
                                "room. Do not add new salient objects, new colors, new counts, or new "
                                "spatial relations not required by the user. Do not mention lighting, "
                                "camera, style, texture, quality, mood, or aesthetic details. If the "
                                "current image conflicts with the user request, follow the user request. "
                                "Output one sentence only, under 22 words."
                            )
                            ideal_max_length = 56
                        elif ideal_prompt_style == "delta_constraints":
                            ideal_instruction = (
                                f"User request: '{prompt}'. Inspect the intermediate image only to "
                                "identify what visual constraint should be corrected. Output exactly "
                                "one short positive sentence for the final image. The sentence must "
                                "contain only the requested objects, counts, colors, attribute "
                                "bindings, and spatial relations. Do not mention background, lighting, "
                                "style, camera, texture, quality, atmosphere, or anything not in the "
                                "user request. Do not describe the current image or its mistakes. "
                                "Output the sentence only, using at most 18 words."
                            )
                            ideal_max_length = 48
                        elif ideal_prompt_style in {
                            "minimal_corrective",
                            "minimal_corrective_anchored",
                        }:
                            ideal_instruction = (
                                f"The user requested: '{prompt}'. Inspect the noisy intermediate "
                                "image and determine what must be corrected. Then output only one "
                                "short, positive caption for the corrected final image, using at "
                                "most 25 words. It must explicitly preserve every requested object, "
                                "exact count, color, attribute binding, and spatial relation. Never "
                                "mention the current wrong appearance, alternatives, negation, "
                                "unrequested objects, camera/style details, background, lighting, "
                                "or atmosphere. Output the caption only."
                            )
                            ideal_max_length = 64
                        elif ideal_prompt_style == "constraint_first":
                            ideal_instruction = (
                                f"The user requested: '{prompt}'. Describe the corrected ideal "
                                "image in at most 50 words. The first sentence must explicitly state "
                                "every requested object, exact count, color, attribute binding, and "
                                "spatial relation. Follow the user request over the noisy image. "
                                "Do not add unrequested objects or atmosphere. Output only the "
                                "description."
                            )
                            ideal_max_length = 96
                        else:
                            ideal_instruction = (
                                f"Based on the text prompt '{prompt}' and the provided noisy "
                                "intermediate image, describe the envisioned perfect final image "
                                "in one concise paragraph. Ignore noise and artifacts, focusing "
                                "strictly on the visual details and atmosphere. If there is any "
                                "conflict between the visual features and the text prompt, you must "
                                "strictly follow the text prompt. Only one paragraph is required."
                            )
                            ideal_max_length = 192
                        if ideal_prompt_style != "user_only" and ideal_instruction is not None:
                            extra_p = self.chat(
                                tokenizer=tokenizer,
                                new_token_ids=new_token_ids,
                                image_transform=image_transform,
                                images=[ideal_img],
                                prompt=ideal_instruction,
                                max_length=ideal_max_length,
                                do_sample=False,
                            )
                        if "键" in extra_p:
                            extra_p = prompt
                        if ideal_prompt_style == "delta_stable_target_v14":
                            extra_p = extra_p.strip()
                            if extra_p.lower().startswith("target:"):
                                extra_p = extra_p.split(":", 1)[1].strip()
                        if ideal_prompt_style == "delta_visual_evidence_guarded":
                            unsafe_terms = (
                                "not ", " no ", " but ", "mistake", "wrong", "incorrect",
                                "lighting", "spotlight", "shadow", "camera", "style",
                                "background", "foreground", "mood", "atmosphere",
                            )
                            lowered_extra_p = f" {extra_p.lower()} "
                            if any(term in lowered_extra_p for term in unsafe_terms):
                                print(
                                    "delta_visual_evidence_guarded fallback to user prompt: "
                                    f"{extra_p}"
                                )
                                extra_p = prompt
                        if ideal_prompt_style in {
                            "minimal_corrective_anchored",
                            "delta_short_user_anchored_v10",
                        }:
                            # Keep the UMM-generated corrective description, but
                            # place the exact user instruction first so truncated
                            # token CE cannot optimize an invented attribute at the
                            # expense of an explicit user constraint.
                            extra_p = f"{prompt}. {extra_p}"
                        print(f"{loss_mode} ideal prompt: {extra_p}")

                    alignment_log = None
                    precomputed_grad = None
                    update_loss_mode = loss_mode
                    use_ce_for_update = (
                        use_understanding_ce_loss
                        and (not hybrid_reca_guidance or re_update == 0)
                    )
                    if internal_feature_guidance:
                        update_loss_mode = "internal_feature"
                        loss, _ = self.ContextualHiddenAlignmentLoss(
                            target_text=extra_p,
                            tokenizer=tokenizer,
                            max_text_tokens=ce_max_tokens,
                        )
                    elif hybrid_reca_guidance:
                        update_loss_mode = "ce_clip_joint"
                        shared_image = self._decode_reca_latent_to_tensor(
                            x_t_0,
                            image_shapes,
                            vae_model,
                        )
                        ce_loss, _ = self.UnderstandingCELoss(
                            x_t_0=x_t_0,
                            target_text=extra_p,
                            vae_model=vae_model,
                            image_shape=image_shapes,
                            image_transform=image_transform,
                            tokenizer=tokenizer,
                            new_token_ids=new_token_ids,
                            ce_max_tokens=ce_max_tokens,
                            ce_vit_max_side=168,
                            umm_dropbp_layers=umm_dropbp_layers,
                            decoded_image=shared_image,
                        )
                        clip_loss, _ = self.RecALoss(
                            prompt,
                            x_t_0_latent,
                            vae_model,
                            image_shapes,
                            tokenizer,
                            new_token_ids,
                            image_transform,
                            clip_model,
                            clip_processor,
                            use_longclip,
                            use_fgclip,
                            extra_p=extra_p,
                            decoded_image=shared_image,
                        )
                        ce_loss_scale = (
                            umm_dropbp_grad_scale if umm_dropbp_calibrated else 1.0
                        )
                        loss = (
                            ce_loss_scale * ce_loss
                            + joint_clip_weight * clip_loss.float().mean()
                        )
                        alignment_log = (
                            f", ce_loss={ce_loss.detach().item():.6g}, "
                            f"clip_loss={clip_loss.detach().float().mean().item():.6g}, "
                            f"clip_weight={joint_clip_weight:g}"
                        )
                    elif dual_ce_guidance:
                        update_loss_mode = "dual_view_ce"
                        shared_image = self._decode_reca_latent_to_tensor(
                            x_t_0,
                            image_shapes,
                            vae_model,
                        )
                        dual_ce_kwargs = dict(
                            x_t_0=x_t_0,
                            vae_model=vae_model,
                            image_shape=image_shapes,
                            image_transform=image_transform,
                            tokenizer=tokenizer,
                            new_token_ids=new_token_ids,
                            ce_vit_max_side=168,
                            umm_dropbp_layers=umm_dropbp_layers,
                            decoded_image=shared_image,
                        )

                        caption_loss, _, visual_context = self.UnderstandingCELoss(
                            target_text=extra_p,
                            ce_max_tokens=ce_max_tokens,
                            return_visual_context=True,
                            **dual_ce_kwargs,
                        )

                        if dual_ce_user_mode == "caption_user":
                            user_loss, _ = self.UnderstandingCELoss(
                                target_text=prompt,
                                ce_max_tokens=ce_max_tokens,
                                prepared_visual_context=visual_context,
                                **dual_ce_kwargs,
                            )
                            user_label = "user_caption"
                        else:
                            binary_instruction = (
                                "Act as a strict visual verifier. Inspect the image and "
                                "decide whether it satisfies every explicit requirement "
                                "in the target prompt. Check object identities, exact "
                                "counts, colors, attribute bindings, and spatial "
                                "relations. If any required detail is missing, wrong, "
                                "ambiguous, or not visibly supported, answer No. "
                                f"Target prompt: {prompt}\n"
                                "Answer exactly Yes or No."
                            )
                            user_loss, _ = self.UnderstandingCELoss(
                                target_text="Yes",
                                contrast_target_text="No",
                                instruction=binary_instruction,
                                ce_max_tokens=1,
                                prepared_visual_context=visual_context,
                                **dual_ce_kwargs,
                            )
                            user_label = "binary"
                        caption_grad_image = torch.autograd.grad(
                            caption_loss,
                            shared_image,
                            retain_graph=True,
                        )[0]
                        user_grad_image = torch.autograd.grad(
                            user_loss,
                            shared_image,
                        )[0]

                        caption_grad_norm = caption_grad_image.norm(p=2).clamp_min(1e-12)
                        user_grad_norm = user_grad_image.norm(p=2).clamp_min(1e-12)
                        caption_grad_unit = caption_grad_image / caption_grad_norm
                        user_grad_unit = user_grad_image / user_grad_norm
                        grad_cosine = torch.sum(
                            caption_grad_unit * user_grad_unit
                        )

                        # c_ideal gives the reflected correction direction; the user
                        # prompt supplies an instruction anchor. For user-anchored mixing,
                        # remove only the component of ideal gradient that conflicts with
                        # the user gradient.
                        conflict = grad_cosine < 0
                        if dual_ce_anchor == "user":
                            projected_caption_grad = torch.where(
                                conflict,
                                caption_grad_unit - grad_cosine * user_grad_unit,
                                caption_grad_unit,
                            )
                            projected_caption_grad = projected_caption_grad / projected_caption_grad.norm(
                                p=2
                            ).clamp_min(1e-12)
                            guided_grad_image = (
                                (1.0 - dual_ce_binary_weight) * projected_caption_grad
                                + dual_ce_binary_weight * user_grad_unit
                            )
                        else:
                            projected_user_grad = torch.where(
                                conflict,
                                user_grad_unit - grad_cosine * caption_grad_unit,
                                user_grad_unit,
                            )
                            projected_user_grad = projected_user_grad / projected_user_grad.norm(
                                p=2
                            ).clamp_min(1e-12)
                            guided_grad_image = (
                                (1.0 - dual_ce_binary_weight) * caption_grad_unit
                                + dual_ce_binary_weight * projected_user_grad
                            )
                        guided_grad_x0 = torch.autograd.grad(
                            shared_image,
                            x_t_0,
                            grad_outputs=guided_grad_image,
                        )[0]
                        precomputed_grad = torch.autograd.grad(
                            x_t_0,
                            x_t,
                            grad_outputs=guided_grad_x0,
                        )[0]
                        loss = (
                            (1.0 - dual_ce_binary_weight) * caption_loss
                            + dual_ce_binary_weight * user_loss
                        )
                        alignment_log = (
                            f", ideal_loss={caption_loss.detach().item():.6g}, "
                            f"{user_label}_loss={user_loss.detach().item():.6g}, "
                            f"ideal_grad_norm={caption_grad_norm.detach().item():.6g}, "
                            f"{user_label}_grad_norm={user_grad_norm.detach().item():.6g}, "
                            f"ideal_user_grad_cos={grad_cosine.detach().item():.6g}, "
                            f"conflict_projected={bool(conflict.detach().item())}, "
                            f"anchor={dual_ce_anchor}, "
                            f"user_weight={dual_ce_binary_weight:.3f}"
                        )
                    elif use_ce_for_update:
                        update_loss_mode = "understanding_ce"
                        if ce_target_mode == "binary_user":
                            update_loss_mode = "binary_user_ce"
                            binary_instruction = (
                                "Act as a strict visual verifier. Inspect the image and decide "
                                "whether it satisfies every explicit requirement in the target "
                                "prompt. Check object identity, exact count, color, attribute "
                                "binding, and spatial relation. If any required detail is missing, "
                                "wrong, ambiguous, or not visibly supported, answer No. "
                                f"Target prompt: {prompt}\n"
                                "Answer exactly Yes or No."
                            )
                            ce_kwargs = dict(
                                x_t_0=x_t_0,
                                target_text="Yes",
                                contrast_target_text="No",
                                instruction=binary_instruction,
                                vae_model=vae_model,
                                image_shape=image_shapes,
                                image_transform=image_transform,
                                tokenizer=tokenizer,
                                new_token_ids=new_token_ids,
                                ce_max_tokens=1,
                                ce_vit_max_side=ce_vit_max_side,
                            )
                        else:
                            ce_instruction = None
                            if ce_instruction_mode == "target_terms":
                                ce_instruction = (
                                    "Describe the image using concise positive visual terms. "
                                    "Focus on the target objects, counts, colors, attributes, "
                                    "spatial relations, recognizable shape, and helpful scene "
                                    "context. Do not critique the image."
                                )
                            ce_kwargs = dict(
                                x_t_0=x_t_0,
                                target_text=extra_p,
                                instruction=ce_instruction,
                                vae_model=vae_model,
                                image_shape=image_shapes,
                                image_transform=image_transform,
                                tokenizer=tokenizer,
                                new_token_ids=new_token_ids,
                                ce_max_tokens=ce_max_tokens,
                                ce_vit_max_side=ce_vit_max_side,
                            )
                        if enable_dropbp and not umm_dropbp_calibrated:
                            loss, _ = self.UnderstandingCELoss(
                                **ce_kwargs,
                                umm_dropbp_layers=(),
                            )
                            full_grad_x0 = torch.autograd.grad(loss, x_t_0)[0]
                            full_norm = full_grad_x0.norm(p=2).clamp_min(1e-12)

                            num_layers = len(self.language_model.model.layers)
                            group_size = 4
                            groups = [
                                tuple(range(start, min(start + group_size, num_layers)))
                                for start in range(0, num_layers, group_size)
                            ]
                            candidate_results = []
                            best_sensitivity = float("inf")
                            best_group = ()
                            best_grad_x0 = None
                            best_cosine = -1.0
                            best_norm_ratio = 1.0

                            for group in groups:
                                candidate_loss, _ = self.UnderstandingCELoss(
                                    **ce_kwargs,
                                    umm_dropbp_layers=group,
                                )
                                candidate_grad_x0 = torch.autograd.grad(
                                    candidate_loss,
                                    x_t_0,
                                )[0]
                                candidate_norm = candidate_grad_x0.norm(p=2).clamp_min(1e-12)
                                cosine = torch.sum(
                                    full_grad_x0 * candidate_grad_x0
                                ) / (full_norm * candidate_norm)
                                norm_ratio = candidate_norm / full_norm
                                sensitivity = (
                                    1.0
                                    - cosine
                                    + 0.1 * torch.abs(torch.log(norm_ratio))
                                )
                                sensitivity_value = sensitivity.item()
                                cosine_value = cosine.item()
                                norm_ratio_value = norm_ratio.item()
                                candidate_results.append(
                                    (sensitivity_value, group, cosine_value, norm_ratio_value)
                                )
                                print(
                                    f"DropBP candidate layers={group}: "
                                    f"sensitivity={sensitivity_value:.6g}, "
                                    f"cos={cosine_value:.6g}, "
                                    f"norm_ratio={norm_ratio_value:.6g}"
                                )
                                if sensitivity_value < best_sensitivity:
                                    best_sensitivity = sensitivity_value
                                    best_group = group
                                    best_grad_x0 = candidate_grad_x0.detach()
                                    best_cosine = cosine_value
                                    best_norm_ratio = norm_ratio_value

                            target_groups = max(
                                1,
                                round(0.25 * num_layers / group_size),
                            )
                            ranked_groups = sorted(candidate_results, key=lambda item: item[0])
                            union_layers = tuple(sorted({
                                layer
                                for _, group, _, _ in ranked_groups[:target_groups]
                                for layer in group
                            }))

                            if union_layers == best_group:
                                union_grad_x0 = best_grad_x0
                                union_cosine = best_cosine
                                union_norm_ratio = best_norm_ratio
                            else:
                                union_loss, _ = self.UnderstandingCELoss(
                                    **ce_kwargs,
                                    umm_dropbp_layers=union_layers,
                                )
                                union_grad_x0 = torch.autograd.grad(
                                    union_loss,
                                    x_t_0,
                                )[0].detach()
                                union_norm = union_grad_x0.norm(p=2).clamp_min(1e-12)
                                union_cosine = (
                                    torch.sum(full_grad_x0 * union_grad_x0)
                                    / (full_norm * union_norm)
                                ).item()
                                union_norm_ratio = (union_norm / full_norm).item()

                            min_drop_cosine = 0.8
                            if union_cosine >= min_drop_cosine:
                                umm_dropbp_layers = union_layers
                                selected_grad_x0 = union_grad_x0
                                selected_cosine = union_cosine
                                selected_norm_ratio = union_norm_ratio
                            elif best_cosine >= min_drop_cosine:
                                umm_dropbp_layers = best_group
                                selected_grad_x0 = best_grad_x0
                                selected_cosine = best_cosine
                                selected_norm_ratio = best_norm_ratio
                            else:
                                umm_dropbp_layers = ()
                                selected_grad_x0 = full_grad_x0.detach()
                                selected_cosine = 1.0
                                selected_norm_ratio = 1.0

                            umm_dropbp_grad_scale = 1.0 / max(
                                selected_norm_ratio,
                                1e-12,
                            )
                            umm_dropbp_calibrated = True
                            alignment_log = (
                                f", dropbp_layers={umm_dropbp_layers}, "
                                f"dropbp_cos={selected_cosine:.6g}, "
                                f"dropbp_norm_ratio={selected_norm_ratio:.6g}"
                            )
                            print(
                                f"DropBP selected layers={umm_dropbp_layers}, "
                                f"cos={selected_cosine:.6g}, "
                                f"grad_scale={umm_dropbp_grad_scale:.6g}"
                            )
                            os.makedirs(os.path.dirname(dropbp_profile_path), exist_ok=True)
                            with open(
                                dropbp_profile_path,
                                "w",
                                encoding="utf-8",
                            ) as profile_file:
                                json.dump(
                                    {
                                        "layers": list(umm_dropbp_layers),
                                        "grad_scale": umm_dropbp_grad_scale,
                                        "calibration_cos": selected_cosine,
                                    },
                                    profile_file,
                                    indent=2,
                                )
                            guided_grad_x0 = (
                                selected_grad_x0 * umm_dropbp_grad_scale
                            )
                        else:
                            loss, _ = self.UnderstandingCELoss(
                                **ce_kwargs,
                                umm_dropbp_layers=umm_dropbp_layers,
                            )
                            guided_grad_x0 = torch.autograd.grad(loss, x_t_0)[0]
                            guided_grad_x0 = (
                                guided_grad_x0 * umm_dropbp_grad_scale
                            )

                        precomputed_grad = torch.autograd.grad(
                            x_t_0,
                            x_t,
                            grad_outputs=guided_grad_x0,
                        )[0]
                    else:
                        update_loss_mode = "clip"
                        loss, _ = self.RecALoss(
                            prompt,
                            x_t_0_latent,
                            vae_model,
                            image_shapes,
                            tokenizer,
                            new_token_ids,
                            image_transform,
                            clip_model,
                            clip_processor,
                            use_longclip,
                            use_fgclip,
                            extra_p=extra_p,
                        )

                    if precomputed_grad is None:
                        grad = torch.autograd.grad(loss, x_t)[0]
                    else:
                        grad = precomputed_grad
                    if internal_feature_guidance:
                        self._reca_generation_hidden = None

                    loss_value = loss.detach().item()
                    grad_norm = grad.norm(p=2)
                    loss_list.append(loss_value)
                    print(
                        f"latent {update_loss_mode} grad step {i}, update {re_update}: "
                        f"loss={loss_value:.6g}, "
                        f"raw_norm={grad_norm.item():.6g}"
                        + (alignment_log or "")
                    )

                    if not torch.isfinite(grad_norm):
                        raise RuntimeError(
                            f"Non-finite ReCA gradient at denoising step {i}, "
                            f"update {re_update}"
                        )
                    default_grad_clip_threshold = (
                        0.002 if internal_feature_guidance else 0.01
                    )
                    grad_clip_threshold = float(os.environ.get(
                        "BAGEL_NATIVE_CE_GRAD_CLIP_THRESHOLD",
                        str(default_grad_clip_threshold),
                    ))
                    if grad_norm > grad_clip_threshold:
                        flag = True
                        grad = grad * (grad_clip_threshold / grad_norm)

                    reuse_count = re_update_num_ if reuse_reca_grad else 1
                    if reuse_reca_grad:
                        early_reuse_count = int(os.environ.get(
                            "BAGEL_REUSE_RECA_GRAD_EARLY_COUNT", "0"
                        ))
                        last_reuse_count = int(os.environ.get(
                            "BAGEL_REUSE_RECA_GRAD_LAST_COUNT", "0"
                        ))
                        if i == last_update_step and last_reuse_count > 0:
                            reuse_count = last_reuse_count
                        elif i != last_update_step and early_reuse_count > 0:
                            reuse_count = early_reuse_count
                    accept_if_ce_decreases = os.environ.get(
                        "BAGEL_NATIVE_CE_ACCEPT_IF_LOSS_DECREASE", "0"
                    ) == "1"
                    accept_mode = os.environ.get(
                        "BAGEL_NATIVE_CE_ACCEPT_MODE", "target"
                    )
                    accept_min_delta = float(os.environ.get(
                        "BAGEL_NATIVE_CE_ACCEPT_MIN_DELTA", "0.0"
                    ))
                    accept_user_min_delta = float(os.environ.get(
                        "BAGEL_NATIVE_CE_ACCEPT_USER_MIN_DELTA", "0.0"
                    ))
                    for reuse_index in range(reuse_count):
                        candidate_x_t = (
                            x_t - update_scale * grad
                        ).detach().requires_grad_()
                        if (
                            accept_if_ce_decreases
                            and use_ce_for_update
                            and update_loss_mode in {
                                "understanding_ce",
                                "binary_user_ce",
                            }
                        ):
                            with torch.no_grad():
                                candidate_x0 = (
                                    candidate_x_t.detach()
                                    - v_t.detach().to(candidate_x_t.device) * t
                                )
                                candidate_ce_kwargs = dict(ce_kwargs)
                                candidate_ce_kwargs["x_t_0"] = candidate_x0
                                candidate_loss, _ = self.UnderstandingCELoss(
                                    **candidate_ce_kwargs,
                                    umm_dropbp_layers=(),
                                )
                                candidate_loss_value = candidate_loss.detach().item()
                            target_ok = (
                                candidate_loss_value <= loss_value - accept_min_delta
                            )
                            user_before_value = None
                            user_after_value = None
                            user_ok = True
                            if accept_mode in {"user_binary", "target_and_user"}:
                                binary_gate_instruction = (
                                    "Act as a strict visual verifier. Inspect the image and decide "
                                    "whether it satisfies every explicit requirement in the target "
                                    "prompt. Check object identity, exact count, color, attribute "
                                    "binding, and spatial relation. If any required detail is missing, "
                                    "wrong, ambiguous, or not visibly supported, answer No. "
                                    f"Target prompt: {prompt}\n"
                                    "Answer exactly Yes or No."
                                )
                                user_gate_kwargs = dict(
                                    x_t_0=x_t_0.detach(),
                                    target_text="Yes",
                                    contrast_target_text="No",
                                    instruction=binary_gate_instruction,
                                    vae_model=vae_model,
                                    image_shape=image_shapes,
                                    image_transform=image_transform,
                                    tokenizer=tokenizer,
                                    new_token_ids=new_token_ids,
                                    ce_max_tokens=1,
                                    ce_vit_max_side=ce_vit_max_side,
                                )
                                before_user_loss, _ = self.UnderstandingCELoss(
                                    **user_gate_kwargs,
                                    umm_dropbp_layers=(),
                                )
                                user_gate_kwargs["x_t_0"] = candidate_x0
                                after_user_loss, _ = self.UnderstandingCELoss(
                                    **user_gate_kwargs,
                                    umm_dropbp_layers=(),
                                )
                                user_before_value = before_user_loss.detach().item()
                                user_after_value = after_user_loss.detach().item()
                                user_ok = (
                                    user_after_value
                                    <= user_before_value - accept_user_min_delta
                                )

                            if accept_mode == "user_binary":
                                accept_update = user_ok
                            elif accept_mode == "target_and_user":
                                accept_update = target_ok and user_ok
                            else:
                                accept_update = target_ok

                            user_log = ""
                            if user_before_value is not None:
                                user_log = (
                                    f", user_before={user_before_value:.6g}, "
                                    f"user_after={user_after_value:.6g}"
                                )
                            if accept_update:
                                x_t = candidate_x_t
                                print(
                                    f"native_ce gated update accepted step {i}, "
                                    f"update {re_update}, reuse {reuse_index}, "
                                    f"mode={accept_mode}: before={loss_value:.6g}, "
                                    f"after={candidate_loss_value:.6g}"
                                    + user_log
                                )
                                loss_value = candidate_loss_value
                            else:
                                print(
                                    f"native_ce gated update rejected step {i}, "
                                    f"update {re_update}, reuse {reuse_index}, "
                                    f"mode={accept_mode}: before={loss_value:.6g}, "
                                    f"after={candidate_loss_value:.6g}"
                                    + user_log
                                )
                                break
                        else:
                            x_t = candidate_x_t
                        if reuse_index > 0:
                            print(
                                f"latent {loss_mode} grad step {i}, "
                                f"update {reuse_index}: reused update 0 gradient"
                            )

                # elif update_flag:
                #     with torch.no_grad():
                #         x_t_0 = x_t - v_t.to(x_t.device) * t
                #         x_t_0_latent = x_t_0.split((packed_seqlens - 2).tolist())
                #         loss, vlm_output_text = self.calc_clip_with_prompt_nograd(prompt, x_t_0_latent, vae_model, image_shapes, tokenizer, new_token_ids, image_transform, clip_model, clip_processor, use_longclip, use_fgclip)
                #         loss_list.append(loss.item())

                if save_grad:
                    os.makedirs(f"/home/ma-user/work/wx1468559/Bagel-Reca/gradints/{prompt.replace(' ', '_')}", exist_ok=True)
                    torch.save(grad, f"/home/ma-user/work/wx1468559/Bagel-Reca/gradints/{prompt.replace(' ', '_')}/grad_{i}.pt")
                    torch.save(x_t, f"/home/ma-user/work/wx1468559/Bagel-Reca/gradints/{prompt.replace(' ', '_')}/x_{i}.pt")
                
                # breakpoint()
                if use_save_pic and update_flag:
                    def decode_image(latent, image_shape):
                        H, W = image_shape
                        h, w = H // self.latent_downsample, W // self.latent_downsample
                        latent = latent.reshape(1, h, w, self.latent_patch_size, self.latent_patch_size, self.latent_channel)
                        latent = torch.einsum("nhwpqc->nchpwq", latent)
                        latent = latent.reshape(1, self.latent_channel, h * self.latent_patch_size, w * self.latent_patch_size)
                        
                        vae_device = next(vae_model.parameters()).device
                        image = vae_model.decode(latent.to(torch.bfloat16).to(vae_device))
                        image = (image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
                        image = Image.fromarray((image).to(torch.uint8).cpu().numpy())
                        return image
                    x_0 = x_t - v_t.to(x_t.device) * t
                    x_0_latent = x_0.split((packed_seqlens - 2).tolist())
                    img = decode_image(x_0_latent[0], image_shapes)
                    os.makedirs("./debug/pic/", exist_ok=True)
                    img.save(f"./debug/pic/debug_middle_{i}_{re_update}.png")


            if (
                in_step_user_ce_gss
                and update_flag
                and len(in_step_gss_candidates_next) > 1
            ):
                if i + 1 >= len(timesteps):
                    raise RuntimeError(
                        "In-step User CE GSS requires a subsequent timestep"
                    )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                select_t = timesteps[i + 1]
                binary_gss_instruction = (
                    "Act as a strict visual verifier. Inspect the image and decide "
                    "whether it satisfies every explicit requirement in the target "
                    "prompt. Check object identity, exact count, color, attribute "
                    "binding, and spatial relation. If any required detail is missing, "
                    "wrong, ambiguous, or not visibly supported, answer No. "
                    f"Target prompt: {ce_selection_prompt}\n"
                    "Answer exactly Yes or No."
                )
                in_step_gss_scores = []
                with torch.no_grad():
                    for candidate_index, candidate_next in enumerate(
                        in_step_gss_candidates_next
                    ):
                        candidate_v = infer_one_step(candidate_next, select_t)
                        candidate_x0 = (
                            candidate_next
                            - candidate_v.to(candidate_next.device) * select_t
                        )
                        candidate_loss, _ = self.UnderstandingCELoss(
                            x_t_0=candidate_x0,
                            target_text="Yes",
                            contrast_target_text="No",
                            instruction=binary_gss_instruction,
                            vae_model=vae_model,
                            image_shape=image_shapes,
                            image_transform=image_transform,
                            tokenizer=tokenizer,
                            new_token_ids=new_token_ids,
                            ce_max_tokens=1,
                            ce_vit_max_side=ce_selection_vit_max_side,
                            umm_dropbp_layers=(),
                        )
                        score = float(candidate_loss.detach().item())
                        in_step_gss_scores.append(score)
                        print(
                            f"in-step User CE GSS step {i}, "
                            f"candidate {candidate_index}: loss={score:.6g}"
                        )
                selected_index = min(
                    range(len(in_step_gss_scores)),
                    key=lambda idx: (in_step_gss_scores[idx], -idx),
                )
                x_t_1 = in_step_gss_candidates_next[selected_index]
                print(
                    f"in-step User CE GSS step {i}: "
                    f"scores={in_step_gss_scores}, selected={selected_index}"
                )

            if final_ideal_ce_select and i == last_update_step:
                if final_select_unmodified_next is None or extra_p is None:
                    raise RuntimeError(
                        "Final ideal-CE selection requires both candidates and c_ideal"
                    )

                final_select_candidates = [
                    final_select_unmodified_next,
                    x_t_1.detach(),
                ]
                final_select_scores = []
                if i + 1 >= len(timesteps):
                    raise RuntimeError(
                        "Final ideal-CE selection requires a subsequent timestep"
                    )
                select_t = timesteps[i + 1]
                with torch.no_grad():
                    for candidate_index, candidate_next in enumerate(
                        final_select_candidates
                    ):
                        candidate_v = infer_one_step(candidate_next, select_t)
                        candidate_x0 = (
                            candidate_next
                            - candidate_v.to(candidate_next.device) * select_t
                        )
                        candidate_loss, _ = self.UnderstandingCELoss(
                            x_t_0=candidate_x0,
                            target_text=(
                                "Yes"
                                if final_ce_select_mode == "user_binary"
                                else extra_p
                            ),
                            vae_model=vae_model,
                            image_shape=image_shapes,
                            image_transform=image_transform,
                            tokenizer=tokenizer,
                            new_token_ids=new_token_ids,
                            ce_max_tokens=(
                                1
                                if final_ce_select_mode == "user_binary"
                                else ce_max_tokens
                            ),
                            ce_vit_max_side=ce_vit_max_side,
                            instruction=(
                                "Act as a strict visual verifier. Inspect the image "
                                "and decide whether it satisfies every explicit "
                                "requirement in the target prompt. Check object "
                                "identities, exact counts, colors, attribute "
                                "bindings, and spatial relations. If any required "
                                "detail is missing, wrong, ambiguous, or not visibly "
                                "supported, answer No. "
                                f"Target prompt: {prompt}\n"
                                "Answer exactly Yes or No."
                                if final_ce_select_mode == "user_binary"
                                else None
                            ),
                            contrast_target_text=(
                                "No"
                                if final_ce_select_mode == "user_binary"
                                else None
                            ),
                            umm_dropbp_layers=(),
                        )
                        score = float(candidate_loss.detach().item())
                        final_select_scores.append(score)
                        print(
                            f"final ideal CE selection step {i}, "
                            f"candidate {candidate_index}, "
                            f"mode={final_ce_select_mode}: loss={score:.6g}"
                        )

                # Lower CE is better. On an exact tie retain the fully
                # rectified (later) candidate, as requested.
                selected_index = (
                    0
                    if final_select_scores[0] < final_select_scores[1]
                    else 1
                )
                x_t_1 = final_select_candidates[selected_index]
                print(
                    f"final ideal CE selection step {i}: "
                    f"scores={final_select_scores}, selected={selected_index}"
                )

            print(f"denoising step {i}, loss: {loss_list}")
            # x_t_0 = x_t - v_t.to(x_t.device) * t
            # x_t_0_latent = x_t_0.split((packed_seqlens - 2).tolist())
            # l, v = self.calc_clip_with_prompt_nograd(prompt, x_t_0_latent, vae_model, image_shapes, tokenizer, new_token_ids, image_transform, clip_model, clip_processor, use_longclip, use_fgclip)
            # score_list.append(1 - l.item())
           
            if use_lookback and update_flag:
                print(x_t_loss)
                best_index = max(
                    range(len(x_t_loss)),
                    key=lambda i: (-x_t_loss[i], i)
                )
                x_t = x_t_list[best_index].detach().requires_grad_()
                print(best_index)
            else:
                x_t = x_t_1.detach().requires_grad_()
            # breakpoint()
            # breakpoint()
        
        if enable_taylorseer:
            del model_pred_cache_dic, model_pred_current
            del model_pred_text_cache_dic, model_pred_text_current
            del model_pred_img_cache_dic, model_pred_img_current

        unpacked_latent = x_t.split((packed_seqlens - 2).tolist())
        # breakpoint()

        # x = torch.load('score.pt')
        # x.append(score_list)
        # torch.save(x,'score.pt')
        return unpacked_latent

    def _forward_flow_reca(
        self,
        x_t: torch.Tensor,
        timestep: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        key_values_lens: torch.IntTensor,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        # cfg_text
        cfg_text_scale: float = 1.0,
        cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_text_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_text_key_values_lens: Optional[torch.Tensor] = None,
        cfg_text_past_key_values: Optional[NaiveCache] = None,
        cfg_text_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        # cfg_img
        cfg_img_scale: float = 1.0,
        cfg_img_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_img_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_img_key_values_lens: Optional[torch.Tensor] = None,
        cfg_img_past_key_values: Optional[NaiveCache] = None,
        cfg_img_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_type: str = "parallel",
        # cache
        model_pred_cache_dic: Optional[Dict[str, Any]] = None,
        model_pred_current: Optional[int] = None,
        model_pred_text_cache_dic: Optional[Dict[str, Any]] = None,
        model_pred_text_current: Optional[int] = None,
        model_pred_img_cache_dic: Optional[Dict[str, Any]] = None,
        model_pred_img_current: Optional[int] = None,
        final_flag = False
    ):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        assert timestep.unique().shape[0] == 1
        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
        # timestep=timestep.to(torch.bfloat16)
        packed_timestep_embeds = self.time_embedder(timestep)
        x_t = self.vae2llm(x_t) + packed_timestep_embeds + packed_pos_embed
        if x_t.dtype != packed_sequence.dtype:
            x_t = x_t.to(packed_sequence.dtype)
        packed_sequence[packed_vae_token_indexes] = x_t

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {
                "mode": "gen",
                "packed_vae_token_indexes": packed_vae_token_indexes,
                "packed_text_indexes": packed_text_indexes
            }
        
        if self.language_model.model.enable_taylorseer:
            self.language_model.model.cache_dic = model_pred_cache_dic
            self.language_model.model.current = model_pred_current

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=False,
            is_causal=False,
            **extra_inputs,
        )
        v_t = self.llm2vae(output.packed_query_sequence)
        v_t = v_t[packed_vae_token_indexes]

        if not final_flag:
            self._reca_generation_hidden = output.packed_query_sequence[
                packed_vae_token_indexes
            ]
            return v_t

        if cfg_text_scale > 1.0:
            if self.language_model.model.enable_taylorseer:
                self.language_model.model.cache_dic = model_pred_text_cache_dic
                self.language_model.model.current = model_pred_text_current
            cfg_text_output = self.language_model.forward_inference(
                packed_query_sequence=packed_sequence,
                query_lens=packed_seqlens,
                packed_query_position_ids=cfg_text_packed_position_ids,
                packed_query_indexes=cfg_text_packed_query_indexes,
                past_key_values=cfg_text_past_key_values,
                key_values_lens=cfg_text_key_values_lens,
                packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs,
            )
            cfg_text_v_t = self.llm2vae(cfg_text_output.packed_query_sequence)
            cfg_text_v_t = cfg_text_v_t[packed_vae_token_indexes]

        if cfg_img_scale > 1.0:
            if self.language_model.model.enable_taylorseer:
                self.language_model.model.cache_dic = model_pred_img_cache_dic
                self.language_model.model.current = model_pred_img_current
            cfg_img_output = self.language_model.forward_inference(
                packed_query_sequence=packed_sequence,
                query_lens=packed_seqlens,
                packed_query_position_ids=cfg_img_packed_position_ids,
                packed_query_indexes=cfg_img_packed_query_indexes,
                past_key_values=cfg_img_past_key_values,
                key_values_lens=cfg_img_key_values_lens,
                packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs,
            )
            cfg_img_v_t = self.llm2vae(cfg_img_output.packed_query_sequence)
            cfg_img_v_t = cfg_img_v_t[packed_vae_token_indexes]

        if cfg_text_scale > 1.0:
            if cfg_renorm_type == "text_channel":
                v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
                norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                norm_v_t_text_ = torch.norm(v_t_text_, dim=-1, keepdim=True)
                scale = (norm_v_t / (norm_v_t_text_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                v_t_text = v_t_text_ * scale
                if cfg_img_scale > 1.0:
                    v_t = cfg_img_v_t + cfg_img_scale * (v_t_text - cfg_img_v_t)
                else:
                    v_t = v_t_text
            else:
                v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
                
                if cfg_img_scale > 1.0:
                    v_t_ = cfg_img_v_t + cfg_img_scale * (v_t_text_ - cfg_img_v_t)
                else:
                    v_t_ = v_t_text_

                # NOTE norm is computed over all dimensions, thus currently only supports batch_size = 1 with navit
                if cfg_renorm_type == "global":
                    norm_v_t = torch.norm(v_t)
                    norm_v_t_ = torch.norm(v_t_)
                elif cfg_renorm_type == "channel":
                    norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                    norm_v_t_ = torch.norm(v_t_, dim=-1, keepdim=True)
                else:
                    raise NotImplementedError(f"{cfg_renorm_type} is not suppoprted")
                scale = (norm_v_t / (norm_v_t_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                v_t = v_t_ * scale
        else:
            # No CFG
            pass

        return v_t

    @torch.no_grad
    def _forward_flow(
        self,
        x_t: torch.Tensor,
        timestep: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        key_values_lens: torch.IntTensor,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        # cfg_text
        cfg_text_scale: float = 1.0,
        cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_text_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_text_key_values_lens: Optional[torch.Tensor] = None,
        cfg_text_past_key_values: Optional[NaiveCache] = None,
        cfg_text_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        # cfg_img
        cfg_img_scale: float = 1.0,
        cfg_img_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_img_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_img_key_values_lens: Optional[torch.Tensor] = None,
        cfg_img_past_key_values: Optional[NaiveCache] = None,
        cfg_img_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_type: str = "parallel",
        # cache
        model_pred_cache_dic: Optional[Dict[str, Any]] = None,
        model_pred_current: Optional[int] = None,
        model_pred_text_cache_dic: Optional[Dict[str, Any]] = None,
        model_pred_text_current: Optional[int] = None,
        model_pred_img_cache_dic: Optional[Dict[str, Any]] = None,
        model_pred_img_current: Optional[int] = None,
    ):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        assert timestep.unique().shape[0] == 1
        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
        packed_timestep_embeds = self.time_embedder(timestep)
        x_t = self.vae2llm(x_t) + packed_timestep_embeds + packed_pos_embed
        if x_t.dtype != packed_sequence.dtype:
            x_t = x_t.to(packed_sequence.dtype)
        packed_sequence[packed_vae_token_indexes] = x_t

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {
                "mode": "gen",
                "packed_vae_token_indexes": packed_vae_token_indexes,
                "packed_text_indexes": packed_text_indexes
            }
        
        if self.language_model.model.enable_taylorseer:
            self.language_model.model.cache_dic = model_pred_cache_dic
            self.language_model.model.current = model_pred_current

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=False,
            is_causal=False,
            **extra_inputs,
        )
        v_t = self.llm2vae(output.packed_query_sequence)
        v_t = v_t[packed_vae_token_indexes]

        if cfg_text_scale > 1.0:
            if self.language_model.model.enable_taylorseer:
                self.language_model.model.cache_dic = model_pred_text_cache_dic
                self.language_model.model.current = model_pred_text_current
            cfg_text_output = self.language_model.forward_inference(
                packed_query_sequence=packed_sequence,
                query_lens=packed_seqlens,
                packed_query_position_ids=cfg_text_packed_position_ids,
                packed_query_indexes=cfg_text_packed_query_indexes,
                past_key_values=cfg_text_past_key_values,
                key_values_lens=cfg_text_key_values_lens,
                packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs,
            )
            cfg_text_v_t = self.llm2vae(cfg_text_output.packed_query_sequence)
            cfg_text_v_t = cfg_text_v_t[packed_vae_token_indexes]

        if cfg_img_scale > 1.0:
            if self.language_model.model.enable_taylorseer:
                self.language_model.model.cache_dic = model_pred_img_cache_dic
                self.language_model.model.current = model_pred_img_current
            cfg_img_output = self.language_model.forward_inference(
                packed_query_sequence=packed_sequence,
                query_lens=packed_seqlens,
                packed_query_position_ids=cfg_img_packed_position_ids,
                packed_query_indexes=cfg_img_packed_query_indexes,
                past_key_values=cfg_img_past_key_values,
                key_values_lens=cfg_img_key_values_lens,
                packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs,
            )
            cfg_img_v_t = self.llm2vae(cfg_img_output.packed_query_sequence)
            cfg_img_v_t = cfg_img_v_t[packed_vae_token_indexes]

        if cfg_text_scale > 1.0:
            if cfg_renorm_type == "text_channel":
                v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
                norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                norm_v_t_text_ = torch.norm(v_t_text_, dim=-1, keepdim=True)
                scale = (norm_v_t / (norm_v_t_text_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                v_t_text = v_t_text_ * scale
                if cfg_img_scale > 1.0:
                    v_t = cfg_img_v_t + cfg_img_scale * (v_t_text - cfg_img_v_t)
                else:
                    v_t = v_t_text
            else:
                v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
                
                if cfg_img_scale > 1.0:
                    v_t_ = cfg_img_v_t + cfg_img_scale * (v_t_text_ - cfg_img_v_t)
                else:
                    v_t_ = v_t_text_

                # NOTE norm is computed over all dimensions, thus currently only supports batch_size = 1 with navit
                if cfg_renorm_type == "global":
                    norm_v_t = torch.norm(v_t)
                    norm_v_t_ = torch.norm(v_t_)
                elif cfg_renorm_type == "channel":
                    norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                    norm_v_t_ = torch.norm(v_t_, dim=-1, keepdim=True)
                else:
                    raise NotImplementedError(f"{cfg_renorm_type} is not suppoprted")
                scale = (norm_v_t / (norm_v_t_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                v_t = v_t_ * scale
        else:
            # No CFG
            pass

        return v_t

    def prepare_start_tokens(self, curr_kvlens, curr_rope, new_token_ids):
        packed_start_tokens, packed_key_value_indexes = list(), list()
        packed_query_position_ids = list()

        curr = 0
        for curr_kvlen, curr_position_id in zip(curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            packed_start_tokens.append(new_token_ids['bos_token_id'])
            packed_query_position_ids.append(curr_position_id)
            curr += curr_kvlen

        generation_input = {
            "packed_start_tokens": torch.tensor(packed_start_tokens, dtype=torch.long),
            "packed_query_position_ids": torch.tensor(packed_query_position_ids, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        return generation_input

    @torch.no_grad
    def generate_text(
        self,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_start_tokens: torch.LongTensor,
        packed_query_position_ids: torch.LongTensor,
        max_length: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        end_token_id: int = None,
    ):
        step = 0
        generated_sequence = []
        curr_tokens = packed_start_tokens
        while step < max_length:
            generated_sequence.append(curr_tokens)
            packed_text_embedding = self.language_model.model.embed_tokens(curr_tokens)
            query_lens = torch.ones_like(curr_tokens)
            packed_query_indexes = torch.cumsum(key_values_lens, dim=0) + torch.arange(
                0, len(key_values_lens), 
                device=key_values_lens.device, 
                dtype=key_values_lens.dtype
            )

            uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                uppacked[i] += i
            packed_key_value_indexes = torch.cat(uppacked, dim=0)

            extra_inputs = {}
            if self.use_moe:
                extra_inputs = {"mode": "und"}

            output = self.language_model.forward_inference(
                packed_query_sequence=packed_text_embedding,
                query_lens=query_lens,
                packed_query_position_ids=packed_query_position_ids,
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=True,
                is_causal=True,
                **extra_inputs,
            )
            past_key_values = output.past_key_values
            packed_query_sequence = output.packed_query_sequence
            pred_logits = self.language_model.lm_head(packed_query_sequence)

            if do_sample:
                probs = nn.functional.softmax(pred_logits / temperature, dim=-1)
                curr_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                curr_tokens = torch.argmax(pred_logits, dim=-1)

            uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                uppacked[i] = torch.cat(
                    [uppacked[i], torch.tensor([uppacked[i][-1] + 1], device=uppacked[i].device)], dim=0
                )
            packed_key_value_indexes = torch.cat(uppacked, dim=0)
            key_values_lens = key_values_lens + 1
            packed_query_position_ids = packed_query_position_ids + 1
            step += 1

            if end_token_id is not None and curr_tokens[0] == end_token_id: # only support batch=1
                break

        output_device = generated_sequence[0].device
        return torch.stack([i.to(output_device) for i in generated_sequence], dim=0)

    # for evaluation
    @torch.no_grad()
    def chat(
        self,
        tokenizer,
        new_token_ids,
        image_transform,
        images,
        prompt,
        max_length: int,
        do_sample: bool = False,
        temperature: float = 1.0,
    ):
        device = next(self.parameters()).device

        if isinstance(new_token_ids, dict):
            for k, v in new_token_ids.items():
                if torch.is_tensor(v):
                    new_token_ids[k] = v.to(device)
        elif torch.is_tensor(new_token_ids):
            new_token_ids = new_token_ids.to(device)

        # prefill
        past_key_values = NaiveCache(self.config.llm_config.num_hidden_layers)
        newlens = [0]
        new_rope = [0]

        # add images
        for image in images:
            generation_input, newlens, new_rope = self.prepare_vit_images(
                curr_kvlens=newlens,
                curr_rope=new_rope, 
                images=[image], 
                transforms=image_transform,
                new_token_ids=new_token_ids,
            )
            for k, v in generation_input.items():
                if torch.is_tensor(v):
                    generation_input[k] = v.to(device)
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                past_key_values = self.forward_cache_update_vit(past_key_values, **generation_input)

        # add text
        generation_input, newlens, new_rope = self.prepare_prompts(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            prompts=[prompt],
            tokenizer=tokenizer, 
            new_token_ids=new_token_ids,
        )
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = self.forward_cache_update_text(past_key_values, **generation_input)

        # decode
        generation_input = self.prepare_start_tokens(newlens, new_rope, new_token_ids)
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            unpacked_latent = self.generate_text(
                past_key_values=past_key_values,
                max_length=max_length,
                do_sample=do_sample,
                temperature=temperature,
                end_token_id=new_token_ids['eos_token_id'],
                **generation_input,
            )
        output = tokenizer.decode(unpacked_latent[:,0])
        output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]
        print(output)

        return output

    @torch.no_grad()
    def _decode_reca_latent_to_pil(self, latent, image_shape, vae_model):
        image = self._decode_reca_latent_to_tensor(latent, image_shape, vae_model)
        image = image[0].permute(1, 2, 0) * 255
        return Image.fromarray(image.to(torch.uint8).cpu().numpy())

    def _decode_reca_latent_to_tensor(self, latent, image_shape, vae_model):
        H, W = image_shape
        h, w = H // self.latent_downsample, W // self.latent_downsample
        latent = latent.reshape(1, h, w, self.latent_patch_size, self.latent_patch_size, self.latent_channel)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, self.latent_channel, h * self.latent_patch_size, w * self.latent_patch_size)
        vae_device = next(vae_model.parameters()).device
        image = vae_model.decode(latent.to(torch.bfloat16).to(vae_device))
        return (image * 0.5 + 0.5).clamp(0, 1)

    def _ce_cache_update_text(
        self,
        past_key_values,
        text,
        curr_kvlens,
        curr_rope,
        tokenizer,
        new_token_ids,
        device,
    ):
        generation_input, newlens, new_rope = self.prepare_prompts(
            curr_kvlens=curr_kvlens,
            curr_rope=curr_rope,
            prompts=[text],
            tokenizer=tokenizer,
            new_token_ids=new_token_ids,
        )
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(device)

        packed_text_embedding = self.language_model.model.embed_tokens(generation_input["packed_text_ids"])
        extra_inputs = {"mode": "und"} if self.use_moe else {}
        output = self.language_model.forward_inference(
            packed_query_sequence=packed_text_embedding,
            query_lens=generation_input["text_token_lens"],
            packed_query_position_ids=generation_input["packed_text_position_ids"],
            packed_query_indexes=generation_input["packed_text_indexes"],
            past_key_values=past_key_values,
            packed_key_value_indexes=generation_input["packed_key_value_indexes"],
            key_values_lens=generation_input["key_values_lens"],
            update_past_key_values=True,
            is_causal=True,
            **extra_inputs,
        )
        return output.past_key_values, newlens, new_rope

    def ContextualHiddenAlignmentLoss(
        self,
        target_text,
        tokenizer,
        max_text_tokens=192,
    ):
        """Align reused generation states with a cached contextual text state."""
        generation_hidden = getattr(self, "_reca_generation_hidden", None)
        if generation_hidden is None:
            raise RuntimeError("ReCA generation hidden was not captured")

        cache_key = (target_text, int(max_text_tokens))
        if getattr(self, "_reca_context_text_key", None) != cache_key:
            device = generation_hidden.device
            text_ids = tokenizer.encode(target_text)[:max_text_tokens]
            if not text_ids:
                raise RuntimeError("Contextual feature target produced no text tokens")
            text_ids = torch.tensor(text_ids, dtype=torch.long, device=device)
            text_lens = torch.tensor([len(text_ids)], dtype=torch.int, device=device)
            text_indexes = torch.arange(len(text_ids), dtype=torch.long, device=device)
            position_ids = torch.arange(len(text_ids), dtype=torch.long, device=device)
            empty_kv_lens = torch.zeros(1, dtype=torch.int, device=device)
            empty_kv_indexes = torch.empty(0, dtype=torch.long, device=device)
            text_cache = NaiveCache(self.config.llm_config.num_hidden_layers)
            extra_inputs = {"mode": "und"} if self.use_moe else {}
            with torch.no_grad():
                text_embeddings = self.language_model.model.embed_tokens(text_ids)
                text_output = self.language_model.forward_inference(
                    packed_query_sequence=text_embeddings,
                    query_lens=text_lens,
                    packed_query_position_ids=position_ids,
                    packed_query_indexes=text_indexes,
                    past_key_values=text_cache,
                    key_values_lens=empty_kv_lens,
                    packed_key_value_indexes=empty_kv_indexes,
                    update_past_key_values=False,
                    is_causal=True,
                    **extra_inputs,
                )
                self._reca_context_text_hidden = (
                    text_output.packed_query_sequence.detach()
                )
            self._reca_context_text_key = cache_key

        generation_features = F.normalize(generation_hidden.float(), dim=-1)
        text_features = F.normalize(
            self._reca_context_text_hidden.to(generation_hidden.device).float(),
            dim=-1,
        )
        similarity_matrix = torch.matmul(
            text_features,
            generation_features.transpose(0, 1),
        )
        temperature = float(os.environ.get("BAGEL_INTERNAL_FEATURE_TEMP", "0.05"))
        token_weights = torch.softmax(similarity_matrix / temperature, dim=1)
        token_similarity = (token_weights * similarity_matrix).sum(dim=1).mean()
        # The last causal state summarizes the ordered ideal description.
        summary_similarity = (
            token_weights[-1] * similarity_matrix[-1]
        ).sum()
        similarity = 0.5 * (summary_similarity + token_similarity)
        loss = 1.0 - similarity

        print(
            f"contextual hidden input: generation_tokens={len(generation_features)}, "
            f"text_tokens={len(text_features)}, summary_sim={summary_similarity.item():.6g}, "
            f"token_sim={token_similarity.item():.6g}, temp={temperature:g}"
        )
        return loss, target_text

    def InternalFeatureAlignmentLoss(
        self,
        x_t_0,
        target_text,
        vae_model,
        image_shape,
        image_transform,
        tokenizer,
        max_text_tokens=192,
        vit_max_side=168,
    ):
        """Align BAGEL connector features with its own Qwen token embeddings."""
        device = next(self.parameters()).device
        image = self._decode_reca_latent_to_tensor(
            x_t_0,
            image_shape,
            vae_model,
        )
        image_tensor = image[0]
        image_h, image_w = image_tensor.shape[-2:]
        if max(image_h, image_w) > vit_max_side:
            scale = vit_max_side / max(image_h, image_w)
            resized_h = max(
                self.vit_patch_size,
                round(image_h * scale / self.vit_patch_size) * self.vit_patch_size,
            )
            resized_w = max(
                self.vit_patch_size,
                round(image_w * scale / self.vit_patch_size) * self.vit_patch_size,
            )
            if os.environ.get("BAGEL_STRICT_DETERMINISTIC", "0") == "1":
                image_tensor = F.interpolate(
                    image_tensor.unsqueeze(0),
                    size=(resized_h, resized_w),
                    mode="bilinear",
                    align_corners=False,
                    antialias=False,
                )[0]
            else:
                image_tensor = F.interpolate(
                    image_tensor.unsqueeze(0),
                    size=(resized_h, resized_w),
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                )[0]

        mean = torch.as_tensor(
            image_transform.normalize_transform.mean,
            device=image_tensor.device,
            dtype=image_tensor.dtype,
        )[:, None, None]
        std = torch.as_tensor(
            image_transform.normalize_transform.std,
            device=image_tensor.device,
            dtype=image_tensor.dtype,
        )[:, None, None]
        image_tensor = (image_tensor - mean) / std

        vit_position_ids = self.get_flattened_position_ids(
            image_tensor.shape[-2],
            image_tensor.shape[-1],
            self.vit_patch_size,
            max_num_patches_per_side=self.vit_max_num_patch_per_side,
        ).to(device)
        packed_vit_tokens = patchify(image_tensor, self.vit_patch_size).to(device)
        num_vit_tokens = packed_vit_tokens.shape[0]
        vit_token_seqlens = torch.tensor(
            [num_vit_tokens],
            dtype=torch.int,
            device=device,
        )
        cu_seqlens = F.pad(
            torch.cumsum(vit_token_seqlens, dim=0),
            (1, 0),
        ).to(torch.int32)
        image_features = self.vit_model(
            packed_pixel_values=packed_vit_tokens,
            packed_flattened_position_ids=vit_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=num_vit_tokens,
        )
        image_features = self.connector(image_features).float()

        text_ids = tokenizer.encode(target_text)[:max_text_tokens]
        if not text_ids:
            raise RuntimeError("Internal feature target produced no text tokens")
        text_ids = torch.tensor(text_ids, dtype=torch.long, device=device)
        text_features = self.language_model.model.embed_tokens(text_ids).float()

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        image_global = F.normalize(image_features.mean(dim=0), dim=0)
        text_global = F.normalize(text_features.mean(dim=0), dim=0)
        global_similarity = torch.sum(image_global * text_global)
        token_similarity = torch.matmul(text_features, image_features.transpose(0, 1))
        token_similarity = token_similarity.max(dim=1).values.mean()
        similarity = 0.5 * (global_similarity + token_similarity)
        loss = 1.0 - similarity

        print(
            f"internal feature input: {image_tensor.shape[-2]}x"
            f"{image_tensor.shape[-1]}, image_tokens={num_vit_tokens}, "
            f"text_tokens={len(text_ids)}, global_sim={global_similarity.item():.6g}, "
            f"token_sim={token_similarity.item():.6g}"
        )
        return loss, target_text

    def _prepare_understanding_ce_visual_context(
        self,
        image,
        image_transform,
        new_token_ids,
        ce_vit_max_side,
    ):
        """Encode one differentiable image prefix for multiple CE objectives."""
        device = next(self.parameters()).device

        def token_id(name):
            value = new_token_ids[name]
            return int(value.item()) if torch.is_tensor(value) else int(value)

        image_tensor = image[0]
        image_h, image_w = image_tensor.shape[-2:]
        if max(image_h, image_w) > ce_vit_max_side:
            scale = ce_vit_max_side / max(image_h, image_w)
            resized_h = max(
                self.vit_patch_size,
                round(image_h * scale / self.vit_patch_size) * self.vit_patch_size,
            )
            resized_w = max(
                self.vit_patch_size,
                round(image_w * scale / self.vit_patch_size) * self.vit_patch_size,
            )
            if os.environ.get("BAGEL_STRICT_DETERMINISTIC", "0") == "1":
                image_tensor = F.interpolate(
                    image_tensor.unsqueeze(0),
                    size=(resized_h, resized_w),
                    mode="bilinear",
                    align_corners=False,
                    antialias=False,
                )[0]
            else:
                image_tensor = F.interpolate(
                    image_tensor.unsqueeze(0),
                    size=(resized_h, resized_w),
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                )[0]
        mean = torch.as_tensor(
            image_transform.normalize_transform.mean,
            device=image_tensor.device,
            dtype=image_tensor.dtype,
        )[:, None, None]
        std = torch.as_tensor(
            image_transform.normalize_transform.std,
            device=image_tensor.device,
            dtype=image_tensor.dtype,
        )[:, None, None]
        image_tensor = (image_tensor - mean) / std

        vit_position_ids = self.get_flattened_position_ids(
            image_tensor.shape[-2],
            image_tensor.shape[-1],
            self.vit_patch_size,
            max_num_patches_per_side=self.vit_max_num_patch_per_side,
        ).to(device)
        packed_vit_tokens = patchify(image_tensor, self.vit_patch_size).to(device)
        num_vit_tokens = packed_vit_tokens.shape[0]
        print(
            f"ce vit input: {image_tensor.shape[-2]}x{image_tensor.shape[-1]}, "
            f"tokens: {num_vit_tokens}"
        )
        vit_token_seqlens = torch.tensor(
            [num_vit_tokens], dtype=torch.int, device=device
        )
        cu_seqlens = F.pad(
            torch.cumsum(vit_token_seqlens, dim=0), (1, 0)
        ).to(torch.int32)
        packed_vit_embed = self.vit_model(
            packed_pixel_values=packed_vit_tokens,
            packed_flattened_position_ids=vit_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=num_vit_tokens,
        )
        packed_vit_embed = self.connector(packed_vit_embed)
        packed_vit_embed = packed_vit_embed + self.vit_pos_embed(vit_position_ids)

        image_boundary_ids = torch.tensor(
            [token_id("start_of_image"), token_id("end_of_image")],
            dtype=torch.long,
            device=device,
        )
        image_boundary_embed = self.language_model.model.embed_tokens(
            image_boundary_ids
        )
        packed_sequence = image_boundary_embed.new_zeros(
            (num_vit_tokens + 2, self.hidden_size)
        )
        packed_sequence[0] = image_boundary_embed[0]
        packed_sequence[-1] = image_boundary_embed[1]
        packed_sequence[1:-1] = packed_vit_embed.to(packed_sequence.dtype)

        ce_cache = NaiveCache(self.config.llm_config.num_hidden_layers)
        image_seqlens = torch.tensor(
            [num_vit_tokens + 2], dtype=torch.int, device=device
        )
        image_indexes = torch.arange(
            num_vit_tokens + 2, dtype=torch.long, device=device
        )
        image_position_ids = torch.zeros(
            num_vit_tokens + 2, dtype=torch.long, device=device
        )
        empty_kv_indexes = torch.empty(0, dtype=torch.long, device=device)
        empty_kv_lens = torch.zeros(1, dtype=torch.int, device=device)
        extra_inputs = {"mode": "und"} if self.use_moe else {}
        image_output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=image_seqlens,
            packed_query_position_ids=image_position_ids,
            packed_query_indexes=image_indexes,
            past_key_values=ce_cache,
            key_values_lens=empty_kv_lens,
            packed_key_value_indexes=empty_kv_indexes,
            update_past_key_values=True,
            is_causal=False,
            **extra_inputs,
        )
        return image_output.past_key_values, [num_vit_tokens + 2], [1]

    def UnderstandingCELoss(
        self,
        x_t_0,
        target_text,
        vae_model,
        image_shape,
        image_transform,
        tokenizer,
        new_token_ids,
        ce_max_tokens=48,
        ce_vit_max_side=224,
        instruction=None,
        contrast_target_text=None,
        umm_dropbp_layers=(),
        decoded_image=None,
        prepared_visual_context=None,
        return_visual_context=False,
    ):
        """Compute image-caption CE through the real VAE -> ViT understanding path."""
        device = next(self.parameters()).device
        vae_device = next(vae_model.parameters()).device
        umm_model = self.language_model.model
        previous_dropbp_layers = getattr(umm_model, "umm_dropbp_layers", ())
        umm_model.umm_dropbp_layers = tuple(umm_dropbp_layers)

        def token_id(name):
            value = new_token_ids[name]
            return int(value.item()) if torch.is_tensor(value) else int(value)

        if prepared_visual_context is None:
            if decoded_image is None:
                image = self._decode_reca_latent_to_tensor(
                    x_t_0,
                    image_shape,
                    vae_model,
                )
            else:
                image = decoded_image
            prepared_visual_context = self._prepare_understanding_ce_visual_context(
                image,
                image_transform,
                new_token_ids,
                ce_vit_max_side,
            )

        base_cache, base_kvlens, base_rope = prepared_visual_context
        ce_cache = NaiveCache(base_cache.num_layers)
        ce_cache.key_cache = dict(base_cache.key_cache)
        ce_cache.value_cache = dict(base_cache.value_cache)
        curr_kvlens = list(base_kvlens)
        curr_rope = list(base_rope)
        extra_inputs = {"mode": "und"} if self.use_moe else {}

        if instruction is None:
            instruction = (
                "Describe this image in one concise paragraph with concrete visual "
                "details and atmosphere. Only one paragraph is required."
            )

        ce_cache, curr_kvlens, curr_rope = self._ce_cache_update_text(
            ce_cache,
            instruction,
            curr_kvlens,
            curr_rope,
            tokenizer,
            new_token_ids,
            device,
        )

        encoded_target_ids = tokenizer.encode(target_text)
        if contrast_target_text is not None:
            encoded_contrast_ids = tokenizer.encode(contrast_target_text)
            if not encoded_target_ids or not encoded_contrast_ids:
                raise RuntimeError("Contrastive CE targets produced no tokens")
            target_ids = encoded_target_ids[:1]
            contrast_ids = encoded_contrast_ids[:1]
            input_ids = [token_id("bos_token_id")]
        else:
            target_was_truncated = len(encoded_target_ids) > ce_max_tokens
            target_ids = encoded_target_ids[:ce_max_tokens]
            if not target_was_truncated:
                target_ids = target_ids + [token_id("eos_token_id")]
            input_ids = [token_id("bos_token_id")] + target_ids[:-1]
        query_lens = torch.tensor([len(input_ids)], dtype=torch.int, device=device)
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
        label_ids = torch.tensor(target_ids, dtype=torch.long, device=device)

        curr_kvlen = int(curr_kvlens[0])
        curr_position = int(curr_rope[0])
        packed_query_indexes = torch.arange(curr_kvlen, curr_kvlen + len(input_ids), dtype=torch.long, device=device)
        packed_key_value_indexes = torch.arange(curr_kvlen, dtype=torch.long, device=device)
        packed_query_position_ids = torch.arange(curr_position, curr_position + len(input_ids), dtype=torch.long, device=device)
        key_values_lens = torch.tensor([curr_kvlen], dtype=torch.int, device=device)

        packed_text_embedding = self.language_model.model.embed_tokens(input_ids)
        output = self.language_model.forward_inference(
            packed_query_sequence=packed_text_embedding,
            query_lens=query_lens,
            packed_query_position_ids=packed_query_position_ids,
            packed_query_indexes=packed_query_indexes,
            past_key_values=ce_cache,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=False,
            is_causal=True,
            **extra_inputs,
        )
        logits = self.language_model.lm_head(output.packed_query_sequence)
        if contrast_target_text is not None:
            contrast_label_ids = torch.tensor(
                contrast_ids,
                dtype=torch.long,
                device=device,
            )
            target_loss = F.cross_entropy(logits.float(), label_ids)
            contrast_loss = F.cross_entropy(logits.float(), contrast_label_ids)
            loss = target_loss - contrast_loss
        else:
            token_losses = F.cross_entropy(logits.float(), label_ids, reduction="none")
            loss = token_losses.mean()

        umm_model.umm_dropbp_layers = previous_dropbp_layers
        if return_visual_context:
            return loss, target_text, prepared_visual_context
        return loss, target_text
    
    @torch.no_grad()
    def calc_clip_with_prompt_nograd(self, prompt, img_latent, vae_model, image_shapes, tokenizer, new_token_ids, image_transform, clip_model, clip_processor, use_longclip, use_fgclip):
        def decode_image(latent, image_shape, vae_model):
                H, W = image_shape
                h, w = H // self.latent_downsample, W // self.latent_downsample
                latent = latent.reshape(1, h, w, self.latent_patch_size, self.latent_patch_size, self.latent_channel)
                latent = torch.einsum("nhwpqc->nchpwq", latent)
                latent = latent.reshape(1, self.latent_channel, h * self.latent_patch_size, w * self.latent_patch_size)
                image = vae_model.decode(latent.to(torch.bfloat16).to("cuda"))
                image = (image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
                image = Image.fromarray((image).to(torch.uint8).cpu().numpy())
                return image

        def _decode_latent_to_tensor(latent, image_shape, vae_model):
            H, W = image_shape
            h, w = H // self.latent_downsample, W // self.latent_downsample
            latent = latent.reshape(1, h, w, self.latent_patch_size, self.latent_patch_size, self.latent_channel)
            latent = torch.einsum("nhwpqc->nchpwq", latent)
            latent = latent.reshape(1, self.latent_channel, h * self.latent_patch_size, w * self.latent_patch_size)
            image_tensor = vae_model.decode(latent.to(torch.bfloat16).to("cuda"))
            image_tensor = (image_tensor * 0.5 + 0.5).clamp(0, 1)
            return image_tensor
        
        img = decode_image(img_latent[0], image_shapes, vae_model)
        img_tensor = _decode_latent_to_tensor(img_latent[0], image_shapes, vae_model)
        # breakpoint()
        img.save(f"./debug/debug_middle.png")
        vlm_output_text = prompt

        # vlm_output_text = "a photo of three buses"
        # breakpoint()
        if use_longclip:
            text_inputs = longclip.tokenize([vlm_output_text]).to("cuda")
            image_inputs = F.interpolate(
                                            img_tensor,
                                            size=(224, 224), # CLIP默认输入尺寸
                                            mode='bilinear',
                                            align_corners=False # 对于下采样，通常设置为 False
                                        ).to(torch.bfloat16)
            image_embeds = clip_model.encode_image(image_inputs)
            text_embeds = clip_model.encode_text(text_inputs)
        elif use_fgclip:
            def determine_max_value(image):
                _, _, w,h = image.shape
                max_val = (w//16)*(h//16)
                if max_val > 784:
                    return 1024
                elif max_val > 576:
                    return 784
                elif max_val > 256:
                    return 576
                elif max_val > 128:
                    return 256
                else:
                    return 128
            # breakpoint()
            print(f"use fgclip")
            image_input = clip_processor['image_processor'](images=img_tensor * 255, max_num_patches=determine_max_value(img_tensor), return_tensors="pt").to("cuda")
            caption_input = clip_processor['clip_tokenizer']([vlm_output_text], padding="max_length", max_length=196, truncation=True, return_tensors="pt").to('cuda')
            image_embeds = clip_model.get_image_features(**image_input)
            text_embeds = clip_model.get_text_features(**caption_input,walk_type="long")
        else:
            text_inputs = clip_processor(text=vlm_output_text, return_tensors="pt", padding=True).to("cuda")
            image_inputs = clip_processor(images=img_tensor.to(torch.float32), do_rescale=False, return_tensors="pt").to("cuda")
            image_inputs['pixel_values'] = F.interpolate(
                                            img_tensor,
                                            size=(224, 224), # CLIP默认输入尺寸
                                            mode='bilinear',
                                            align_corners=False # 对于下采样，通常设置为 False
                                        ).to(torch.bfloat16)
            image_embeds = clip_model.get_image_features(**image_inputs)
            text_embeds = clip_model.get_text_features(**text_inputs)

        image_embeds_norm = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds_norm = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        similarity = image_embeds_norm @ text_embeds_norm.T

        return 1 - similarity, vlm_output_text
    
    def RecALoss(self, prompt, img_latent, vae_model, image_shapes, tokenizer, new_token_ids, image_transform, clip_model, clip_processor, use_longclip, use_fgclip, extra_p, decoded_image=None):
        def _decode_latent_to_tensor(latent, image_shape, vae_model):
            H, W = image_shape
            h, w = H // self.latent_downsample, W // self.latent_downsample
            latent = latent.reshape(1, h, w, self.latent_patch_size, self.latent_patch_size, self.latent_channel)
            latent = torch.einsum("nhwpqc->nchpwq", latent)
            latent = latent.reshape(1, self.latent_channel, h * self.latent_patch_size, w * self.latent_patch_size)
            vae_device = next(vae_model.parameters()).device
            image_tensor = vae_model.decode(latent.to(torch.bfloat16).to(vae_device))
            image_tensor = (image_tensor * 0.5 + 0.5).clamp(0, 1)
            return image_tensor

        def tensor_to_pil(image_tensor):
            image = image_tensor.detach()[0].permute(1, 2, 0) * 255
            return Image.fromarray(image.to(torch.uint8).cpu().numpy())
        
        if decoded_image is None:
            img_tensor = _decode_latent_to_tensor(
                img_latent[0],
                image_shapes,
                vae_model,
            )
        else:
            img_tensor = decoded_image
        # breakpoint()
        # vlm_output_text=prompt
        if extra_p is None:
            img = tensor_to_pil(img_tensor)
            img.save(f"./debug/debug_middle.png")
            with torch.no_grad():
                vlm_output_text = self.chat(tokenizer=tokenizer,
                    new_token_ids=new_token_ids,
                    image_transform=image_transform,
                    images=[img],#images=[img],
                    prompt=f"Based on the text prompt '{prompt}' and the provided noisy intermediate image, describe the envisioned perfect final image in one concise paragraph. Ignore noise and artifacts, focusing strictly on the visual details and atmosphere. If there is any conflict between the visual features and the text prompt, you must strictly follow the text prompt. Only one paragraph is required.",
                    max_length=192)
        else :
            vlm_output_text=extra_p

        if "键" in vlm_output_text:
            vlm_output_text = prompt
        # breakpoint()
        clip_device = next(clip_model.parameters()).device
        if use_longclip:
            image_inputs = F.interpolate(
                                            img_tensor.to(clip_device),
                                            size=(224, 224), # CLIP默认输入尺寸
                                            mode='bilinear',
                                            align_corners=False # 对于下采样，通常设置为 False
                                        ).to(torch.bfloat16)
            image_embeds = clip_model.encode_image(image_inputs)
            if self.self_prompt != vlm_output_text or self.prompt_clip_feature is None:
                text_inputs = longclip.tokenize([vlm_output_text]).to(clip_device)
                with torch.no_grad():
                    self.prompt_clip_feature = clip_model.encode_text(text_inputs).detach()
                self.self_prompt = vlm_output_text
            text_embeds = self.prompt_clip_feature
        elif use_fgclip:
            def determine_max_value(image):
                _, _, w,h = image.shape
                max_val = (w//16)*(h//16)
                if max_val > 784:
                    return 1024
                elif max_val > 576:
                    return 784
                elif max_val > 256:
                    return 576
                elif max_val > 128:
                    return 256
                else:
                    return 128
            # breakpoint()
            print(f"use fgclip")
            image_input = clip_processor['image_processor'](images=img_tensor * 255, max_num_patches=determine_max_value(img_tensor), return_tensors="pt").to(clip_device)
            caption_input = clip_processor['clip_tokenizer']([vlm_output_text], padding="max_length", max_length=196, truncation=True, return_tensors="pt").to(clip_device)
            image_embeds = clip_model.get_image_features(**image_input)
            text_embeds = clip_model.get_text_features(**caption_input,walk_type="long")
        else:
            text_inputs = clip_processor(text=vlm_output_text, return_tensors="pt", padding=True).to(clip_device)
            image_inputs = clip_processor(images=img_tensor.to(torch.float32), do_rescale=False, return_tensors="pt").to(clip_device)
            image_inputs['pixel_values'] = F.interpolate(
                                            img_tensor.to(clip_device),
                                            size=(224, 224), # CLIP默认输入尺寸
                                            mode='bilinear',
                                            align_corners=False # 对于下采样，通常设置为 False
                                        ).to(torch.bfloat16)
            image_embeds = clip_model.get_image_features(**image_inputs)
            text_embeds = clip_model.get_text_features(**text_inputs)

        image_embeds_norm = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds_norm = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        similarity = image_embeds_norm @ text_embeds_norm.T

        return 1 - similarity, vlm_output_text
