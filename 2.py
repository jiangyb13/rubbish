
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
        lookback_steps=0
    ):
        # breakpoint()
        self.eval()
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
            loss_list = []
            flag = False

            x_t_list = []
            x_t_score = []
            x_t_loss= []
            for re_update in range(re_update_num_ + 1):
                final_flag = True
                if update_flag and re_update < re_update_num_:
                    final_flag = False
                # with torch.set_grad_enabled(True):
                with torch.set_grad_enabled(not final_flag):
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
                        final_flag=final_flag
                    )
                # breakpoint()
                x_t_1 = x_t - v_t.to(x_t.device) * dts[i].cpu()

                if use_lookback and update_flag:
                    x_t_lookback = x_t.detach()
                    x_t_0_lookback = x_t_lookback - v_t.to(x_t.device) * t
                    for ii, tt in tqdm(enumerate(timesteps[i:lookback_steps]), total=len(timesteps[i:lookback_steps])):
                        v_t_lookback = infer_one_step(x_t_lookback, tt)
                        x_t_lookback = x_t_lookback - v_t_lookback.to(x_t.device) * dts[i + ii].cpu()
                        x_t_0_lookback = x_t_lookback - v_t_lookback.to(x_t.device) * tt
                    x_t_0_latent = x_t_0_lookback.split((packed_seqlens - 2).tolist())
                    with torch.no_grad():
                        l, v = self.calc_clip_with_prompt_nograd(prompt, x_t_0_latent, vae_model, image_shapes, tokenizer, new_token_ids, image_transform, clip_model, clip_processor, use_longclip, use_fgclip)
                    
                    if use_save_pic:
                        def decode_image(latent, image_shape):
                            H, W = image_shape
                            h, w = H // self.latent_downsample, W // self.latent_downsample
                            latent = latent.reshape(1, h, w, self.latent_patch_size, self.latent_patch_size, self.latent_channel)
                            latent = torch.einsum("nhwpqc->nchpwq", latent)
                            latent = latent.reshape(1, self.latent_channel, h * self.latent_patch_size, w * self.latent_patch_size)
                            
                            image = vae_model.decode(latent.to(torch.bfloat16).to("cuda"))
                            image = (image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
                            image = Image.fromarray((image).to(torch.uint8).cpu().numpy())
                            return image
                        img = decode_image(x_t_0_latent[0], image_shapes)
                        import os
                        os.makedirs(f"./cmp_reca/{prompt.replace(' ', '_')}/", exist_ok=True)
                        img.save(f"./cmp_reca/{prompt.replace(' ', '_')}/{i}_{re_update}_lookback.png")

                    x_t_list.append(x_t_1.detach().requires_grad_())
                    x_t_loss.append(l.item())
                    # x_t_score.append(vlm_output_text)
                
                if not final_flag:
                    # grad = torch.autograd.grad(torch.sum(v_t * 1000), x_t)[0]
                    # breakpoint()
                    x_t_0 = x_t - v_t.to(x_t.device) * t
                    x_t_0_latent = x_t_0.split((packed_seqlens - 2).tolist())
                    # breakpoint()
                    loss, vlm_output_text = self.ce_guidance_loss(
                        prompt,
                        x_t_0_latent,
                        vae_model,
                        image_shapes,
                        tokenizer,
                        new_token_ids,
                        image_transform,
                    )
                    # vlm_output_text = self.chat_only(prompt, x_t_0_latent, vae_model, image_shapes, tokenizer, new_token_ids, image_transform, clip_model, clip_processor, use_longclip, use_fgclip)
                    # breakpoint()
                    loss_list.append(loss.item())
                    # loss = torch.sum(x_t_0)
                    # grad = torch.autograd.grad(loss, x_t_0_latent)[0]
                    # breakpoint()
                    grad = torch.autograd.grad(loss, x_t)[0]

                    if grad.norm(p=2) > 0.01:
                        flag = True
                        grad = grad * (0.01 / grad.norm(p=2))
                    
                    # if grad.norm(p=2) < 0.005:
                    #     grad = grad * (0.005 / grad.norm(p=2))
                    # breakpoint()
                    x_t = x_t - update_scale * grad
                        # breakpoint()
                    print(f"update_scale: {update_scale}")
                    x_t = x_t.detach().requires_grad_()

                # elif update_flag:
                #     with torch.no_grad():
                #         x_t_0 = x_t - v_t.to(x_t.device) * t
                #         x_t_0_latent = x_t_0.split((packed_seqlens - 2).tolist())
                #         loss, vlm_output_text = self.calc_clip_with_prompt_nograd(prompt, x_t_0_latent, vae_model, image_shapes, tokenizer, new_token_ids, image_transform, clip_model, clip_processor, use_longclip, use_fgclip)
                #         loss_list.append(loss.item())

                # if save_grad:
                #     import os
                #     os.makedirs(f"/home/ma-user/work/wx1468559/Bagel-Reca/gradints/{prompt.replace(' ', '_')}", exist_ok=True)
                #     torch.save(grad, f"/home/ma-user/work/wx1468559/Bagel-Reca/gradints/{prompt.replace(' ', '_')}/grad_{i}.pt")
                #     torch.save(x_t, f"/home/ma-user/work/wx1468559/Bagel-Reca/gradints/{prompt.replace(' ', '_')}/x_{i}.pt")
                
                # breakpoint()
                if use_save_pic and update_flag:
                    def decode_image(latent, image_shape):
                        H, W = image_shape
                        h, w = H // self.latent_downsample, W // self.latent_downsample
                        latent = latent.reshape(1, h, w, self.latent_patch_size, self.latent_patch_size, self.latent_channel)
                        latent = torch.einsum("nhwpqc->nchpwq", latent)
                        latent = latent.reshape(1, self.latent_channel, h * self.latent_patch_size, w * self.latent_patch_size)
                        
                        image = vae_model.decode(latent.to(torch.bfloat16).to("cuda"))
                        image = (image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
                        image = Image.fromarray((image).to(torch.uint8).cpu().numpy())
                        return image
                    x_0 = x_t - v_t.to(x_t.device) * t
                    x_0_latent = x_0.split((packed_seqlens - 2).tolist())
                    img = decode_image(x_0_latent[0], image_shapes)
                    import os
                    # os.makedirs("./debug/pic/", exist_ok=True)
                    # img.save(f"./debug/pic/{i}_{re_update}.png")
                    os.makedirs(f"./cmp_reca/{prompt.replace(' ', '_')}/", exist_ok=True)
                    img.save(f"./cmp_reca/{prompt.replace(' ', '_')}/{i}_{re_update}.png")
                elif use_save_pic:
                    def decode_image(latent, image_shape):
                        H, W = image_shape
                        h, w = H // self.latent_downsample, W // self.latent_downsample
                        latent = latent.reshape(1, h, w, self.latent_patch_size, self.latent_patch_size, self.latent_channel)
                        latent = torch.einsum("nhwpqc->nchpwq", latent)
                        latent = latent.reshape(1, self.latent_channel, h * self.latent_patch_size, w * self.latent_patch_size)
                        
                        image = vae_model.decode(latent.to(torch.bfloat16).to("cuda"))
                        image = (image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
                        image = Image.fromarray((image).to(torch.uint8).cpu().numpy())
                        return image
                    x_0 = x_t - v_t.to(x_t.device) * t
                    x_0_latent = x_0.split((packed_seqlens - 2).tolist())
                    img = decode_image(x_0_latent[0], image_shapes)
                    import os
                    os.makedirs(f"./cmp_bagel/{prompt.replace(' ', '_')}/", exist_ok=True)
                    img.save(f"./cmp_bagel/{prompt.replace(' ', '_')}/{i}.png")

            print(f"denoising step {i}, loss: {loss_list}")
            # breakpoint()
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

    def _get_token_id(self, new_token_ids, key):
        value = new_token_ids[key]
        if torch.is_tensor(value):
            return int(value.item())
        return int(value)

    def _decode_latent_to_tensor(self, latent, image_shape, vae_model):
        H, W = image_shape
        h, w = H // self.latent_downsample, W // self.latent_downsample
        latent = latent.reshape(1, h, w, self.latent_patch_size, self.latent_patch_size, self.latent_channel)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, self.latent_channel, h * self.latent_patch_size, w * self.latent_patch_size)
        image_tensor = vae_model.decode(latent.to(torch.bfloat16).to("cuda"))
        image_tensor = (image_tensor * 0.5 + 0.5).clamp(0, 1)
        return image_tensor

    def _tensor_to_pil(self, image_tensor):
        tensor = image_tensor.detach().cpu()[0]
        image = (tensor * 255).clamp(0, 255).permute(1, 2, 0).to(torch.uint8).numpy()
        image = Image.fromarray(image)
        return pil_img2rgb(image)

    def _resize_tensor_like_transform(self, tensor, resize_transform):
        stride = getattr(resize_transform, 'stride', self.vit_patch_size)
        max_size = getattr(resize_transform, 'max_size', self.vit_max_num_patch_per_side * self.vit_patch_size)
        min_size = getattr(resize_transform, 'min_size', stride)
        max_pixels = getattr(resize_transform, 'max_pixels', max_size * max_size)

        def make_divisible(value):
            return max(stride, int(round(value / stride) * stride))

        def apply_scale(width, height, scale):
            new_width = make_divisible(width * scale)
            new_height = make_divisible(height * scale)
            return new_width, new_height

        _, height, width = tensor.shape
        scale = min(max_size / max(width, height), 1.0)
        scale = max(scale, min_size / min(width, height))
        new_width, new_height = apply_scale(width, height, scale)

        if new_width * new_height > max_pixels:
            scale = max_pixels / (new_width * new_height)
            new_width, new_height = apply_scale(new_width, new_height, scale)

        if max(new_width, new_height) > max_size:
            scale = max_size / max(new_width, new_height)
            new_width, new_height = apply_scale(new_width, new_height, scale)

        tensor = F.interpolate(
            tensor.unsqueeze(0), size=(new_height, new_width), mode="bilinear", align_corners=False
        ).squeeze(0)
        return tensor

    def _ensure_patch_aligned(self, tensor):
        _, height, width = tensor.shape
        max_side = self.vit_max_num_patch_per_side * self.vit_patch_size
        target_h = min(height, max_side)
        target_w = min(width, max_side)

        remainder_h = target_h % self.vit_patch_size
        remainder_w = target_w % self.vit_patch_size
        target_h = target_h - remainder_h
        target_w = target_w - remainder_w
        if target_h == 0:
            target_h = self.vit_patch_size
        if target_w == 0:
            target_w = self.vit_patch_size

        pad_h = max(0, target_h - height)
        pad_w = max(0, target_w - width)
        if pad_h > 0 or pad_w > 0:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='replicate')

        tensor = tensor[:, :target_h, :target_w]
        return tensor

    def _transform_tensor_for_vit(self, image_tensor, image_transform):
        tensor = image_tensor.squeeze(0).to(torch.float32)
        if image_transform is not None and hasattr(image_transform, 'resize_transform'):
            tensor = self._resize_tensor_like_transform(tensor, image_transform.resize_transform)
        else:
            target = self.vit_max_num_patch_per_side * self.vit_patch_size
            tensor = F.interpolate(
                tensor.unsqueeze(0), size=(target, target), mode="bilinear", align_corners=False
            ).squeeze(0)

        tensor = tensor.clamp(0, 1)
        if image_transform is not None and hasattr(image_transform, 'normalize_transform'):
            mean = torch.as_tensor(image_transform.normalize_transform.mean, device=tensor.device, dtype=tensor.dtype).view(-1, 1, 1)
            std = torch.as_tensor(image_transform.normalize_transform.std, device=tensor.device, dtype=tensor.dtype).view(-1, 1, 1)
        else:
            mean = torch.tensor([0.5, 0.5, 0.5], device=tensor.device, dtype=tensor.dtype).view(-1, 1, 1)
            std = torch.tensor([0.5, 0.5, 0.5], device=tensor.device, dtype=tensor.dtype).view(-1, 1, 1)
        tensor = (tensor - mean) / std
        tensor = self._ensure_patch_aligned(tensor)
        return tensor.to(self.language_model.model.embed_tokens.weight.dtype)

    def _forward_ce_understanding(
        self,
        sequence_length,
        packed_text_ids,
        packed_text_indexes,
        sample_lens,
        split_lens,
        attn_modes,
        packed_position_ids,
        packed_vit_tokens,
        packed_vit_token_indexes,
        packed_vit_position_ids,
        vit_token_seqlens,
        use_checkpoint=True,
    ):
        device = packed_text_ids.device
        model_module = self.language_model.model

        packed_text_embedding = model_module.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sequence_length, self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding.detach()

        sparse_mask = create_sparse_mask(sample_lens, split_lens, attn_modes, packed_text_embedding.device)
        seqlen = sum(sample_lens)
        block_mask = create_block_mask(
            sparse_mask,
            B=1,
            H=self.num_heads,
            Q_LEN=seqlen,
            KV_LEN=seqlen,
            device=packed_text_embedding.device,
            BLOCK_SIZE=128,
            _compile=False,
        )
        attention_mask = block_mask

        packed_und_token_indexes = packed_text_indexes
        needs_grad = False
        if self.config.visual_und and packed_vit_tokens is not None:
            cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0)).to(torch.int32)
            max_seqlen = torch.max(vit_token_seqlens).item()
            with torch.set_grad_enabled(True):
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
            packed_und_token_indexes = torch.cat([packed_text_indexes, packed_vit_token_indexes], dim=0)
            needs_grad = packed_vit_token_embed.requires_grad

        if needs_grad:
            packed_sequence.requires_grad_(True)

        cos, sin = model_module.rotary_emb(packed_sequence, packed_position_ids.unsqueeze(0))
        packed_position_embeddings = (cos.squeeze(0).detach(), sin.squeeze(0).detach())

        extra_inputs = {}
        packed_gen_token_indexes = None
        if self.use_moe:
            empty_gen = packed_und_token_indexes.new_zeros((0,), dtype=torch.long)
            packed_gen_token_indexes = empty_gen
            extra_inputs.update(
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_gen_token_indexes,
            )

        def apply_layer(layer, seq):
            layer_kwargs = dict(
                packed_sequence=seq,
                sample_lens=sample_lens,
                attention_mask=attention_mask,
                packed_position_embeddings=packed_position_embeddings,
                **extra_inputs,
            )
            if hasattr(layer, 'forward_train'):
                was_training = layer.training
                if not was_training:
                    layer.train(True)
                try:
                    return layer.forward_train(**layer_kwargs)
                finally:
                    layer.train(was_training)
            return layer(**layer_kwargs)

        for layer in model_module.layers:
            if use_checkpoint and needs_grad:
                def layer_fn(seq, layer_ref=layer):
                    return apply_layer(layer_ref, seq)
                packed_sequence = checkpoint(layer_fn, packed_sequence, use_reentrant=False)
            else:
                packed_sequence = apply_layer(layer, packed_sequence)

        if self.use_moe:
            packed_sequence_ = torch.zeros_like(packed_sequence)
            packed_sequence_[packed_und_token_indexes] = model_module.norm(packed_sequence[packed_und_token_indexes])
            if model_module.config.freeze_und:
                packed_sequence_[packed_und_token_indexes] = packed_sequence_[packed_und_token_indexes].detach()
            packed_gen_token_indexes = extra_inputs.get('packed_gen_token_indexes')
            if packed_gen_token_indexes is not None and packed_gen_token_indexes.numel() > 0:
                packed_sequence_[packed_gen_token_indexes] = model_module.norm_moe_gen(
                    packed_sequence[packed_gen_token_indexes]
                )
            packed_sequence = packed_sequence_
        else:
            packed_sequence = model_module.norm(packed_sequence)

        return packed_sequence

    def _build_ce_forward_inputs(self, vit_tensor, question_text, answer_text, tokenizer, new_token_ids, device):
        start_of_image = self._get_token_id(new_token_ids, 'start_of_image')
        end_of_image = self._get_token_id(new_token_ids, 'end_of_image')
        bos = self._get_token_id(new_token_ids, 'bos_token_id')
        eos = self._get_token_id(new_token_ids, 'eos_token_id')

        vit_tokens = patchify(vit_tensor, self.vit_patch_size)
        num_img_tokens = vit_tokens.shape[0]
        vit_position_ids = self.get_flattened_position_ids(
            vit_tensor.size(1), vit_tensor.size(2),
            self.vit_patch_size,
            max_num_patches_per_side=self.vit_max_num_patch_per_side,
        )

        packed_text_ids, packed_text_indexes = [], []
        packed_vit_token_indexes = []
        packed_position_ids = []
        split_lens, attn_modes = [], []
        ce_positions, label_ids = [], []
        curr = 0
        curr_position_id = 0

        def append_image_block():
            nonlocal curr, curr_position_id
            split_start = curr
            packed_text_ids.append(start_of_image)
            packed_text_indexes.append(curr)
            curr += 1

            packed_vit_token_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens

            packed_text_ids.append(end_of_image)
            packed_text_indexes.append(curr)
            curr += 1

            split_len = curr - split_start
            split_lens.append(split_len)
            attn_modes.append("full")
            packed_position_ids.extend([curr_position_id] * split_len)
            curr_position_id += 1

        def append_text_block(text, enable_loss=False):
            nonlocal curr, curr_position_id
            text_ids = tokenizer.encode(text)
            shifted_ids = [bos] + text_ids
            split_start = curr
            for token in shifted_ids:
                packed_text_ids.append(token)
                packed_text_indexes.append(curr)
                if enable_loss:
                    ce_positions.append(curr)
                curr += 1

            packed_text_ids.append(eos)
            packed_text_indexes.append(curr)
            curr += 1

            if enable_loss:
                label_ids.extend(text_ids + [eos])

            split_len = curr - split_start
            split_lens.append(split_len)
            attn_modes.append("causal")
            packed_position_ids.extend(range(curr_position_id, curr_position_id + split_len))
            curr_position_id += split_len

        append_image_block()
        append_text_block(question_text, enable_loss=False)
        append_text_block(answer_text, enable_loss=True)

        sequence_length = curr
        ce_mask = torch.zeros(sequence_length, dtype=torch.bool, device=device)
        if ce_positions:
            ce_mask[ce_positions] = True

        packed_vit_tokens = vit_tokens.to(device=device, dtype=self.language_model.model.embed_tokens.weight.dtype)

        inputs = dict(
            sequence_length=sequence_length,
            sample_lens=[sequence_length],
            split_lens=split_lens,
            attn_modes=attn_modes,
            packed_text_ids=torch.tensor(packed_text_ids, dtype=torch.long, device=device),
            packed_text_indexes=torch.tensor(packed_text_indexes, dtype=torch.long, device=device),
            packed_position_ids=torch.tensor(packed_position_ids, dtype=torch.long, device=device),
            packed_vit_tokens=packed_vit_tokens,
            packed_vit_token_indexes=torch.tensor(packed_vit_token_indexes, dtype=torch.long, device=device),
            packed_vit_position_ids=vit_position_ids.to(device),
            vit_token_seqlens=torch.tensor([num_img_tokens], dtype=torch.int, device=device),
            ce_loss_indexes=ce_mask,
            packed_label_ids=torch.tensor(label_ids, dtype=torch.long, device=device),
        )
        return inputs

    def _compute_ce_from_tensor(self, image_tensor, question_text, answer_text, tokenizer, new_token_ids, image_transform):
        device = next(self.parameters()).device
        vit_tensor = self._transform_tensor_for_vit(image_tensor, image_transform)
        inputs = self._build_ce_forward_inputs(vit_tensor, question_text, answer_text, tokenizer, new_token_ids, device)
        model_outputs = self._forward_ce_understanding(
            sequence_length=inputs['sequence_length'],
            packed_text_ids=inputs['packed_text_ids'],
            packed_text_indexes=inputs['packed_text_indexes'],
            sample_lens=inputs['sample_lens'],
            split_lens=inputs['split_lens'],
            attn_modes=inputs['attn_modes'],
            packed_position_ids=inputs['packed_position_ids'],
            packed_vit_tokens=inputs['packed_vit_tokens'],
            packed_vit_token_indexes=inputs['packed_vit_token_indexes'],
            packed_vit_position_ids=inputs['packed_vit_position_ids'],
            vit_token_seqlens=inputs['vit_token_seqlens'],
            use_checkpoint=True,
        )
        ce_mask = inputs['ce_loss_indexes']
        label_ids = inputs['packed_label_ids']
        packed_ce_preds = self.language_model.lm_head(model_outputs[ce_mask])
        ce = F.cross_entropy(packed_ce_preds, label_ids, reduction='mean')
        return ce

    def ce_guidance_loss(self, prompt, img_latent, vae_model, image_shapes, tokenizer, new_token_ids, image_transform):
        image_tensor = self._decode_latent_to_tensor(img_latent[0], image_shapes, vae_model)
        pil_image = self._tensor_to_pil(image_tensor)

        vlm_output_text = self.chat(
            tokenizer=tokenizer,
            new_token_ids=new_token_ids,
            image_transform=image_transform,
            images=[pil_image],
            prompt=f"Based on the text prompt '{prompt}' and the provided noisy intermediate image, describe the envisioned perfect final image in one concise paragraph. Ignore noise and artifacts, focusing strictly on the visual details and atmosphere. If there is any conflict between the visual features and the text prompt, you must strictly follow the text prompt. Only one paragraph is required.",
            max_length=192,
        )
        if "键" in vlm_output_text or not vlm_output_text.strip():
            vlm_output_text = prompt

        ce_loss = self._compute_ce_from_tensor(
            image_tensor=image_tensor,
            question_text=f"Describe the perfect final image.",
            answer_text=vlm_output_text,
            tokenizer=tokenizer,
            new_token_ids=new_token_ids,
            image_transform=image_transform,
        )

        return ce_loss, vlm_output_text
    
    def RecALoss(self, prompt, img_latent, vae_model, image_shapes, tokenizer, new_token_ids, image_transform, clip_model, clip_processor, use_longclip, use_fgclip):
        img_tensor = self._decode_latent_to_tensor(img_latent[0], image_shapes, vae_model)
        img = self._tensor_to_pil(img_tensor)
        # breakpoint()
        img.save(f"./debug/debug_middle.png")
        vlm_output_text = self.chat(tokenizer=tokenizer,
            new_token_ids=new_token_ids,
            image_transform=image_transform,
            images=[img],#images=[img],
            prompt=f"Based on the text prompt '{prompt}' and the provided noisy intermediate image, describe the envisioned perfect final image in one concise paragraph. Ignore noise and artifacts, focusing strictly on the visual details and atmosphere. If there is any conflict between the visual features and the text prompt, you must strictly follow the text prompt. Only one paragraph is required.",
            max_length=192)
        if "键" in vlm_output_text:
            vlm_output_text = prompt
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


    