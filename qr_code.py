import argparse
import json
import os

import numpy as np
import pandas as pd
from PIL import Image
import torch
from tqdm import tqdm

from inference import initialize_full_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge two JSONL result sets using BAGEL understanding CE."
    )
    parser.add_argument("--file_a", default="merged_best_results_union4.jsonl")
    parser.add_argument(
        "--file_b",
        default=(
            "/home/ma-user/work/wx1468559/geneval/Bagel/"
            "outputs_bagel_set1seed_dual_h.jsonl"
        ),
    )
    parser.add_argument("--output_file", default="merged_best_results_union4.jsonl")
    parser.add_argument(
        "--model_path",
        default="./pretrained_models/BAGEL-7B-MoT",
    )
    parser.add_argument("--ce_max_tokens", type=int, default=192)
    parser.add_argument("--vit_max_side", type=int, default=168)
    return parser.parse_args()


def load_and_prepare(filepath):
    dataframe = pd.read_json(filepath, orient="records", lines=True)
    dataframe["meta_key"] = dataframe["metadata"].apply(
        lambda value: (
            json.dumps(value, sort_keys=True) if isinstance(value, dict) else value
        )
    )
    dataframe["sample_id"] = dataframe["filename"].apply(os.path.basename)
    return dataframe


class BagelCEScorer:
    def __init__(self, model_path, ce_max_tokens, vit_max_side):
        (
            self.model,
            self.vae_model,
            self.tokenizer,
            self.new_token_ids,
            _,
            self.vit_transform,
        ) = initialize_full_model(model_path)
        self.vae_model = self.vae_model.to("cpu")
        self.ce_max_tokens = ce_max_tokens
        self.vit_max_side = vit_max_side
        self.cache = {}
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def image_to_tensor(image):
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        return (
            torch.from_numpy(array)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .contiguous()
            .to(torch.bfloat16)
        )

    def score(self, image_path, prompt):
        cache_key = (image_path, prompt)
        if cache_key in self.cache:
            return self.cache[cache_key]
        if not os.path.exists(image_path):
            print(f"  [warning] Image does not exist: {image_path}")
            return float("inf")

        try:
            with Image.open(image_path) as image:
                width, height = image.size
                image_tensor = self.image_to_tensor(image)
            with torch.no_grad():
                loss, _ = self.model.UnderstandingCELoss(
                    x_t_0=None,
                    target_text=prompt,
                    vae_model=self.vae_model,
                    image_shape=(height, width),
                    image_transform=self.vit_transform,
                    tokenizer=self.tokenizer,
                    new_token_ids=self.new_token_ids,
                    ce_max_tokens=self.ce_max_tokens,
                    ce_vit_max_side=self.vit_max_side,
                    umm_dropbp_layers=(),
                    decoded_image=image_tensor,
                )
            score = loss.float().item()
            self.cache[cache_key] = score
            return score
        except Exception as error:
            print(f"  [error] Failed to score {image_path}: {error}")
            return float("inf")


def main():
    args = parse_args()
    print("Loading BAGEL understanding model...")
    scorer = BagelCEScorer(
        args.model_path,
        args.ce_max_tokens,
        args.vit_max_side,
    )

    print("Reading result files A and B...")
    df_a = load_and_prepare(args.file_a)
    df_b = load_and_prepare(args.file_b)
    all_keys = pd.concat(
        [df_a[["meta_key", "sample_id"]], df_b[["meta_key", "sample_id"]]]
    ).drop_duplicates()

    final_results = []
    stats = {"a_only": 0, "b_only": 0, "a_better": 0, "b_better": 0}
    for _, key_row in tqdm(
        all_keys.iterrows(),
        total=len(all_keys),
        desc="Selecting by BAGEL CE",
    ):
        meta_key = key_row["meta_key"]
        sample_id = key_row["sample_id"]
        match_a = df_a[
            (df_a["meta_key"] == meta_key) & (df_a["sample_id"] == sample_id)
        ]
        match_b = df_b[
            (df_b["meta_key"] == meta_key) & (df_b["sample_id"] == sample_id)
        ]

        if not match_a.empty and match_b.empty:
            selected_row = match_a.iloc[0].to_dict()
            stats["a_only"] += 1
        elif match_a.empty and not match_b.empty:
            selected_row = match_b.iloc[0].to_dict()
            stats["b_only"] += 1
        else:
            row_a = match_a.iloc[0]
            row_b = match_b.iloc[0]
            metadata = json.loads(meta_key) if isinstance(meta_key, str) else meta_key
            prompt = metadata["prompt"]
            ce_a = scorer.score(row_a["filename"], prompt)
            ce_b = scorer.score(row_b["filename"], prompt)
            if ce_a <= ce_b:
                selected_row = row_a.to_dict()
                stats["a_better"] += 1
            else:
                selected_row = row_b.to_dict()
                stats["b_better"] += 1

        selected_row.pop("meta_key", None)
        selected_row.pop("sample_id", None)
        final_results.append(selected_row)

    with open(args.output_file, "w", encoding="utf-8") as output_file:
        for entry in final_results:
            output_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print("\n" + "=" * 50)
    print("Union merge complete")
    print(f"Total samples: {len(final_results)}")
    print(f"A only: {stats['a_only']}")
    print(f"B only: {stats['b_only']}")
    print(f"A selected by lower CE: {stats['a_better']}")
    print(f"B selected by lower CE: {stats['b_better']}")
    print(f"Output: {args.output_file}")
    print("=" * 50)


if __name__ == "__main__":
    main()
