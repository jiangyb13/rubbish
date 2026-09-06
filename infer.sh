if os.environ.get("MMDIT_VERIFY_TEXT_REF", "0") == "1":
    if self.training or self.fa_keep_prob != 1.0:
        raise RuntimeError("验证时需要 model.eval() 且 fa_keep_prob=1")

    with torch.no_grad():
        def probe_attn(pq, pk, pv, pmask):
            # 与正常 attention 使用相同的 kernel、scale 和 mask
            return torch_npu.npu_fusion_attention(
                pq, pk, pv, n_head,
                atten_mask=pmask,
                scale=(C // self.n_head) ** -0.5,
                keep_prob=1.0,
                input_layout="BNSD",
            )[0]

        # 1. 第一条分支：保留 text 的 V，video 的 V 置零。
        # Q、K、mask 不变，因此 softmax 分母不变。
        v_text_only = v.clone()
        v_text_only[:, :, :T, :] = 0

        text_contrib = probe_attn(
            q, k, v_text_only, mask
        )[:, :, :T, :]  # 只统计 video query 的输出
        del v_text_only

        # 2. 第二条分支：保留 ref 的 V，video 的 V 置零。
        # K 仍然包含 video + ref，不能删除 video 的 K。
        ref_contrib = probe_attn(
            q_x,
            torch.cat([k_x, k_ref], dim=2),
            torch.cat([torch.zeros_like(v_x), v_ref], dim=2),
            None,
        )

        # 3. 排除 CP 为整除而补齐的空 heads。
        local_heads = text_contrib.shape[1]
        cp_rank = (
            dist.get_rank(self.cp_group)
            if self.use_context_parallelism else 0
        )
        real_heads = max(
            0, min(local_heads, self.n_head - cp_rank * local_heads)
        )

        if real_heads > 0:
            a = text_contrib[:, :real_heads].float()
            b = ref_contrib[:, :real_heads].float()

            # [B, T]：每个 video token 在本 rank 上的平方范数、内积
            aa = a.square().sum(dim=(1, 3))
            bb = b.square().sum(dim=(1, 3))
            ab = (a * b).sum(dim=(1, 3))
            stats = torch.stack([aa, bb, ab], dim=-1)
            del a, b
        else:
            stats = torch.zeros(
                text_contrib.shape[0], T, 3,
                device=text_contrib.device, dtype=torch.float32,
            )

        # 所有 CP rank 都必须参与，先汇总内积，再计算余弦。
        if self.use_context_parallelism:
            dist.all_reduce(stats, group=self.cp_group)

        # 只由全局 rank 0 打印；CFG 的不同 batch 样本分别统计。
        if dist.get_rank() == 0:
            for sample_idx, s in enumerate(stats.cpu().double()):
                aa, bb, ab = s.unbind(dim=-1)
                text_norm = aa.sum().clamp_min(0).sqrt().item()
                ref_norm = bb.sum().clamp_min(0).sqrt().item()

                ratio = text_norm / ref_norm if ref_norm > 0 else None
                denom = text_norm * ref_norm
                cos_flat = (
                    max(-1.0, min(1.0, ab.sum().item() / denom))
                    if denom > 0 else None
                )

                valid = (aa > 0) & (bb > 0)
                cos_token = (
                    (ab[valid] / (aa[valid] * bb[valid]).sqrt())
                    .clamp(-1, 1).mean().item()
                    if valid.any() else None
                )

                print(
                    f"[text-ref] block={num_layer} sample={sample_idx} "
                    f"text_norm={text_norm:.6f} ref_norm={ref_norm:.6f} "
                    f"text/ref={ratio} "
                    f"cos_flat={cos_flat} cos_token={cos_token}",
                    flush=True,
                )
