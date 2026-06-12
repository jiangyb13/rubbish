Traceback (most recent call last):
  File "/home/ma-user/work/wx1468559/Bagel-Reca/inference.py", line 262, in <module>
    output_dict = inferencer.resc_for_gen(prompt=prompt, think=False, use_longclip=use_longclip, use_fgclip=use_fgclip, re_update_num=3, update_lr=[(5,10)],use_save_pic=False, update_scale=300, use_lookback=False, lookback_steps=10, save_grad=False, **inference_hyper)
  File "/home/ma-user/work/wx1468559/Bagel-Reca/inferencer.py", line 621, in resc_for_gen
    img = self.gen_image_reca(
  File "/home/ma-user/work/wx1468559/Bagel-Reca/inferencer.py", line 350, in gen_image_reca
    unpacked_latent = self.model.generate_image_reca(
  File "/home/ma-user/work/wx1468559/Bagel-Reca/modeling/bagel/bagel3.py", line 1245, in generate_image_reca
    loss, vlm_output_text = self.ce_guidance_loss(
  File "/home/ma-user/work/wx1468559/Bagel-Reca/modeling/bagel/bagel3.py", line 2255, in ce_guidance_loss
    ce_loss = self._compute_ce_from_tensor(
  File "/home/ma-user/work/wx1468559/Bagel-Reca/modeling/bagel/bagel3.py", line 2220, in _compute_ce_from_tensor
    model_outputs = self._forward_ce_understanding(
  File "/home/ma-user/work/wx1468559/Bagel-Reca/modeling/bagel/bagel3.py", line 2030, in _forward_ce_understanding
    block_mask = create_block_mask(
  File "/root/anaconda3/envs/bagel/lib/python3.10/site-packages/torch/nn/attention/flex_attention.py", line 850, in create_block_mask
    partial_block_mask, full_block_mask = inner_func(
  File "/root/anaconda3/envs/bagel/lib/python3.10/site-packages/torch/nn/attention/flex_attention.py", line 776, in _create_block_mask_inner
    partial_block_mask, full_block_mask = _convert_mask_to_block_mask(
  File "/root/anaconda3/envs/bagel/lib/python3.10/site-packages/torch/nn/attention/flex_attention.py", line 625, in _convert_mask_to_block_mask
    mask_block_sum = mask.sum(
  File "/root/anaconda3/envs/bagel/lib/python3.10/site-packages/torch/_higher_order_ops/flex_attention.py", line 85, in __torch_function__
    return func(*args, **(kwargs or {}))
RuntimeError: CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [68,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [69,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [70,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [71,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [72,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [73,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [74,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [75,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [76,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [77,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [78,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [79,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [80,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [81,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [82,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [83,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [84,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [85,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [86,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [87,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [88,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [89,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [90,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [91,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [92,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [93,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [94,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [95,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [96,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [97,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [98,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [99,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [100,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [101,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [102,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [103,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [104,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [105,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [106,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [107,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [108,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [109,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [110,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [111,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [112,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [113,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [114,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [115,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [116,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [117,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [118,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [119,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [120,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [121,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [122,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [123,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [124,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [125,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [126,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [9,0,0], thread: [127,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
(bagel) develop-iqp2ih40-5f986b85dc-4x594# 
