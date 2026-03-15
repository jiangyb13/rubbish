Traceback (most recent call last):
  File "/data/huanan/code/jwx1416454/HUAWEI_CrossPairDataset/main.py", line 132, in <module>
    seg_model = SAM2ImagePredictor(build_sam2("sam2_hiera_l.yaml",
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/huanan/code/jwx1416454/HUAWEI_CrossPairDataset/utils/sam2/sam2/build_sam.py", line 93, in build_sam2
    _load_checkpoint(model, ckpt_path)
  File "/data/huanan/code/jwx1416454/HUAWEI_CrossPairDataset/utils/sam2/sam2/build_sam.py", line 167, in _load_checkpoint
    missing_keys, unexpected_keys = model.load_state_dict(sd)
                                    ^^^^^^^^^^^^^^^^^^^^^
  File "/home/ma-user/anaconda3/envs/PyTorch-2.6.0/lib/python3.11/site-packages/omegaconf/dictconfig.py", line 355, in __getattr__
    self._format_and_raise(
  File "/home/ma-user/anaconda3/envs/PyTorch-2.6.0/lib/python3.11/site-packages/omegaconf/base.py", line 231, in _format_and_raise
    format_and_raise(
  File "/home/ma-user/anaconda3/envs/PyTorch-2.6.0/lib/python3.11/site-packages/omegaconf/_utils.py", line 899, in format_and_raise
    _raise(ex, cause)
  File "/home/ma-user/anaconda3/envs/PyTorch-2.6.0/lib/python3.11/site-packages/omegaconf/_utils.py", line 797, in _raise
    raise ex.with_traceback(sys.exc_info()[2])  # set env var OC_CAUSE=1 for full trace
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ma-user/anaconda3/envs/PyTorch-2.6.0/lib/python3.11/site-packages/omegaconf/dictconfig.py", line 351, in __getattr__
    return self._get_impl(
           ^^^^^^^^^^^^^^^
  File "/home/ma-user/anaconda3/envs/PyTorch-2.6.0/lib/python3.11/site-packages/omegaconf/dictconfig.py", line 442, in _get_impl
    node = self._get_child(
           ^^^^^^^^^^^^^^^^
  File "/home/ma-user/anaconda3/envs/PyTorch-2.6.0/lib/python3.11/site-packages/omegaconf/basecontainer.py", line 73, in _get_child
    child = self._get_node(
            ^^^^^^^^^^^^^^^
  File "/home/ma-user/anaconda3/envs/PyTorch-2.6.0/lib/python3.11/site-packages/omegaconf/dictconfig.py", line 480, in _get_node
    raise ConfigKeyError(f"Missing key {key!s}")
omegaconf.errors.ConfigAttributeError: Missing key load_state_dict
    full_key: model.load_state_dict
    object_type=dict
[ERROR] 2026-03-15-15:36:34 (PID:3100116, Device:-1, RankID:-1) ERR99999 UNKNOWN applicaiton exception
