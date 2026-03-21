Traceback (most recent call last):
  File "/data/huanan/code/jwx1416454/HUAWEI_CrossPairDataset/main.py", line 107, in <module>
    mainForOneShot()
  File "/data/huanan/code/jwx1416454/HUAWEI_CrossPairDataset/main.py", line 67, in mainForOneShot
    pipeline = OneShotProcessPipeline(config)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/huanan/code/jwx1416454/HUAWEI_CrossPairDataset/utils/one_shot.py", line 652, in __init__
    self.person_detector = PersonDetector(config)
                           ^^^^^^^^^^^^^^^^^^^^^^
  File "/data/huanan/code/jwx1416454/HUAWEI_CrossPairDataset/utils/one_shot.py", line 362, in __init__
    self.dino_model = AutoModel.from_pretrained(config.dinov3_model, device_map="auto")
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ma-user/anaconda3/envs/PyTorch-2.6.0/lib/python3.11/site-packages/transformers/models/auto/auto_factory.py", line 374, in from_pretrained
    return model_class.from_pretrained(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ma-user/anaconda3/envs/PyTorch-2.6.0/lib/python3.11/site-packages/transformers/modeling_utils.py", line 4001, in from_pretrained
    device_map = check_and_set_device_map(device_map)  # warn, error and fix the device map
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ma-user/anaconda3/envs/PyTorch-2.6.0/lib/python3.11/site-packages/transformers/integrations/accelerate.py", line 134, in check_and_set_device_map
    raise ValueError(
ValueError: Using a `device_map`, `tp_plan`, `torch.device` context manager or setting `torch.set_default_device(device)` requires `accelerate`. You can install it with `pip install accelerate`
[ERROR] 2026-03-21-16:54:20 (PID:154859, Device:0, RankID:-1) ERR99999 UNKNOWN applicaiton exception
