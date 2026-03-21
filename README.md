  File "/data/huanan/code/jwx1416454/HUAWEI_CrossPairDataset/main.py", line 101, in <module>
    mainForOneShot()
  File "/data/huanan/code/jwx1416454/HUAWEI_CrossPairDataset/main.py", line 61, in mainForOneShot
    pipeline = OneShotProcessPipeline(config)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/huanan/code/jwx1416454/HUAWEI_CrossPairDataset/utils/one_shot.py", line 652, in __init__
    self.person_detector = PersonDetector(config)
                           ^^^^^^^^^^^^^^^^^^^^^^
  File "/data/huanan/code/jwx1416454/HUAWEI_CrossPairDataset/utils/one_shot.py", line 357, in __init__
    self.deca_model = DECA(config=deca_cfg, device=self.device)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/huanan/code/jwx1416454/HUAWEI_CrossPairDataset/utils/DECA/decalib/deca.py", line 58, in __init__
    self._create_model(self.cfg.model)
  File "/data/huanan/code/jwx1416454/HUAWEI_CrossPairDataset/utils/DECA/decalib/deca.py", line 93, in _create_model
    self.flame = FLAME(model_cfg).to(self.device)
                 ^^^^^^^^^^^^^^^^
  File "/data/huanan/code/jwx1416454/HUAWEI_CrossPairDataset/utils/DECA/decalib/models/FLAME.py", line 47, in __init__
    ss = pickle.load(f, encoding='latin1')
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ma-user/anaconda3/envs/PyTorch-2.6.0/lib/python3.11/site-packages/chumpy/__init__.py", line 1, in <module>
    from .ch import *
  File "/home/ma-user/anaconda3/envs/PyTorch-2.6.0/lib/python3.11/site-packages/chumpy/ch.py", line 1319, in <module>
    from . import linalg
  File "/home/ma-user/anaconda3/envs/PyTorch-2.6.0/lib/python3.11/site-packages/chumpy/linalg.py", line 178, in <module>
    class SvdD(Ch):
  File "/home/ma-user/anaconda3/envs/PyTorch-2.6.0/lib/python3.11/site-packages/chumpy/linalg.py", line 181, in SvdD
    @depends_on('x')
     ^^^^^^^^^^^^^^^
  File "/home/ma-user/anaconda3/envs/PyTorch-2.6.0/lib/python3.11/site-packages/chumpy/ch.py", line 1203, in _depends_on
    want_out = 'out' in inspect.getargspec(func).args
                        ^^^^^^^^^^^^^^^^^^
AttributeError: module 'inspect' has no attribute 'getargspec'. Did you mean: 'getargs'?
[ERROR] 2026-03-21-16:22:49 (PID:76968, Device:0, RankID:-1) ERR99999 UNKNOWN applicaiton exception
