: Traceback (most recent call last):
[rank1]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/scripts/inference_mmdit_i2v_id.py", line 578, in <module>
[rank1]:     main()
[rank1]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/scripts/inference_mmdit_i2v_id.py", line 308, in main
[rank1]:     vae = build_backbone(cfg, cfg.vae.get("backbone", "vae_causal_16ch_dist"))
[rank1]:           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/mimogpt/models/build.py", line 10, in build_backbone
[rank1]:     backbone = BACKBONE_REGISTRY.get(backbone_name)(cfg)
[rank1]:                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/mimogpt/models/vae/distributed_vae.py", line 1186, in motionvae_16ch_dist
[rank1]:     return DistributedVAE(vae_config)
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/mimogpt/models/vae/distributed_vae.py", line 960, in __init__
[rank1]:     self.model = instantiate_from_config(config.model_config)
[rank1]:                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/mimogpt/models/modules/ldm/utils.py", line 14, in instantiate_from_config
[rank1]:     return get_obj_from_str(config["target"])(**config.get("params", dict()))
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/mimogpt/models/modules/ldm/utils.py", line 10, in get_obj_from_str
[rank1]:     return getattr(importlib.import_module(module, package=None), cls)
[rank1]:                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/home/ma-user/anaconda/lib/python3.11/importlib/__init__.py", line 126, in import_module
[rank1]:     return _bootstrap._gcd_import(name[level:], package, level)
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "<frozen importlib._bootstrap>", line 1204, in _gcd_import
[rank1]:   File "<frozen importlib._bootstrap>", line 1176, in _find_and_load
[rank1]:   File "<frozen importlib._bootstrap>", line 1147, in _find_and_load_unlocked
[rank1]:   File "<frozen importlib._bootstrap>", line 690, in _load_unlocked
[rank1]:   File "<frozen importlib._bootstrap_external>", line 940, in exec_module
[rank1]:   File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
[rank1]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/mimogpt/models/modules/ldm/encoder/autoencoder3d_t8_flow_decom_sim_dist.py", line 12, in <module>
[rank1]:     from mimogpt.models.modules.ldm.encoder.spynet import SPyNet
[rank1]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/mimogpt/models/modules/ldm/encoder/spynet.py", line 4, in <module>
[rank1]:     from mmcv.cnn import ConvModule
[rank1]:   File "/home/ma-user/anaconda/lib/python3.11/site-packages/mmcv/cnn/__init__.py", line 14, in <module>
[rank1]:     from .builder import MODELS, build_model_from_cfg
[rank1]:   File "/home/ma-user/anaconda/lib/python3.11/site-packages/mmcv/cnn/builder.py", line 2, in <module>
[rank1]:     from ..runner import Sequential
[rank1]:   File "/home/ma-user/anaconda/lib/python3.11/site-packages/mmcv/runner/__init__.py", line 45, in <module>
[rank1]:     from mmcv.device import ipu  # isort:skip  # noqa
[rank1]:     ^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/home/ma-user/anaconda/lib/python3.11/site-packages/mmcv/device/__init__.py", line 2, in <module>
[rank1]:     from . import ipu, mlu, mps, npu
[rank1]:   File "/home/ma-user/anaconda/lib/python3.11/site-packages/mmcv/device/npu/__init__.py", line 3, in <module>
[rank1]:     from .data_parallel import NPUDataParallel
[rank1]:   File "/home/ma-user/anaconda/lib/python3.11/site-packages/mmcv/device/npu/data_parallel.py", line 20, in <module>
[rank1]:     for m in sys.modules:
[rank1]: RuntimeError: dictionary changed size during iteration
[rank5]: Traceback (most recent call last):
[rank5]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/scripts/inference_mmdit_i2v_id.py", line 578, in <module>
[rank5]:     main()
[rank5]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/scripts/inference_mmdit_i2v_id.py", line 308, in main
[rank5]:     vae = build_backbone(cfg, cfg.vae.get("backbone", "vae_causal_16ch_dist"))
[rank5]:           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank5]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/mimogpt/models/build.py", line 10, in build_backbone
[rank5]:     backbone = BACKBONE_REGISTRY.get(backbone_name)(cfg)
[rank5]:                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank5]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/mimogpt/models/vae/distributed_vae.py", line 1186, in motionvae_16ch_dist
[rank5]:     return DistributedVAE(vae_config)
[rank5]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank5]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/mimogpt/models/vae/distributed_vae.py", line 960, in __init__
[rank5]:     self.model = instantiate_from_config(config.model_config)
[rank5]:                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank5]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/mimogpt/models/modules/ldm/utils.py", line 14, in instantiate_from_config
[rank5]:     return get_obj_from_str(config["target"])(**config.get("params", dict()))
[rank5]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank5]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/mimogpt/models/modules/ldm/utils.py", line 10, in get_obj_from_str
[rank5]:     return getattr(importlib.import_module(module, package=None), cls)
[rank5]:                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank5]:   File "/home/ma-user/anaconda/lib/python3.11/importlib/__init__.py", line 126, in import_module
[rank5]:     return _bootstrap._gcd_import(name[level:], package, level)
[rank5]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank5]:   File "<frozen importlib._bootstrap>", line 1204, in _gcd_import
[rank5]:   File "<frozen importlib._bootstrap>", line 1176, in _find_and_load
[rank5]:   File "<frozen importlib._bootstrap>", line 1147, in _find_and_load_unlocked
[rank5]:   File "<frozen importlib._bootstrap>", line 690, in _load_unlocked
[rank5]:   File "<frozen importlib._bootstrap_external>", line 940, in exec_module
[rank5]:   File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
[rank5]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/mimogpt/models/modules/ldm/encoder/autoencoder3d_t8_flow_decom_sim_dist.py", line 12, in <module>
[rank5]:     from mimogpt.models.modules.ldm.encoder.spynet import SPyNet
[rank5]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/mimogpt/models/modules/ldm/encoder/spynet.py", line 4, in <module>
[rank5]:     from mmcv.cnn import ConvModule
[rank5]:   File "/home/ma-user/anaconda/lib/python3.11/site-packages/mmcv/cnn/__init__.py", line 14, in <module>
[rank5]:     from .builder import MODELS, build_model_from_cfg
[rank5]:   File "/home/ma-user/anaconda/lib/python3.11/site-packages/mmcv/cnn/builder.py", line 2, in <module>
[rank5]:     from ..runner import Sequential
[rank5]:   File "/home/ma-user/anaconda/lib/python3.11/site-packages/mmcv/runner/__init__.py", line 45, in <module>
[rank5]:     from mmcv.device import ipu  # isort:skip  # noqa
[rank5]:     ^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank5]:   File "/home/ma-user/anaconda/lib/python3.11/site-packages/mmcv/device/__init__.py", line 2, in <module>
[rank5]:     from . import ipu, mlu, mps, npu
[rank5]:   File "/home/ma-user/anaconda/lib/python3.11/site-packages/mmcv/device/npu/__init__.py", line 3, in <module>
[rank5]:     from .data_parallel import NPUDataParallel
[rank5]:   File "/home/ma-user/anaconda/lib/python3.11/site-packages/mmcv/device/npu/data_parallel.py", line 20, in <module>
[rank5]:     for m in sys.modules:
[rank5]: RuntimeError: dictionary changed size during iteration
[rank4]: Traceback (most recent call last):
[rank4]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/scripts/inference_mmdit_i2v_id.py", line 578, in <module>
[rank4]:     main()
[rank4]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/scripts/inference_mmdit_i2v_id.py", line 308, in main
[rank4]:     vae = build_backbone(cfg, cfg.vae.get("backbone", "vae_causal_16ch_dist"))
[rank4]:           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank4]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/mimogpt/models/build.py", line 10, in build_backbone
[rank4]:     backbone = BACKBONE_REGISTRY.get(backbone_name)(cfg)
[rank4]:                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank4]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/mimogpt/models/vae/distributed_vae.py", line 1186, in motionvae_16ch_dist
[rank4]:     return DistributedVAE(vae_config)
[rank4]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank4]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/mimogpt/models/vae/distributed_vae.py", line 960, in __init__
[rank4]:     self.model = instantiate_from_config(config.model_config)
[rank4]:                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank4]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/mimogpt/models/modules/ldm/utils.py", line 14, in instantiate_from_config
[rank4]:     return get_obj_from_str(config["target"])(**config.get("params", dict()))
[rank4]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank4]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/mimogpt/models/modules/ldm/utils.py", line 10, in get_obj_from_str
[rank4]:     return getattr(importlib.import_module(module, package=None), cls)
[rank4]:                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank4]:   File "/home/ma-user/anaconda/lib/python3.11/importlib/__init__.py", line 126, in import_module
[rank4]:     return _bootstrap._gcd_import(name[level:], package, level)
[rank4]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank4]:   File "<frozen importlib._bootstrap>", line 1204, in _gcd_import
[rank4]:   File "<frozen importlib._bootstrap>", line 1176, in _find_and_load
[rank4]:   File "<frozen importlib._bootstrap>", line 1147, in _find_and_load_unlocked
[rank4]:   File "<frozen importlib._bootstrap>", line 690, in _load_unlocked
[rank4]:   File "<frozen importlib._bootstrap_external>", line 940, in exec_module
[rank4]:   File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
[rank4]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/mimogpt/models/modules/ldm/encoder/autoencoder3d_t8_flow_decom_sim_dist.py", line 12, in <module>
[rank4]:     from mimogpt.models.modules.ldm.encoder.spynet import SPyNet
[rank4]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/mimogpt/models/modules/ldm/encoder/spynet.py", line 4, in <module>
[rank4]:     from mmcv.cnn import ConvModule
[rank4]:   File "/home/ma-user/anaconda/lib/python3.11/site-packages/mmcv/cnn/__init__.py", line 14, in <module>
[rank4]:     from .builder import MODELS, build_model_from_cfg
[rank4]:   File "/home/ma-user/anaconda/lib/python3.11/site-packages/mmcv/cnn/builder.py", line 2, in <module>
[rank4]:     from ..runner import Sequential
[rank4]:   File "/home/ma-user/anaconda/lib/python3.11/site-packages/mmcv/runner/__init__.py", line 45, in <module>
[rank4]:     from mmcv.device import ipu  # isort:skip  # noqa
[rank4]:     ^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank4]:   File "/home/ma-user/anaconda/lib/python3.11/site-packages/mmcv/device/__init__.py", line 2, in <module>
[rank4]:     from . import ipu, mlu, mps, npu
[rank4]:   File "/home/ma-user/anaconda/lib/python3.11/site-packages/mmcv/device/npu/__init__.py", line 3, in <module>
[rank4]:     from .data_parallel import NPUDataParallel
[rank4]:   File "/home/ma-user/anaconda/lib/python3.11/site-packages/mmcv/device/npu/data_parallel.py", line 20, in <module>
[rank4]:     for m in sys.modules:
[rank4]: RuntimeError: dictionary changed size during iteration
/home/ma-user/anaconda/lib/python3.11/site-packages/mmcv/device/npu/data_parallel.py:22: UserWarning: Torchaudio's I/O functions now support par-call bakcend dispatch. Importing backend implementation directly is no longer guaranteed to work. Please use `backend` keyword with load/save/info function, instead of calling the udnerlying implementation directly.
  if hasattr(sys.modules[m], '_check_balance'):
[rank3]: Traceback (most recent call last):
[rank3]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/scripts/inference_mmdit_i2v_id.py", line 578, in <module>
[rank3]:     main()
[rank3]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/scripts/inference_mmdit_i2v_id.py", line 308, in main
[rank3]:     vae = build_backbone(cfg, cfg.vae.get("backbone", "vae_causal_16ch_dist"))
[rank3]:           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank3]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/mimogpt/models/build.py", line 10, in build_backbone
[rank3]:     backbone = BACKBONE_REGISTRY.get(backbone_name)(cfg)
[rank3]:                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank3]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/mimogpt/models/vae/distributed_vae.py", line 1186, in motionvae_16ch_dist
[rank3]:     return DistributedVAE(vae_config)
[rank3]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank3]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/mimogpt/models/vae/distributed_vae.py", line 960, in __init__
[rank3]:     self.model = instantiate_from_config(config.model_config)
[rank3]:                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank3]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/mimogpt/models/modules/ldm/utils.py", line 14, in instantiate_from_config
[rank3]:     return get_obj_from_str(config["target"])(**config.get("params", dict()))
[rank3]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank3]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/mimogpt/models/modules/ldm/utils.py", line 10, in get_obj_from_str
[rank3]:     return getattr(importlib.import_module(module, package=None), cls)
[rank3]:                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank3]:   File "/home/ma-user/anaconda/lib/python3.11/importlib/__init__.py", line 126, in import_module
[rank3]:     return _bootstrap._gcd_import(name[level:], package, level)
[rank3]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank3]:   File "<frozen importlib._bootstrap>", line 1204, in _gcd_import
[rank3]:   File "<frozen importlib._bootstrap>", line 1176, in _find_and_load
[rank3]:   File "<frozen importlib._bootstrap>", line 1147, in _find_and_load_unlocked
[rank3]:   File "<frozen importlib._bootstrap>", line 690, in _load_unlocked
[rank3]:   File "<frozen importlib._bootstrap_external>", line 940, in exec_module
[rank3]:   File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
[rank3]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/mimogpt/models/modules/ldm/encoder/autoencoder3d_t8_flow_decom_sim_dist.py", line 12, in <module>
[rank3]:     from mimogpt.models.modules.ldm.encoder.spynet import SPyNet
[rank3]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/mimogpt/models/modules/ldm/encoder/spynet.py", line 4, in <module>
[rank3]:     from mmcv.cnn import ConvModule
[rank3]:   File "/home/ma-user/anaconda/lib/python3.11/site-packages/mmcv/cnn/__init__.py", line 14, in <module>
[rank3]:     from .builder import MODELS, build_model_from_cfg
[rank3]:   File "/home/ma-user/anaconda/lib/python3.11/site-packages/mmcv/cnn/builder.py", line 2, in <module>
[rank3]:     from ..runner import Sequential
[rank3]:   File "/home/ma-user/anaconda/lib/python3.11/site-packages/mmcv/runner/__init__.py", line 45, in <module>
[rank3]:     from mmcv.device import ipu  # isort:skip  # noqa
[rank3]:     ^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank3]:   File "/home/ma-user/anaconda/lib/python3.11/site-packages/mmcv/device/__init__.py", line 2, in <module>
[rank3]:     from . import ipu, mlu, mps, npu
[rank3]:   File "/home/ma-user/anaconda/lib/python3.11/site-packages/mmcv/device/npu/__init__.py", line 3, in <module>
[rank3]:     from .data_parallel import NPUDataParallel
[rank3]:   File "/home/ma-user/anaconda/lib/python3.11/site-packages/mmcv/device/npu/data_parallel.py", line 20, in <module>
[rank3]:     for m in sys.modules:
[rank3]: RuntimeError: dictionary changed size during iteration
[rank2]: Traceback (most recent call last):
[rank2]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/scripts/inference_mmdit_i2v_id.py", line 578, in <module>
[rank2]:     main()
[rank2]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/scripts/inference_mmdit_i2v_id.py", line 308, in main
[rank2]:     vae = build_backbone(cfg, cfg.vae.get("backbone", "vae_causal_16ch_dist"))
[rank2]:           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank2]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/mimogpt/models/build.py", line 10, in build_backbone
[rank2]:     backbone = BACKBONE_REGISTRY.get(backbone_name)(cfg)
[rank2]:                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank2]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/mimogpt/models/vae/distributed_vae.py", line 1186, in motionvae_16ch_dist
[rank2]:     return DistributedVAE(vae_config)
[rank2]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank2]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/mimogpt/models/vae/distributed_vae.py", line 960, in __init__
[rank2]:     self.model = instantiate_from_config(config.model_config)
[rank2]:                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank2]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/mimogpt/models/modules/ldm/utils.py", line 14, in instantiate_from_config
[rank2]:     return get_obj_from_str(config["target"])(**config.get("params", dict()))
[rank2]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank2]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/mimogpt/models/modules/ldm/utils.py", line 10, in get_obj_from_str
[rank2]:     return getattr(importlib.import_module(module, package=None), cls)
[rank2]:                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank2]:   File "/home/ma-user/anaconda/lib/python3.11/importlib/__init__.py", line 126, in import_module
[rank2]:     return _bootstrap._gcd_import(name[level:], package, level)
[rank2]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank2]:   File "<frozen importlib._bootstrap>", line 1204, in _gcd_import
[rank2]:   File "<frozen importlib._bootstrap>", line 1176, in _find_and_load
[rank2]:   File "<frozen importlib._bootstrap>", line 1147, in _find_and_load_unlocked
[rank2]:   File "<frozen importlib._bootstrap>", line 690, in _load_unlocked
[rank2]:   File "<frozen importlib._bootstrap_external>", line 940, in exec_module
[rank2]:   File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
[rank2]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/mimogpt/models/modules/ldm/encoder/autoencoder3d_t8_flow_decom_sim_dist.py", line 12, in <module>
[rank2]:     from mimogpt.models.modules.ldm.encoder.spynet import SPyNet
[rank2]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/mimogpt/models/modules/ldm/encoder/spynet.py", line 4, in <module>
[rank2]:     from mmcv.cnn import ConvModule
[rank2]:   File "/home/ma-user/anaconda/lib/python3.11/site-packages/mmcv/cnn/__init__.py", line 14, in <module>
[rank2]:     from .builder import MODELS, build_model_from_cfg
[rank2]:   File "/home/ma-user/anaconda/lib/python3.11/site-packages/mmcv/cnn/builder.py", line 2, in <module>
[rank2]:     from ..runner import Sequential
[rank2]:   File "/home/ma-user/anaconda/lib/python3.11/site-packages/mmcv/runner/__init__.py", line 45, in <module>
[rank2]:     from mmcv.device import ipu  # isort:skip  # noqa
[rank2]:     ^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank2]:   File "/home/ma-user/anaconda/lib/python3.11/site-packages/mmcv/device/__init__.py", line 2, in <module>
[rank2]:     from . import ipu, mlu, mps, npu
[rank2]:   File "/home/ma-user/anaconda/lib/python3.11/site-packages/mmcv/device/npu/__init__.py", line 3, in <module>
[rank2]:     from .data_parallel import NPUDataParallel
[rank2]:   File "/home/ma-user/anaconda/lib/python3.11/site-packages/mmcv/device/npu/data_parallel.py", line 20, in <module>
[rank2]:     for m in sys.modules:
[rank2]: RuntimeError: dictionary changed size during iteration
[rank7]: Traceback (most recent call last):
[rank7]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/scripts/inference_mmdit_i2v_id.py", line 578, in <module>
[rank7]:     main()
[rank7]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/scripts/inference_mmdit_i2v_id.py", line 308, in main
[rank7]:     vae = build_backbone(cfg, cfg.vae.get("backbone", "vae_causal_16ch_dist"))
[rank7]:           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank7]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/mimogpt/models/build.py", line 10, in build_backbone
[rank7]:     backbone = BACKBONE_REGISTRY.get(backbone_name)(cfg)
[rank7]:                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank7]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/mimogpt/models/vae/distributed_vae.py", line 1186, in motionvae_16ch_dist
[rank7]:     return DistributedVAE(vae_config)
[rank7]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank7]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/mimogpt/models/vae/distributed_vae.py", line 960, in __init__
[rank7]:     self.model = instantiate_from_config(config.model_config)
[rank7]:                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank7]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/mimogpt/models/modules/ldm/utils.py", line 14, in instantiate_from_config
[rank7]:     return get_obj_from_str(config["target"])(**config.get("params", dict()))
[rank7]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank7]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/mimogpt/models/modules/ldm/utils.py", line 10, in get_obj_from_str
[rank7]:     return getattr(importlib.import_module(module, package=None), cls)
[rank7]:                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank7]:   File "/home/ma-user/anaconda/lib/python3.11/importlib/__init__.py", line 126, in import_module
[rank7]:     return _bootstrap._gcd_import(name[level:], package, level)
[rank7]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank7]:   File "<frozen importlib._bootstrap>", line 1204, in _gcd_import
[rank7]:   File "<frozen importlib._bootstrap>", line 1176, in _find_and_load
[rank7]:   File "<frozen importlib._bootstrap>", line 1147, in _find_and_load_unlocked
[rank7]:   File "<frozen importlib._bootstrap>", line 690, in _load_unlocked
[rank7]:   File "<frozen importlib._bootstrap_external>", line 940, in exec_module
[rank7]:   File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
[rank7]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/mimogpt/models/modules/ldm/encoder/autoencoder3d_t8_flow_decom_sim_dist.py", line 12, in <module>
[rank7]:     from mimogpt.models.modules.ldm.encoder.spynet import SPyNet
[rank7]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/mimogpt/models/modules/ldm/encoder/spynet.py", line 4, in <module>
[rank7]:     from mmcv.cnn import ConvModule
[rank7]:   File "/home/ma-user/anaconda/lib/python3.11/site-packages/mmcv/cnn/__init__.py", line 14, in <module>
[rank7]:     from .builder import MODELS, build_model_from_cfg
[rank7]:   File "/home/ma-user/anaconda/lib/python3.11/site-packages/mmcv/cnn/builder.py", line 2, in <module>
[rank7]:     from ..runner import Sequential
[rank7]:   File "/home/ma-user/anaconda/lib/python3.11/site-packages/mmcv/runner/__init__.py", line 45, in <module>
[rank7]:     from mmcv.device import ipu  # isort:skip  # noqa
[rank7]:     ^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank7]:   File "/home/ma-user/anaconda/lib/python3.11/site-packages/mmcv/device/__init__.py", line 2, in <module>
[rank7]:     from . import ipu, mlu, mps, npu
[rank7]:   File "/home/ma-user/anaconda/lib/python3.11/site-packages/mmcv/device/npu/__init__.py", line 3, in <module>
[rank7]:     from .data_parallel import NPUDataParallel
[rank7]:   File "/home/ma-user/anaconda/lib/python3.11/site-packages/mmcv/device/npu/data_parallel.py", line 20, in <module>
[rank7]:     for m in sys.modules:
[rank7]: RuntimeError: dictionary changed size during iteration
[rank0]: Traceback (most recent call last):
[rank0]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/scripts/inference_mmdit_i2v_id.py", line 578, in <module>
[rank0]:     main()
[rank0]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/scripts/inference_mmdit_i2v_id.py", line 308, in main
[rank0]:     vae = build_backbone(cfg, cfg.vae.get("backbone", "vae_causal_16ch_dist"))
[rank0]:           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/mimogpt/models/build.py", line 10, in build_backbone
[rank0]:     backbone = BACKBONE_REGISTRY.get(backbone_name)(cfg)
[rank0]:                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/mimogpt/models/vae/distributed_vae.py", line 1186, in motionvae_16ch_dist
[rank0]:     return DistributedVAE(vae_config)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/mimogpt/models/vae/distributed_vae.py", line 960, in __init__
[rank0]:     self.model = instantiate_from_config(config.model_config)
[rank0]:                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/mimogpt/models/modules/ldm/utils.py", line 14, in instantiate_from_config
[rank0]:     return get_obj_from_str(config["target"])(**config.get("params", dict()))
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/mimogpt/models/modules/ldm/utils.py", line 10, in get_obj_from_str
[rank0]:     return getattr(importlib.import_module(module, package=None), cls)
[rank0]:                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/ma-user/anaconda/lib/python3.11/importlib/__init__.py", line 126, in import_module
[rank0]:     return _bootstrap._gcd_import(name[level:], package, level)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "<frozen importlib._bootstrap>", line 1204, in _gcd_import
[rank0]:   File "<frozen importlib._bootstrap>", line 1176, in _find_and_load
[rank0]:   File "<frozen importlib._bootstrap>", line 1147, in _find_and_load_unlocked
[rank0]:   File "<frozen importlib._bootstrap>", line 690, in _load_unlocked
[rank0]:   File "<frozen importlib._bootstrap_external>", line 940, in exec_module
[rank0]:   File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
[rank0]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/mimogpt/models/modules/ldm/encoder/autoencoder3d_t8_flow_decom_sim_dist.py", line 12, in <module>
[rank0]:     from mimogpt.models.modules.ldm.encoder.spynet import SPyNet
[rank0]:   File "/data/guiyang/code/jwx1416454/20260514-ID-cross-v20-new/mimogpt/models/modules/ldm/encoder/spynet.py", line 4, in <module>
[rank0]:     from mmcv.cnn import ConvModule
[rank0]:   File "/home/ma-user/anaconda/lib/python3.11/site-packages/mmcv/cnn/__init__.py", line 14, in <module>
[rank0]:     from .builder import MODELS, build_model_from_cfg
[rank0]:   File "/home/ma-user/anaconda/lib/python3.11/site-packages/mmcv/cnn/builder.py", line 2, in <module>
[rank0]:     from ..runner import Sequential
[rank0]:   File "/home/ma-user/anaconda/lib/python3.11/site-packages/mmcv/runner/__init__.py", line 45, in <module>
[rank0]:     from mmcv.device import ipu  # isort:skip  # noqa
[rank0]:     ^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/ma-user/anaconda/lib/python3.11/site-packages/mmcv/device/__init__.py", line 2, in <module>
[rank0]:     from . import ipu, mlu, mps, npu
[rank0]:   File "/home/ma-user/anaconda/lib/python3.11/site-packages/mmcv/device/npu/__init__.py", line 3, in <module>
[rank0]:     from .data_parallel import NPUDataParallel
[rank0]:   File "/home/ma-user/anaconda/lib/python3.11/site-packages/mmcv/device/npu/data_parallel.py", line 20, in <module>
[rank0]:     for m in sys.modules:
[rank0]: RuntimeError: dictionary changed size during iteration
[ERROR] 2026-09-06-16:14:16 (PID:3938695, Device:1, RankID:-1) ERR99999 UNKNOWN applicaiton exception
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
[ERROR] 2026-09-06-16:14:17 (PID:3938699, Device:5, RankID:-1) ERR99999 UNKNOWN applicaiton exception
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
[ERROR] 2026-09-06-16:14:17 (PID:3938701, Device:7, RankID:-1) ERR99999 UNKNOWN applicaiton exception
[ERROR] 2026-09-06-16:14:18 (PID:3938700, Device:6, RankID:-1) ERR99999 UNKNOWN applicaiton exception
[ERROR] 2026-09-06-16:14:18 (PID:3938696, Device:2, RankID:-1) ERR99999 UNKNOWN applicaiton exception
[ERROR] 2026-09-06-16:14:19 (PID:3938694, Device:0, RankID:-1) ERR99999 UNKNOWN applicaiton exception
[ERROR] 2026-09-06-16:14:20 (PID:3938698, Device:4, RankID:-1) ERR99999 UNKNOWN applicaiton exception
W0906 16:14:21.241000 3938419 site-packages/torch/distributed/elastic/multiprocessing/api.py:897] Sending process 3938694 closing signal SIGTERM
W0906 16:14:21.242000 3938419 site-packages/torch/distributed/elastic/multiprocessing/api.py:897] Sending process 3938696 closing signal SIGTERM
W0906 16:14:21.242000 3938419 site-packages/torch/distributed/elastic/multiprocessing/api.py:897] Sending process 3938697 closing signal SIGTERM
W0906 16:14:21.243000 3938419 site-packages/torch/distributed/elastic/multiprocessing/api.py:897] Sending process 3938698 closing signal SIGTERM
W0906 16:14:21.243000 3938419 site-packages/torch/distributed/elastic/multiprocessing/api.py:897] Sending process 3938700 closing signal SIGTERM
W0906 16:14:21.243000 3938419 site-packages/torch/distributed/elastic/multiprocessing/api.py:897] Sending process 3938701 closing signal SIGTERM
/home/ma-user/anaconda/lib/python3.11/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 30 leaked semaphore objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d '
E0906 16:14:27.182000 3938419 site-packages/torch/distributed/elastic/multiprocessing/api.py:869] failed (exitcode: 1) local_rank: 1 (pid: 3938695) of binary: /home/ma-user/anaconda/bin/python
Traceback (most recent call last):
  File "/home/ma-user/anaconda/bin/torchrun", line 8, in <module>
    sys.exit(main())
             ^^^^^^
  File "/home/ma-user/anaconda/lib/python3.11/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 355, in wrapper
    return f(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^
  File "/home/ma-user/anaconda/lib/python3.11/site-packages/torch/distributed/run.py", line 918, in main
    run(args)
  File "/home/ma-user/anaconda/lib/python3.11/site-packages/torch/distributed/run.py", line 909, in run
    elastic_launch(
  File "/home/ma-user/anaconda/lib/python3.11/site-packages/torch/distributed/launcher/api.py", line 138, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ma-user/anaconda/lib/python3.11/site-packages/torch/distributed/launcher/api.py", line 269, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
============================================================
scripts/inference_mmdit_i2v_id.py FAILED
------------------------------------------------------------
Failures:
[1]:
  time      : 2026-09-06_16:14:21
  host      : notebook-7fa88179-4a92-4d27-bef7-2c4d6f2a8b84.notebook-7fa88179-4a92-4d27-bef7-2c4d6f2a8b84-distributed.default.svc.cluster.local
  rank      : 5 (local_rank: 5)
  exitcode  : 1 (pid: 3938699)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2026-09-06_16:14:21
  host      : notebook-7fa88179-4a92-4d27-bef7-2c4d6f2a8b84.notebook-7fa88179-4a92-4d27-bef7-2c4d6f2a8b84-distributed.default.svc.cluster.local
  rank      : 1 (local_rank: 1)
  exitcode  : 1 (pid: 3938695)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================
[ERROR] 2026-09-06-16:14:27 (PID:3938419, Device:-1, RankID:-1) ERR99999 UNKNOWN applicaiton exception

2026-09-06 16:14:29:INFO:copy from s3://bucket-9861-guiyang/code/jwx1416454/I2V-ID/train_runs/cross_v55_formal_20260901/output/mmdit4.5B_ip_cloud_i2v_720p_lognorm_ts8_cross_v55.yml/ckpt/iter_4249.pth to /cache/bucket-9861-guiyang/code/jwx1416454/I2V-ID/train_runs/cross_v55_formal_20260901/output/mmdit4.5B_ip_cloud_i2v_720p_lognorm_ts8_cross_v55.yml/ckpt/iter_4249.pth

2026-09-06 16:14:29:INFO:rm -rf /cache/bucket-9861-guiyang/code/jwx1416454/I2V-ID/train_runs/cross_v55_formal_20260901/output/mmdit4.5B_ip_cloud_i2v_720p_lognorm_ts8_cross_v55.yml/ckpt/iter_4249.pth
[ma-user 20260514-ID-cross-v20-new]$
