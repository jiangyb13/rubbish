import torch
import torch_npu  # 910B 必须导入 NPU 支持
from omegaconf import OmegaConf
from hydra.utils import instantiate
from sam2.sam2_image_predictor import SAM2ImagePredictor

# 1. 指定绝对路径 (请确保这两个路径是正确的)
yaml_path = "/data/huanan/code/jwx1416454/HUAWEI_CrossPairDataset/utils/sam2/sam2_configs/sam2_hiera_l.yaml"
ckpt_path = "/data/huanan/code/jwx1416454/HUAWEI_CrossPairDataset/pretrained_models/sam2-hiera-large/sam2_hiera_large.pt"

# 2. 绕过 Hydra 全局搜索，直接读取本地 YAML
cfg = OmegaConf.load(yaml_path)

# 3. 强行通过配置实例化真正的 PyTorch 模型
# 如果这里报错，说明你的 PYTHONPATH 没配置 utils/sam2
sam2_core_model = instantiate(cfg.model, _recursive_=True)

# 4. 手动加载权重并分配到 NPU
device = "npu"
state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
if "model" in state_dict:
    state_dict = state_dict["model"]

sam2_core_model.load_state_dict(state_dict)
sam2_core_model.to(device).eval()

# 5. 包装成 Predictor 供后续推理使用
seg_model = SAM2ImagePredictor(sam2_core_model)
