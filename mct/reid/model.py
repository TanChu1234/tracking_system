"""
TransReID model setup and transform utilities.
"""
import os
import torch
import torchvision.transforms as T

from config import cfg
from model import make_model


def setup_transreid(config_file, weights_path, device='cuda'):
    """
    Initialize TransReID model for person re-identification.
    
    Args:
        config_file: Path to TransReID YAML config
        weights_path: Path to trained weights
        device: 'cuda' or 'cpu'
    
    Returns:
        model: TransReID model in eval mode
    """
    if config_file != "":
        cfg.merge_from_file(config_file)
    
    cfg.MODEL.DEVICE_ID = "0"

    # Check for local weights
    local_weights = os.path.abspath("weights/jx_vit_base_p16_224-80ecf9dd.pth")
    if os.path.exists(local_weights):
        cfg.MODEL.PRETRAIN_PATH = local_weights
    else:
        cfg.MODEL.PRETRAIN_PATH = ""
        
    cfg.freeze()
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    
    model = make_model(cfg, num_class=751, camera_num=6, view_num=0)
    
    if weights_path and os.path.exists(weights_path):
        print(f"Loading TransReID weights from {weights_path}")
        model.load_param(weights_path)
    else:
        print(f"Warning: ReID Weights not found at {weights_path}")
        
    model.to(device)
    model.eval()
    return model


def get_transforms():
    """
    Get image transforms for TransReID inference.
    Uses cfg.INPUT settings.
    
    Returns:
        torchvision.transforms.Compose
    """
    transform = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])
    return transform
