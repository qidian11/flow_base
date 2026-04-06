import torch
import numpy as np
from normalizing_flow import RealNVP, compute_action, CONFIG, device


def load_trained_model(config):
    """加载保存的模型权重"""
    model = RealNVP(config=config).to(device)
    # map_location 确保在没有 GPU 的机器上也能加载
    state_dict = torch.load(config['model_save_path'], map_location=device)

    # 如果保存的是整个 dict (包含 optimizer)，则取 model_state_dict
    if 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)

    model.eval()  # 必须切换到评估模式
    return model


def test_model(batch_size):
    for _ in range(len(CONFIG)):
        config = CONFIG[_]
        model = load_trained_model(config)
        z = torch.randn(batch_size, config.L, config.L, device=device)
        phi, log_q = model(z)


