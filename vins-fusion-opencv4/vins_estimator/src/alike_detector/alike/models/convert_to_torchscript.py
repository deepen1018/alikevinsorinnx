import os
import torch
from alike import ALike, configs

# 模型转换函数
def convert_to_torchscript(model_name, output_path, device='cuda'):
    """
    Convert the ALike model to TorchScript format for LibTorch compatibility.
    :param model_name: str, model configuration name from configs (e.g., 'alike-t').
    :param output_path: str, output path for the TorchScript file.
    :param device: str, 'cuda' or 'cpu'.
    """
    # 加载模型配置
    if model_name not in configs:
        raise ValueError(f"Model {model_name} is not in the configurations!")
    config = configs[model_name]

    # 创建模型实例
    model = ALike(**config, device=device)
    model.eval()  # 设置为评估模式

    # 打印模型信息
    print(f"Model {model_name} initialized.")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # 转换为TorchScript
    try:
        scripted_model = torch.jit.script(model)  # 使用TorchScript脚本化
        torch.jit.save(scripted_model, output_path)  # 保存为.pt文件
        print(f"Model successfully converted to TorchScript and saved at {output_path}")
    except Exception as e:
        print(f"Error during conversion: {e}")


if __name__ == "__main__":
    # 配置参数
    MODEL_NAME = "alike-t"  # 可选: alike-t, alike-s, alike-n, alike-l
    OUTPUT_PATH = "alike-t.pt"  # 保存路径
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 根据设备自动选择

    # 转换模型
    convert_to_torchscript(MODEL_NAME, OUTPUT_PATH, device=DEVICE)

