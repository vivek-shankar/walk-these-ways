import torch


def load_jit_model(jit_model_path):
    return torch.jit.load(jit_model_path).to('cuda:0')


def convert_jit_to_onnx(jit_model_path, onnx_model_path, dummy_input, input_names, output_names, opset_version=13):
    model = load_jit_model(jit_model_path)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        input_names=input_names,
        output_names=output_names,
        verbose=True,
        opset_version=opset_version
    )


# 设置模型文件路径

label = "gait-conditioned-agility/pretain-a1/train/081719.645276"
dirs = glob.glob(f"../../runs/{label}/*")
logdir = sorted(dirs)[0]
print(logdir)

body_jit_path = logdir + '/checkpoints/body_latest.jit'
adaptation_module_jit_path = logdir + '/checkpoints/adaptation_module_latest.jit'
body_onnx_path = logdir + '/onnx_models/body_model.onnx'
adaptation_module_onnx_path = logdir + '/onnx_models/adaptation_module_model.onnx'

# 创建虚拟输入
dummy_input = torch.randn(1, 2102).to('cuda:0')  # 输入维度为 [1, 2102]

# 转换 adaptation_module 模型
convert_jit_to_onnx(
    adaptation_module_jit_path,
    adaptation_module_onnx_path,
    dummy_input=dummy_input,
    input_names=['input'],
    output_names=['output'],
    opset_version=13
)

# 转换 body 模型
convert_jit_to_onnx(
    body_jit_path,
    body_onnx_path,
    dummy_input=dummy_input,
    input_names=['input'],
    output_names=['output'],
    opset_version=13
)
