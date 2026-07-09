"""
模型量化脚本：将 slider.onnx 从 FP32 量化为 INT8
"""

import os
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

# 路径
input_model = "/workspace/captcha_recognizer/models/slider.onnx"
output_model = "/workspace/captcha_recognizer/models/slider_int8.onnx"

# 使用动态量化（不需要校准数据，更简单且通用）
print("=" * 50)
print("开始动态量化（FP32 -> INT8）")
print(f"输入模型: {input_model}")
print(f"输出模型: {output_model}")
print("=" * 50)

quantize_dynamic(
    input_model,
    output_model,
    weight_type=QuantType.QInt8,
)

# 检查结果
original_size = os.path.getsize(input_model) / (1024 * 1024)
quantized_size = os.path.getsize(output_model) / (1024 * 1024)

print(f"\n✅ 量化完成！")
print(f"原模型大小: {original_size:.1f} MB")
print(f"量化模型大小: {quantized_size:.1f} MB")
print(f"压缩率: {quantized_size / original_size * 100:.1f}%")

# 验证量化模型能否正常加载
print("\n验证量化模型...")
sess = ort.InferenceSession(output_model, providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
input_shape = sess.get_inputs()[0].shape
print(f"输入名称: {input_name}")
print(f"输入形状: {input_shape}")
print(f"✅ 量化模型验证通过！")
