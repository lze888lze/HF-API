"""
YOLOv8-seg 动态量化脚本
========================

动态量化：只量化权重为INT8，激活保持FP32
适合YOLOv8-seg这种对精度敏感的模型
"""

import os
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

INPUT_MODEL = "/workspace/captcha_recognizer/models/slider.onnx"
OUTPUT_MODEL = "/workspace/captcha_recognizer/models/slider_int8.onnx"

print("=" * 60)
print("YOLOv8-seg 动态量化")
print("=" * 60)
print(f"输入模型: {INPUT_MODEL}")
print(f"输出模型: {OUTPUT_MODEL}")
print("=" * 60)

# 执行动态量化
quantize_dynamic(
    INPUT_MODEL,
    OUTPUT_MODEL,
    weight_type=QuantType.QInt8,
)

# 输出统计信息
original_size = os.path.getsize(INPUT_MODEL) / (1024 * 1024)
quantized_size = os.path.getsize(OUTPUT_MODEL) / (1024 * 1024)

print("\n" + "=" * 60)
print("✅ 动态量化完成！")
print("=" * 60)
print(f"原模型大小（FP32）: {original_size:.1f} MB")
print(f"量化模型大小（INT8）: {quantized_size:.1f} MB")
print(f"压缩率: {quantized_size / original_size * 100:.1f}%")

# 验证量化模型
print("\n验证量化模型...")
sess = ort.InferenceSession(OUTPUT_MODEL, providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
input_shape = sess.get_inputs()[0].shape
print(f"输入名称: {input_name}")
print(f"输入形状: {input_shape}")

test_input = np.zeros((1, 3, 640, 640), dtype=np.float32)
outputs = sess.run(None, {input_name: test_input})
print(f"输出数量: {len(outputs)}")
print(f"输出形状: {[o.shape for o in outputs]}")

print("\n✅ 动态量化模型验证通过！")