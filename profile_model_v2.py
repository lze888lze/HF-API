"""
性能剖析脚本 v2：分析 FP32 模型推理各阶段耗时
"""
import sys
import time
import cv2
import numpy as np

sys.path.insert(0, '/workspace')
from captcha_recognizer.slider import Slider


def profile_detailed(slider, img, runs=10):
    """详细剖析各阶段耗时"""
    print("=" * 70)
    print("FP32 模型性能剖析")
    print("=" * 70)
    print(f"图片尺寸: {img.shape[1]}x{img.shape[0]}")
    print(f"ONNX Runtime: {slider.session._model_path}")
    print(f"测试轮数: {runs}")
    print()
    
    # 1. 完整流程
    total_times = []
    for _ in range(runs):
        t0 = time.time()
        result = slider.identify_both(source=img)
        total_times.append((time.time() - t0) * 1000)
    
    print(f"总耗时 (identify_both):")
    print(f"  平均: {np.mean(total_times):.1f}ms")
    print(f"  最小: {np.min(total_times):.1f}ms")
    print(f"  最大: {np.max(total_times):.1f}ms")
    print()
    
    # 2. 分阶段剖析
    preprocess_times = []
    inference_times = []
    postprocess_times = []
    
    for _ in range(runs):
        # 预处理
        t0 = time.time()
        prep_img = slider.preprocess(img, (640, 640))
        preprocess_times.append((time.time() - t0) * 1000)
        
        # 模型推理
        t0 = time.time()
        outs = slider.session.run(None, {slider.session.get_inputs()[0].name: prep_img})
        inference_times.append((time.time() - t0) * 1000)
        
        # 后处理
        t0 = time.time()
        slider.postprocess(img, prep_img, outs, conf=0.5, iou=0.8)
        postprocess_times.append((time.time() - t0) * 1000)
    
    print("各阶段耗时分解:")
    print(f"  1. 预处理 (letterbox+归一化): {np.mean(preprocess_times):.1f}ms "
          f"({np.mean(preprocess_times)/np.mean(total_times)*100:.1f}%)")
    print(f"  2. ONNX 模型推理:           {np.mean(inference_times):.1f}ms "
          f"({np.mean(inference_times)/np.mean(total_times)*100:.1f}%)")
    print(f"  3. 后处理 (NMS+掩膜+解析):   {np.mean(postprocess_times):.1f}ms "
          f"({np.mean(postprocess_times)/np.mean(total_times)*100:.1f}%)")
    print()
    
    # 3. 后处理细分
    preds, protos = outs
    
    # 3.1 NMS
    nms_times = []
    for _ in range(runs):
        t0 = time.time()
        slider.non_max_suppression(preds, conf_thres=0.5, iou_thres=0.8)
        nms_times.append((time.time() - t0) * 1000)
    
    # 3.2 掩膜处理
    mask_times = []
    for _ in range(runs):
        t0 = time.time()
        # 模拟完整的后处理流程
        pred_list = slider.non_max_suppression(preds, conf_thres=0.5, iou_thres=0.8)
        for i, pred in enumerate(pred_list):
            if len(pred) == 0:
                continue
            # 生成掩膜
            slider.process_mask_np(
                protos[i], pred[:, 6:], pred[:, :4],
                img.shape[:2]
            )
        mask_times.append((time.time() - t0) * 1000)
    
    print("后处理细分:")
    print(f"  - NMS 非极大值抑制: {np.mean(nms_times):.1f}ms")
    print(f"  - 掩膜生成+处理:    {np.mean(mask_times) - np.mean(nms_times):.1f}ms")
    print()
    
    # 4. 模型层统计
    print("模型信息:")
    print(f"  输入: {[i.shape for i in slider.session.get_inputs()]}")
    print(f"  输出: {[o.shape for o in slider.session.get_outputs()]}")
    print(f"  节点数: {len(slider.session.get_outputs())}")
    print()
    
    print("=" * 70)
    print(f"结论: 瓶颈在 ONNX 模型推理 ({np.mean(inference_times)/np.mean(total_times)*100:.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    # 强制使用 FP32 模型
    import os
    fp32_path = '/workspace/captcha_recognizer/models/slider.onnx'
    
    slider = Slider()
    if 'int8' in slider.session._model_path.lower():
        # 如果是 INT8，重新加载 FP32
        import onnxruntime as ort
        slider.session = ort.InferenceSession(fp32_path, providers=['CPUExecutionProvider'])
        print(f"已切换到 FP32 模型: {fp32_path}")
    
    test_img = cv2.imread('/workspace/calibration_images/example8.png')
    if test_img is None:
        test_img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    
    profile_detailed(slider, test_img, runs=10)