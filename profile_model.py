"""
性能剖析脚本：分析 FP32 模型推理各阶段耗时
"""
import sys
import time
import cv2
import numpy as np

sys.path.insert(0, '/workspace')
from captcha_recognizer.slider import Slider


def profile_identify(slider, img, runs=5):
    """详细剖析 identify_both 各阶段耗时"""
    
    # ========== 1. 预处理 ==========
    preprocess_times = []
    for _ in range(runs):
        t0 = time.time()
        # 复制预处理逻辑
        im0 = img.copy()
        img_processed, ratio, pad_w, pad_h = slider._letterbox(im0, new_shape=(640, 640))
        blob = img_processed.astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)
        blob = np.ascontiguousarray(blob)
        preprocess_times.append((time.time() - t0) * 1000)
    
    # ========== 2. ONNX 推理 ==========
    inference_times = []
    for _ in range(runs):
        t0 = time.time()
        outputs = slider.session.run(None, {'images': blob})
        inference_times.append((time.time() - t0) * 1000)
    
    # ========== 3. 后处理整体 ==========
    total_times = []
    for _ in range(runs):
        t0 = time.time()
        result = slider.identify_both(source=img)
        total_times.append((time.time() - t0) * 1000)
    
    # ========== 4. 更细粒度后处理 ==========
    t0 = time.time()
    outputs = slider.session.run(None, {'images': blob})
    t_inference = (time.time() - t0) * 1000
    
    t0 = time.time()
    pred = np.squeeze(outputs[0]).T
    pred = pred[np.max(pred[:, 4:], axis=1) > slider.CONF_THRESHOLD]
    t_nms_filter = (time.time() - t0) * 1000
    
    t0 = time.time()
    boxes = []
    scores = []
    mask_coefs = []
    for det in pred:
        box = det[:4]
        score = np.max(det[4:])
        cls_id = np.argmax(det[4:])
        mask_coef = det[4 + slider.num_classes:]
        x_center, y_center, w, h = box
        x1 = (x_center - w / 2 - pad_w) / ratio
        y1 = (y_center - h / 2 - pad_h) / ratio
        x2 = (x_center + w / 2 - pad_w) / ratio
        y2 = (y_center + h / 2 - pad_h) / ratio
        boxes.append([x1, y1, x2, y2])
        scores.append(float(score))
        mask_coefs.append(mask_coef)
    t_box_decode = (time.time() - t0) * 1000
    
    t0 = time.time()
    if len(boxes) > 0:
        nms_indices = cv2.dnn.NMSBoxes(
            boxes, scores,
            slider.CONF_THRESHOLD,
            slider.IOU_THRESHOLD
        )
        if isinstance(nms_indices, np.ndarray):
            nms_indices = nms_indices.flatten()
    else:
        nms_indices = []
    t_nms = (time.time() - t0) * 1000
    
    t0 = time.time()
    if len(nms_indices) > 0:
        proto = outputs[1]
        for idx in nms_indices:
            mask = slider._seg_mask_from_proto(
                proto[0], np.array(mask_coefs[idx]),
                np.array(boxes[idx]),
                (640, 640),
                (int(img.shape[0]), int(img.shape[1]))
            )
    t_mask_process = (time.time() - t0) * 1000
    
    print("=" * 60)
    print("性能剖析结果")
    print("=" * 60)
    print(f"图片尺寸: {img.shape[1]}x{img.shape[0]}")
    print(f"模型输入: 640x640")
    print()
    print(f"总耗时 (identify_both): {np.mean(total_times):.1f}ms "
          f"(min={np.min(total_times):.1f}ms, max={np.max(total_times):.1f}ms)")
    print()
    print("各阶段耗时:")
    print(f"  1. 预处理 (letterbox+归一化): {np.mean(preprocess_times):.1f}ms")
    print(f"  2. ONNX 模型推理:           {np.mean(inference_times):.1f}ms "
          f"({np.mean(inference_times)/np.mean(total_times)*100:.1f}%)")
    print(f"  3. 后处理 (解析+NMS+掩膜):   "
          f"{np.mean(total_times) - np.mean(inference_times) - np.mean(preprocess_times):.1f}ms")
    print()
    print("后处理细分:")
    print(f"  - 置信度过滤:   {t_nms_filter:.1f}ms")
    print(f"  - 框坐标解析:   {t_box_decode:.1f}ms")
    print(f"  - NMS 去重:     {t_nms:.1f}ms")
    print(f"  - 掩膜处理:     {t_mask_process:.1f}ms")
    print()
    print("=" * 60)
    print(f"瓶颈: ONNX 推理占 {np.mean(inference_times)/np.mean(total_times)*100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    slider = Slider()
    
    test_img = cv2.imread('/workspace/calibration_images/example8.png')
    if test_img is None:
        test_img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    
    profile_identify(slider, test_img, runs=5)