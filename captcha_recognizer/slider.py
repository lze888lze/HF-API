"""
滑块验证码识别核心模块
========================
基于 YOLOv8-seg（实例分割）的 ONNX 模型，识别滑块验证码的滑块和缺口位置。

整体流程：
  原始图片 → 预处理(letterbox+归一化) → ONNX模型推理 → 后处理(NMS+掩膜) → 滑块/缺口坐标

主要对外方法：
  identify(source)        → 返回 (gap_box, confidence)  旧版接口，只返回缺口
  identify_both(source)   → 返回 dict，包含滑块+缺口+偏移量  新版接口
  identify_offset(source) → 返回 (offset, confidence)  只返回滑块x坐标

关键参数（可调，改这里调整识别灵敏度）：
  CONF_THRESHOLD = 0.5   置信度阈值，低于此值的目标会被丢弃
  IOU_THRESHOLD  = 0.8   NMS 用的 IoU 阈值
  Y_IOU_THRESHOLD = 0.85 Y轴方向 IoU 阈值，用于区分滑块和缺口
"""

import base64
import os
import random
import time
from pathlib import Path
from typing import List, Tuple, Union

import cv2      # OpenCV：图片读写、缩放、绘制
import numpy as np  # 数值计算
import onnxruntime as ort  # ONNX 模型推理引擎
from shapely.geometry import Polygon  # 多边形几何计算（计算IoU用）


# ============================================================
# 全局阈值参数（改这里可以调整识别的灵敏度）
# ============================================================

CONF_THRESHOLD = 0.5    # 置信度阈值：模型认为"这可能是缺口"的最低分数
                       # 调低 → 识别更多目标（可能误报增多）
                       # 调高 → 只保留高置信度结果（可能漏报）

IOU_THRESHOLD = 0.8    # NMS（非极大值抑制）的 IoU 阈值
                       # 两个框重叠度超过此值，只保留分数更高的那个

Y_IOU_THRESHOLD = 0.85  # Y轴方向 IoU 阈值，用于 pick_out_mask
                        # 判断两个目标是否在"同一水平线"上（滑块和缺口通常y位置接近）


class Slider:

    def __init__(self):
        """
        初始化：加载 ONNX 模型文件
        模型路径: captcha_recognizer/models/slider.onnx
        """
        root_dir = os.path.dirname(os.path.dirname(__file__))
        slider_model_path = os.path.join(root_dir, 'captcha_recognizer', 'models', 'slider.onnx')

        # 根据是否有 GPU 选择推理设备
        # HF Spaces 免费层没有 GPU，所以通常走 CPUExecutionProvider
        self.session = ort.InferenceSession(
            slider_model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"] if ort.get_device() == 'GPU' else [
                "CPUExecutionProvider"],
        )

        # 模型类别：只有一类 's'（slider/缺口）
        # 如果以后换成多类别模型，这里要加
        self.classes = {0: 's'}

    # ============================================================
    # 核心推理流程
    # ============================================================

    def predict(self, img: np.ndarray, conf: float = 0.25, iou: float = 0.7,
                imgsz: Union[int, Tuple[int, int]] = 640) -> List:
        """
        完整推理流程：预处理 → 模型推理 → 后处理
        
        参数：
          img:   原始图片（BGR格式的numpy数组）
          conf:  置信度阈值（传给NMS）
          iou:   IoU阈值（传给NMS）
          imgsz: 模型输入尺寸，默认640x640
                  ↑ 更大=更准但更慢，更小=更快但可能不准
                  ↑ 常见值: 320, 640, 1280
        """
        imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
        prep_img = self.preprocess(img, imgsz)
        outs = self.session.run(None, {self.session.get_inputs()[0].name: prep_img})
        return self.postprocess(img, prep_img, outs, conf=conf, iou=iou)

    @staticmethod
    def letterbox(img: np.ndarray, new_shape: Tuple[int, int] = (640, 640)) -> np.ndarray:
        """
        Letterbox 缩放：保持宽高比缩放图片，不足部分用灰色(114,114,114)填充
        
        为什么不直接 resize？因为直接拉伸会变形，影响识别准确率。
        Letterbox 相当于"等比缩放 + 补边"，是 YOLO 系列模型的标准做法。
        
        示例：原图 300x200，目标 640x640
          → 等比缩放到 640x427
          → 上下各补 106 像素灰色边，最终 640x640
        """
        shape = img.shape[:2]  # 当前尺寸 [height, width]

        # 计算缩放比例（取宽高比中较小的，保证整个图片都能放下）
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

        # 确保尺寸不越界
        new_unpad = (max(1, min(new_unpad[0], new_shape[1])),
                     max(1, min(new_unpad[1], new_shape[0])))

        # 缩放
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        # 计算需要填充的像素数
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw, dh = float(dw), float(dh)

        # 上下、左右各填一半
        top, bottom = int(round(dh / 2)), int(round(dh / 2))
        left, right = int(round(dw / 2)), int(round(dw / 2))

        # 填充灰色边
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # 最终确保精确尺寸（防止四舍五入导致差1像素）
        if img.shape[0] != new_shape[0] or img.shape[1] != new_shape[1]:
            img = cv2.resize(img, new_shape, interpolation=cv2.INTER_LINEAR)

        return img

    def preprocess(self, img: np.ndarray, new_shape: Tuple[int, int]) -> np.ndarray:
        """
        图片预处理：letterbox缩放 → BGR转RGB → 转置 → 归一化
        
        处理后格式: (1, 3, 640, 640) float32，值域 [0, 1]
        这是 YOLO 模型的标准输入格式
        """
        img = self.letterbox(img, new_shape)     # 等比缩放+填充
        img = img[..., ::-1].transpose([2, 0, 1])[None]  # BGR→RGB, HWC→CHW, 加batch维度
        img = np.ascontiguousarray(img)          # 确保内存连续
        img = img.astype(np.float32) / 255       # 归一化到 0~1
        return img

    def postprocess(self, img: np.ndarray, prep_img: np.ndarray, outs: List, conf: float = 0.25,
                    iou: float = 0.7) -> List:
        """
        后处理：模型输出 → 有意义的检测框和掩膜
        
        模型输出两个部分：
          preds:  检测框 + 类别 + 置信度
          protos: 掩膜原型（用于生成分割掩膜）
        """
        preds, protos = outs
        preds = self.non_max_suppression(preds, conf, iou, nc=len(self.classes))

        results = []
        for i, pred in enumerate(preds):
            if len(pred) == 0:
                results.append([pred, None])
                continue
            # 把检测框从模型输入坐标映射回原图坐标
            pred[:, :4] = self.scale_boxes(prep_img.shape[2:], pred[:, :4], img.shape)
            # 用掩膜原型 + 检测框系数 → 生成分割掩膜
            masks = self.process_mask(protos[i], pred[:, 6:], pred[:, :4], img.shape[:2])
            results.append([pred[:, :6], masks])

        return results

    # ============================================================
    # 掩膜处理相关
    # ============================================================

    def process_mask(self, protos: np.ndarray, masks_in: np.ndarray, bboxes: np.ndarray,
                     shape: Tuple[int, int]) -> np.ndarray:
        """
        从掩膜原型生成最终的二值掩膜
        
        原理：掩膜系数 × 掩膜原型 = 每个目标的分割掩膜
        """
        c, mh, mw = protos.shape
        masks = (masks_in @ protos.reshape(c, -1)).reshape(-1, mh, mw)
        masks = self.scale_masks(masks, shape)
        masks = self.crop_mask(masks, bboxes)
        return masks > 0.0

    @staticmethod
    def masks_to_segments(masks: Union[np.ndarray,], strategy: str = "largest") -> List[np.ndarray]:
        """
        将二值掩膜转换为多边形轮廓点
        
        为什么需要这个？因为后面要计算两个多边形的 IoU
        来判断哪个是滑块、哪个是缺口
        
        strategy:
          'largest' - 只保留最大轮廓（默认，适合大多数情况）
          'all'     - 合并所有轮廓
          'none'    - 保留所有轮廓不合并
        """
        masks_np = masks.astype("uint8")

        if masks_np.ndim == 2:
            masks_np = masks_np[np.newaxis, ...]

        segments = []
        for mask in masks_np:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                segments.append(np.zeros((0, 2), dtype=np.float32))
                continue

            if strategy == "all" and len(contours) > 1:
                contour = np.concatenate([x.reshape(-1, 2) for x in contours])
            elif strategy == "largest":
                contour = max(contours, key=lambda x: cv2.arcLength(x, closed=True))
                contour = contour.reshape(-1, 2)
            else:
                contour = contours[0].reshape(-1, 2)

            segments.append(contour.astype(np.float32))

        return segments[0] if masks_np.shape[0] == 1 else segments

    @staticmethod
    def draw_segments(image, boxes, masks,
                      mask_alpha=0.5, box_thickness=2, draw_labels=True):
        """
        在图片上绘制检测框和掩膜（调试用，API服务中不会调用）
        """
        output = image.copy()

        if boxes is None and masks is None:
            return output

        if masks is not None:
            color_mask = np.zeros_like(image)
            for i, mask in enumerate(masks):
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                mask = mask.astype(bool)
                color_mask[mask] = color
            output = cv2.addWeighted(output, 1, color_mask, mask_alpha, 0)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2, score, class_id = box[:6]
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), color, box_thickness)
                if draw_labels:
                    label = f"{int(class_id)}: {score:.2f}"
                    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(output, (int(x1), int(y1) - label_height - 5),
                                  (int(x1) + label_width, int(y1)), color, -1)
                    cv2.putText(output, label, (int(x1), int(y1) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        return output

    # ============================================================
    # 图片输入处理
    # ============================================================

    @staticmethod
    def image_to_array(source: Union[str, Path, bytes, np.ndarray] = None):
        """
        把各种格式的输入统一转成 OpenCV 图片（numpy数组）
        
        支持的输入：
          - base64 字符串（data:image/png;base64,...）
          - 文件路径
          - 字节流
          - 已经是 numpy 数组的（直接返回）
        """
        if isinstance(source, str) and source.startswith('data:image'):
            header, encoded = source.split(',', 1)
            data = base64.b64decode(encoded)
            np_arr = np.frombuffer(data, np.uint8)
            return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        elif isinstance(source, (str, Path)):
            return cv2.imread(str(source))
        elif isinstance(source, bytes):
            np_arr = np.frombuffer(source, np.uint8)
            return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        elif isinstance(source, np.ndarray):
            return source
        else:
            raise TypeError("Unsupported source type. Only str, Path, bytes, or numpy.ndarray are supported.")

    # ============================================================
    # IoU 计算相关（用于区分滑块和缺口）
    # ============================================================

    @staticmethod
    def normalize_points(points):
        """将多边形点集归一化到以质心为原点（用于形状比较，消除位置影响）"""
        centroid = np.mean(points, axis=0)
        normalized_points = points - centroid
        return normalized_points

    @staticmethod
    def y_iou(segment1, segment2):
        """
        计算 Y 轴方向的一维 IoU
        
        用途：判断两个目标是否在同一水平线上
        滑块和缺口通常 y 坐标接近（在同一高度），而背景干扰物可能在不同高度
        """
        start = max(segment1[0], segment2[0])
        end = min(segment1[1], segment2[1])
        intersection = max(0, end - start)

        len1 = segment1[1] - segment1[0]
        len2 = segment2[1] - segment2[0]
        union = len1 + len2 - intersection

        iou = intersection / union if union != 0 else 0
        return iou

    def polygon_iou(self, poly1, poly2):
        """
        计算两个多边形的形状 IoU（先归一化位置，只比较形状相似度）
        
        用途：滑块的形状和缺口的形状通常相似（都是拼图块形状）
        通过比较形状 IoU 来判断哪个目标是缺口
        """
        p1 = self.normalize_points(poly1)
        p2 = self.normalize_points(poly2)

        poly1 = Polygon(p1).buffer(0)  # buffer(0) 修复自相交等无效多边形
        poly2 = Polygon(p2).buffer(0)

        intersect = poly1.intersection(poly2).area
        union = poly1.union(poly2).area

        iou = intersect / union if union > 0 else 0.0
        return iou

    # ============================================================
    # 核心：从多个检测目标中区分滑块和缺口
    # ============================================================

    def pick_out_mask(self, boxes: list, segments):
        """
        从多个检测目标中区分滑块和缺口
        
        返回: (slider_box, gap_box)
          slider_box: 滑块（x最小的检测框，通常在图片左侧）
          gap_box: 缺口（与滑块形状最相似的目标，通常在图片右侧）
        
        策略：
        1. 找 x 坐标最小的目标 → 这通常是滑块（在图片左侧）
        2. 在剩余目标中，找 y 位置与滑块接近的（同一水平线）
        3. 在同水平线的目标中，找形状与滑块最相似的 → 这就是缺口
        
        为什么这样判断？
        - 滑块验证码的布局：滑块在左，缺口在右
        - 滑块和缺口形状相同（都是拼图块），但位置不同
        - 背景干扰物形状不同，且可能不在同一水平线
        """
        # 第一步：找 x 最小的 = 滑块
        box_slider = min(boxes, key=lambda x: x[0])
        box_slider_index = boxes.index(box_slider)
        segment_slider = segments[box_slider_index]

        # 剩余目标（排除滑块）
        box_sample = boxes[:box_slider_index] + boxes[box_slider_index + 1:]
        segment_sample = segments[:box_slider_index] + segments[box_slider_index + 1:]

        # 第二步：Y轴方向 IoU 过滤（只保留同一水平线的目标）
        box_filtered = []
        segment_filtered = []

        for index, box in enumerate(box_sample):
            if self.y_iou([box_slider[1], box_slider[3]], [box[1], box[3]]) > Y_IOU_THRESHOLD:
                box_filtered.append(box)
                segment_filtered.append(segment_sample[index])
        # 如果Y轴过滤没有保留任何目标，退回到全部候选
        if not box_filtered:
            box_filtered = box_sample
            segment_filtered = segment_sample

        if len(box_filtered) == 1:
            return box_slider, box_filtered[0]

        # 第三步：找形状与滑块最相似的 = 缺口
        iou_flag = 0
        iou_index = 0
        for index, segment in enumerate(segment_filtered):
            segment_iou = self.polygon_iou(segment_slider, segment)
            if segment_iou > iou_flag:
                iou_flag = segment_iou
                iou_index = index

        return box_slider, box_filtered[iou_index]

    # ============================================================
    # 对外接口：识别缺口
    # ============================================================

    def identify(self, source: Union[str, Path, bytes, np.ndarray], conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, show=False):
        """
        识别滑块验证码缺口位置（兼容旧接口）
        
        返回：
          (gap_box, confidence)
          gap_box = [x1, y1, x2, y2] 缺口的左上角和右下角坐标
          confidence = 0~1 置信度
        
        如果没检测到缺口：返回 ([], 0.0)
        """
        result = self.identify_both(source, conf=conf, iou=iou, show=show)
        return result['gap'], result['gap_confidence']

    def identify_both(self, source: Union[str, Path, bytes, np.ndarray], conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, show=False):
        """
        同时识别滑块和缺口位置（新版接口）
        
        参数：
          source: 图片输入（路径/base64/字节/numpy数组）
          conf:   置信度阈值（默认0.5）
          iou:    NMS IoU阈值（默认0.8）
          show:   是否显示识别结果（调试用，服务器上别开）
        
        返回 dict：
          slider: [x1, y1, x2, y2] 滑块坐标（空列表=未检测到）
          slider_confidence: float 滑块置信度
          gap: [x1, y1, x2, y2] 缺口坐标（空列表=未检测到）
          gap_confidence: float 缺口置信度
          offset: int 滑动距离 = 缺口x1 - 滑块x1（0表示无法计算）
        """
        slider_box_list = []
        gap_box_list = []

        original_image: np.ndarray = self.image_to_array(source)
        results = self.predict(original_image, conf=conf, iou=iou, imgsz=640)

        if results:
            boxes, masks = results[0]
            if len(boxes) == 0:
                pass  # 没检测到任何目标
            elif len(boxes) == 1:
                # 只检测到一个目标，无法区分滑块/缺口，当作缺口
                gap_box_list = boxes[0].tolist()
            else:
                # 多目标：区分滑块和缺口
                segments = self.masks_to_segments(masks)
                slider_box_list, gap_box_list = self.pick_out_mask(boxes.tolist(), segments)

        # 调试用：显示识别结果
        if show:
            draw_boxes = []
            draw_masks = []
            boxes_np, masks_np = results[0] if results else (np.zeros((0, 6)), None)
            if gap_box_list and masks_np is not None:
                gap_idx = boxes_np.tolist().index(gap_box_list) if gap_box_list in boxes_np.tolist() else -1
                if gap_idx >= 0:
                    draw_boxes.append(gap_box_list)
                    draw_masks.append(masks_np[gap_idx])
            if slider_box_list and masks_np is not None:
                slider_idx = boxes_np.tolist().index(slider_box_list) if slider_box_list in boxes_np.tolist() else -1
                if slider_idx >= 0:
                    draw_boxes.append(slider_box_list)
                    draw_masks.append(masks_np[slider_idx])
            if draw_boxes:
                sample = self.draw_segments(original_image, draw_boxes, draw_masks)
                cv2.imshow('result', sample)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # 提取坐标和置信度
        slider = [int(x) for x in slider_box_list[:4]] if slider_box_list else []
        slider_conf = float(slider_box_list[4]) if slider_box_list else 0.0
        gap = [int(x) for x in gap_box_list[:4]] if gap_box_list else []
        gap_conf = float(gap_box_list[4]) if gap_box_list else 0.0

        # 计算滑动距离：缺口x1 - 滑块x1
        # 这就是滑块需要从当前位置滑动到缺口的距离
        if slider and gap:
            offset = gap[0] - slider[0]
        else:
            offset = 0

        return {
            'slider': slider,
            'slider_confidence': slider_conf,
            'gap': gap,
            'gap_confidence': gap_conf,
            'offset': offset,
        }

    def identify_offset(self, source: Union[str, Path, bytes, np.ndarray], conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
                        show=False):
        """
        识别缺口并直接返回偏移量（滑块初始x坐标）
        
        注意：这里返回的 offset 是滑块自身的 x1 坐标，
        不是滑动距离。要算滑动距离请用 identify_both() 的 offset 字段。
        
        用途：某些验证码的滑块有固定偏移量，可用此方法获取
        """
        box_list = []
        mask_ndarray = None

        original_image: np.ndarray = self.image_to_array(source)
        results = self.predict(original_image, conf=conf, iou=iou, imgsz=640)

        if results:
            boxes, masks = results[0]
            if len(boxes) == 0:
                pass
            elif len(boxes) == 1:
                box_list = boxes[0].tolist()
                mask_ndarray = masks[0]
            else:
                # 多目标时选 x 最小的（最左边的 = 滑块位置）
                box_left = min(boxes, key=lambda x: x[0])
                box_list = box_left.tolist()
                mask_ndarray = masks[boxes.tolist().index(box_list)]

        if show and box_list and mask_ndarray is not None:
            sample = self.draw_segments(original_image, [box_list, ], [mask_ndarray, ])
            cv2.imshow('result', sample)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if box_list:
            box = box_list[:4]
            box_conf = float(box_list[4])
            offset = box[0]  # 缺口/滑块的 x1 坐标
        else:
            offset = 0
            box_conf = 0.0

        return offset, box_conf

    # ============================================================
    # 以下是 YOLO 后处理的标准工具方法
    # 一般不需要修改，除非换模型
    # ============================================================

    def scale_boxes(self, img1_shape, boxes, img0_shape, ratio_pad=None, padding=True, xywh=False):
        """将检测框从模型输入坐标映射回原图坐标（逆letterbox）"""
        if ratio_pad is None:
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
            pad = (
                round((img1_shape[1] - img0_shape[1] * gain) / 2),
                round((img1_shape[0] - img0_shape[0] * gain) / 2),
            )
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        if padding:
            boxes[..., 0] -= pad[0]
            boxes[..., 1] -= pad[1]
            if not xywh:
                boxes[..., 2] -= pad[0]
                boxes[..., 3] -= pad[1]
        boxes[..., :4] /= gain
        return self.clip_boxes(boxes, img0_shape)

    @staticmethod
    def get_covariance_matrix(boxes: np.ndarray):
        """从旋转边界框生成协方差矩阵（用于旋转框NMS）"""
        gbbs = np.concatenate((np.power(boxes[:, 2:4], 2) / 12, boxes[:, 4:]), axis=-1)
        a, b, c = np.split(gbbs, [1, 2], axis=-1)
        cos = np.cos(c)
        sin = np.sin(c)
        cos2 = np.power(cos, 2)
        sin2 = np.power(sin, 2)
        return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin

    def batch_probiou(self, obb1, obb2, eps=1e-7):
        """计算旋转边界框的概率IoU"""
        x1, y1 = np.split(obb1[..., :2], 2, axis=-1)
        x2, y2 = (np.expand_dims(x.squeeze(-1), 0) for x in np.split(obb2[..., :2], 2, axis=-1))
        a1, b1, c1 = self.get_covariance_matrix(obb1)
        a2, b2, c2 = (np.expand_dims(x.squeeze(-1), 0) for x in self.get_covariance_matrix(obb2))

        t1 = (
                     ((a1 + a2) * np.power(y1 - y2, 2) + (b1 + b2) * np.power(x1 - x2, 2)) / (
                     (a1 + a2) * (b1 + b2) - np.power(c1 + c2, 2) + eps)
             ) * 0.25
        t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - np.power(c1 + c2, 2) + eps)) * 0.5

        term1_log = (a1 * b1 - np.power(c1, 2)).clip(0)
        term2_log = (a2 * b2 - np.power(c2, 2)).clip(0)

        denominator = 4 * np.sqrt(term1_log * term2_log) + eps
        t3_numerator = (a1 + a2) * (b1 + b2) - np.power(c1 + c2, 2)
        t3_arg = np.clip(t3_numerator / denominator + eps, eps, None)
        t3 = np.log(t3_arg) * 0.5

        bd = (t1 + t2 + t3).clip(eps, 100.0)
        hd = np.sqrt(1.0 - np.exp(-bd) + eps)
        return 1 - hd

    def nms_rotated(self, boxes, scores, threshold=0.45):
        """旋转边界框的NMS"""
        sorted_idx = np.argsort(scores)[::-1]
        boxes = boxes[sorted_idx]
        ious = self.batch_probiou(boxes, boxes)
        n = boxes.shape[0]
        ious[np.tril_indices(n)] = 0
        pick = np.where((ious >= threshold).sum(axis=0) <= 0)[0]
        return sorted_idx[pick]

    def clip_boxes(self, boxes, shape):
        """将检测框裁剪到图片范围内（防止越界）"""
        boxes[..., [0, 2]] = np.clip(boxes[..., [0, 2]], 0, shape[1])
        boxes[..., [1, 3]] = np.clip(boxes[..., [1, 3]], 0, shape[0])
        return boxes

    @staticmethod
    def xywh2xyxy(x):
        """坐标格式转换：中心点+宽高 → 左上角+右下角"""
        assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
        y = np.empty_like(x, dtype=np.float32)
        xy = x[..., :2]
        wh = x[..., 2:] / 2
        y[..., :2] = xy - wh
        y[..., 2:] = xy + wh
        return y

    @staticmethod
    def crop_mask(masks, boxes):
        """将掩膜裁剪到检测框范围内（框外的掩膜置零）"""
        _, h, w = masks.shape
        boxes = boxes[:, :, None] if boxes.ndim == 2 else boxes
        x1, y1, x2, y2 = np.split(boxes, 4, axis=1)
        r = np.arange(w, dtype=x1.dtype)[None, None, :]
        c = np.arange(h, dtype=x1.dtype)[None, :, None]
        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

    def process_mask_np(self, protos, masks_in, bboxes, shape, upsample=False):
        """另一版本的掩膜处理（带下采样坐标映射）"""
        c, mh, mw = protos.shape
        ih, iw = shape

        masks = (masks_in @ protos.reshape(c, -1)).reshape(-1, mh, mw)
        width_ratio = mw / iw
        height_ratio = mh / ih

        downsampled_bboxes = bboxes.copy()
        downsampled_bboxes[:, 0] *= width_ratio
        downsampled_bboxes[:, 2] *= width_ratio
        downsampled_bboxes[:, 3] *= height_ratio
        downsampled_bboxes[:, 1] *= height_ratio

        masks = self.crop_mask(masks, downsampled_bboxes)
        if upsample:
            masks = cv2.resize(masks.transpose((1, 2, 0)),
                               (shape[1], shape[0]),
                               interpolation=cv2.INTER_LINEAR).transpose((2, 0, 1))

        return masks > 0.0

    @staticmethod
    def scale_masks(masks, shape, padding=True):
        """将掩膜从模型输出尺寸缩放到原图尺寸"""
        mh, mw = masks.shape[1:]
        gain = min(mh / shape[0], mw / shape[1])
        pad = [mw - shape[1] * gain, mh - shape[0] * gain]

        if padding:
            pad[0] /= 2
            pad[1] /= 2

        top, left = (int(round(pad[1])), int(round(pad[0]))) if padding else (0, 0)
        bottom, right = (mh - int(round(pad[1])), mw - int(round(pad[0])))

        masks_cropped = masks[:, top:bottom, left:right]

        resized_masks = np.zeros((masks_cropped.shape[0], shape[0], shape[1]), dtype=masks_cropped.dtype)
        for i, mask in enumerate(masks_cropped):
            resized_masks[i] = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)

        return resized_masks

    def non_max_suppression(self, prediction, conf_thres=0.25, iou_thres=0.45,
                            classes=None, agnostic=False, multi_label=False, labels=(),
                            max_det=300, nc=0, max_time_img=0.05, max_nms=30000,
                            max_wh=7680, in_place=True, rotated=False, end2end=False,
                            return_idxs=False):
        """
        非极大值抑制（NMS）
        
        作用：模型可能对同一个目标检测出多个重叠的框，
        NMS 会保留分数最高的，去掉与它重叠太多的其他框。
        
        一般不需要改这里的参数，调 CONF_THRESHOLD 和 IOU_THRESHOLD 就够了
        """
        assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}"
        assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}"

        if isinstance(prediction, (list, tuple)):
            prediction = prediction[0]
        if classes is not None:
            classes = np.array(classes)

        if prediction.shape[-1] == 6 or end2end:
            output = [pred[pred[:, 4] > conf_thres][:max_det] for pred in prediction]
            if classes is not None:
                output = [pred[np.any(pred[:, 5:6] == classes, axis=1)] for pred in output]
            return output

        bs = prediction.shape[0]
        nc = nc or (prediction.shape[1] - 4)
        extra = prediction.shape[1] - nc - 4
        mi = 4 + nc
        xc = np.amax(prediction[:, 4:mi], axis=1) > conf_thres
        xinds = np.stack([np.arange(len(i)) for i in xc])[..., None]

        time_limit = 2.0 + max_time_img * bs
        multi_label &= nc > 1

        prediction = np.transpose(prediction, (0, 2, 1))
        if not rotated:
            if in_place:
                prediction[..., :4] = self.xywh2xyxy(prediction[..., :4])
            else:
                prediction = np.concatenate((self.xywh2xyxy(prediction[..., :4]), prediction[..., 4:]), axis=-1)

        t = time.time()
        output = [np.zeros((0, 6 + extra), dtype=np.float32)] * bs
        keepi = [np.zeros((0, 1), dtype=np.int64)] * bs
        for xi, (x, xk) in enumerate(zip(prediction, xinds)):
            filt = xc[xi]
            x, xk = x[filt], xk[filt]

            if labels and len(labels) > xi and len(labels[xi]) and not rotated:
                lb = np.array(labels[xi])
                if lb.size > 0:
                    v = np.zeros((len(lb), nc + extra + 4), dtype=np.float32)
                    v[:, :4] = self.xywh2xyxy(lb[:, 1:5])
                    v[range(len(lb)), lb[:, 0].astype(np.int64) + 4] = 1.0
                    x = np.concatenate((x, v), axis=0)

            if not x.shape[0]:
                continue

            box, cls, mask = np.split(x, [4, 4 + nc], axis=1)

            if multi_label:
                i, j = np.where(cls > conf_thres)
                x = np.concatenate((box[i], x[i, 4 + j, None], j[:, None].astype(np.float32), mask[i]), axis=1)
                xk = xk[i]
            else:
                conf = np.amax(cls, axis=1, keepdims=True)
                j = np.argmax(cls, axis=1, keepdims=True)
                filt = conf.squeeze(-1) > conf_thres
                x = np.concatenate((box, conf, j.astype(np.float32), mask), axis=1)[filt]
                xk = xk[filt]

            if classes is not None:
                filt = np.any(x[:, 5:6] == classes, axis=1)
                x, xk = x[filt], xk[filt]

            n = x.shape[0]
            if not n:
                continue
            if n > max_nms:
                filt = np.argsort(x[:, 4])[::-1][:max_nms]
                x, xk = x[filt], xk[filt]

            c = x[:, 5:6] * (0 if agnostic else max_wh)
            scores = x[:, 4]

            if rotated:
                boxes = np.concatenate((x[:, :2] + c, x[:, 2:4], x[:, -1:]), axis=-1)
                i = self.nms_rotated(boxes, scores, iou_thres)
            else:
                boxes = x[:, :4] + c
                i = []
                if boxes.shape[0] > 0:
                    y1, x1, y2, x2 = boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]
                    area = (x2 - x1) * (y2 - y1)
                    order = scores.argsort()[::-1]
                    while order.size > 0:
                        idx = order[0]
                        i.append(idx)
                        xx1 = np.maximum(x1[idx], x1[order[1:]])
                        yy1 = np.maximum(y1[idx], y1[order[1:]])
                        xx2 = np.minimum(x2[idx], x2[order[1:]])
                        yy2 = np.minimum(y2[idx], y2[order[1:]])
                        w = np.maximum(0.0, xx2 - xx1)
                        h = np.maximum(0.0, yy2 - yy1)
                        inter = w * h
                        iou = inter / (area[idx] + area[order[1:]] - inter)
                        order = order[np.where(iou <= iou_thres)[0] + 1]
                i = np.array(i)

            i = i[:max_det]

            output[xi], keepi[xi] = x[i], xk[i].reshape(-1)
            if (time.time() - t) > time_limit:
                break

        return (output, keepi) if return_idxs else output


if __name__ == "__main__":
    """本地测试：直接运行识别单张图片"""
    model = Slider()
    # 测试新版接口（同时返回滑块和缺口）
    res = model.identify_both(source='img_example.png', show=True)
    print(f'滑块: {res["slider"]}, 置信度: {res["slider_confidence"]}')
    print(f'缺口: {res["gap"]}, 置信度: {res["gap_confidence"]}')
    print(f'滑动距离: {res["offset"]}')
