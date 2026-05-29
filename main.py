"""
滑块验证码识别 API 服务
========================
基于 FastAPI + ONNX 模型，提供 HTTP 接口识别滑块验证码的滑块和缺口位置。

接口说明：
  GET  /                  - 健康检查
  GET  /health            - 健康检查

  POST /slide             - 上传图片文件，只返回滑块位置
  POST /slide-base64      - 上传 base64 图片，只返回滑块位置
  POST /hole              - 上传图片文件，只返回缺口位置
  POST /hole-base64       - 上传 base64 图片，只返回缺口位置
  POST /puzzle            - 上传图片文件，返回滑块+缺口+滑缺距
  POST /puzzle-base64     - 上传 base64 图片，返回滑块+缺口+滑缺距

返回格式：
  /slide、/slide-base64：
    {
      "滑块": [x1, y1, x2, y2],       // 滑块坐标（左边的那个）
      "相似度": 0.92,                  // 滑块检测置信度
      "消息": null                     // 错误信息，正常为 null
    }

  /hole、/hole-base64：
    {
      "缺口": [x1, y1, x2, y2],       // 缺口坐标（左边的那个）
      "相似度": 0.88,                  // 缺口检测置信度
      "消息": null                     // 错误信息，正常为 null
    }

  /puzzle、/puzzle-base64：
    {
      "滑块": [x1, y1, x2, y2],       // 滑块坐标
      "滑块相似度": 0.92,              // 滑块检测置信度
      "缺口": [x1, y1, x2, y2],       // 缺口坐标
      "缺口相似度": 0.88,              // 缺口检测置信度
      "滑缺距": 162,                   // 滑动距离 = 缺口x1 - 滑块x1
      "消息": null                     // 错误信息，正常为 null
    }
"""

import gc       # 垃圾回收，手动释放内存（HF免费层512MB）
import os       # 读取环境变量（端口号）
from typing import Optional

import cv2      # OpenCV，图片解码
import numpy as np  # 数组操作

from captcha_recognizer.slider import Slider  # 核心：滑块识别模型
from fastapi import FastAPI, File, UploadFile  # Web 框架
from fastapi.middleware.cors import CORSMiddleware  # 跨域支持


# ============================================================
# FastAPI 应用初始化
# ============================================================

app = FastAPI(
    docs_url=None,   # 关闭 /docs（Swagger UI），避免暴露接口文档
    redoc_url=None,  # 关闭 /redoc，同上
)

# 允许跨域请求（任何域名都能调用这个 API）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # 允许的来源域名，* 表示全部
    allow_methods=["*"],    # 允许的 HTTP 方法
    allow_headers=["*"],    # 允许的请求头
)


# ============================================================
# 模型加载（服务启动时只加载一次）
# ============================================================
# Slider() 会加载 slider.onnx 模型文件到内存
# 放在全局意味着整个服务生命周期只加载一次，不用每次请求都读模型
slider_model = Slider()


# ============================================================
# 健康检查接口
# ============================================================

@app.get("/")
def hello_captcha():
    """根路径，简单返回服务状态，可用于判断服务是否存活"""
    return {"Hello": "Captcha", "status": "running"}


@app.get("/health")
def health():
    """健康检查接口，部署平台（如 HF Spaces）会定期调用"""
    return {"status": "ok"}


# ============================================================
# 内部公共方法：图片字节 → 模型识别结果
# ============================================================

def _recognize(image_bytes: bytes) -> dict:
    """
    图片字节 → 识别结果（原始 dict）

    流程：字节 → OpenCV图片 → 模型推理 → 返回 identify_both 的原始结果

    返回 dict 结构（与 Slider.identify_both 一致）：
      slider:              [x1, y1, x2, y2] 滑块坐标，空列表=未检测到
      slider_confidence:   float 滑块置信度
      gap:                 [x1, y1, x2, y2] 缺口坐标，空列表=未检测到
      gap_confidence:      float 缺口置信度
      offset:              int 滑动距离 = 缺口x1 - 滑块x1

    所有接口共用这个方法，避免重复代码
    """
    # 字节流 → numpy 数组 → OpenCV 图片
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return None  # 图片解码失败，由调用方处理

    # 调用模型，同时获取滑块和缺口的原始结果
    result = slider_model.identify_both(source=image)

    # 手动释放内存（HF 免费层内存有限，防止 OOM）
    del image, nparr
    gc.collect()

    return result


# ============================================================
# base64 解码公共方法
# ============================================================

def _decode_base64(data: dict) -> Optional[bytes]:
    """
    从请求 body 中提取 base64 图片并解码为字节

    参数：
      data: 请求 JSON，需包含 "image" 字段
             支持两种格式：
             - 带前缀：data:image/png;base64,iVBORw0KGgo...
             - 纯 base64：iVBORw0KGgo...

    返回：
      成功 → 图片字节流
      失败 → None（调用方根据返回值判断错误类型）
    """
    import base64

    image_b64 = data.get("image", "")
    if not image_b64:
        return None

    # 去掉 base64 头（如果有），如 "data:image/png;base64,"
    if "," in image_b64:
        image_b64 = image_b64.split(",")[1]

    try:
        return base64.b64decode(image_b64)
    except Exception:
        return None


# ============================================================
# 接口1 & 2：/slide、/slide-base64 — 只返回滑块位置
# ============================================================

@app.post("/slide")
async def slide(file: UploadFile = File(...)):
    """
    上传图片文件，只返回滑块位置（图片左侧的那个拼图块）

    用法：POST /slide，body 里传图片文件（form-data）

    返回示例：
    {
      "滑块": [91, 1045, 248, 1200],
      "相似度": 0.9487,
      "消息": null
    }
    """
    contents = await file.read()
    result = _recognize(contents)

    # 图片解码失败
    if result is None:
        return {"滑块": [], "相似度": 0.0, "消息": "不支持的图片格式"}

    return {
        "滑块": result["slider"],
        "相似度": round(result["slider_confidence"], 4),  # 保留4位小数，避免返回超长浮点数
        "消息": None,
    }


@app.post("/slide-base64")
async def slide_base64(data: dict):
    """
    上传 base64 图片，只返回滑块位置

    请求格式：{"image": "base64字符串"}
    适合懒人精灵等脚本通过 HTTP POST 调用

    base64 字符串支持两种格式：
    - 带 data:image 前缀的：data:image/png;base64,iVBORw0KGgo...
    - 纯 base64：iVBORw0KGgo...
    """
    image_bytes = _decode_base64(data)

    # base64 解码失败或缺少参数
    if image_bytes is None:
        return {"滑块": [], "相似度": 0.0, "消息": "缺少image参数或base64解码失败"}

    result = _recognize(image_bytes)

    # 图片解码失败
    if result is None:
        return {"滑块": [], "相似度": 0.0, "消息": "不支持的图片格式"}

    return {
        "滑块": result["slider"],
        "相似度": round(result["slider_confidence"], 4),
        "消息": None,
    }


# ============================================================
# 接口3 & 4：/hole、/hole-base64 — 只返回缺口位置
# ============================================================

@app.post("/hole")
async def hole(file: UploadFile = File(...)):
    """
    上传图片文件，只返回缺口位置（背景上的那个空白拼图区域）

    用法：POST /hole，body 里传图片文件（form-data）

    返回示例：
    {
      "缺口": [803, 1044, 957, 1201],
      "相似度": 0.9474,
      "消息": null
    }
    """
    contents = await file.read()
    result = _recognize(contents)

    if result is None:
        return {"缺口": [], "相似度": 0.0, "消息": "不支持的图片格式"}

    return {
        "缺口": result["gap"],
        "相似度": round(result["gap_confidence"], 4),
        "消息": None,
    }


@app.post("/hole-base64")
async def hole_base64(data: dict):
    """
    上传 base64 图片，只返回缺口位置

    请求格式：{"image": "base64字符串"}
    """
    image_bytes = _decode_base64(data)

    if image_bytes is None:
        return {"缺口": [], "相似度": 0.0, "消息": "缺少image参数或base64解码失败"}

    result = _recognize(image_bytes)

    if result is None:
        return {"缺口": [], "相似度": 0.0, "消息": "不支持的图片格式"}

    return {
        "缺口": result["gap"],
        "相似度": round(result["gap_confidence"], 4),
        "消息": None,
    }


# ============================================================
# 接口5 & 6：/puzzle、/puzzle-base64 — 返回滑块+缺口+滑缺距
# ============================================================

@app.post("/puzzle")
async def puzzle(file: UploadFile = File(...)):
    """
    上传图片文件，返回滑块位置 + 缺口位置 + 滑缺距（滑动距离）

    用法：POST /puzzle，body 里传图片文件（form-data）

    返回示例：
    {
      "滑块": [91, 1045, 248, 1200],
      "滑块相似度": 0.9487,
      "缺口": [803, 1044, 957, 1201],
      "缺口相似度": 0.9474,
      "滑缺距": 712,
      "消息": null
    }

    "滑缺距" = 缺口x1 - 滑块x1，就是滑块需要从当前位置滑到缺口的像素距离
    """
    contents = await file.read()
    result = _recognize(contents)

    if result is None:
        return {
            "滑块": [], "滑块相似度": 0.0,
            "缺口": [], "缺口相似度": 0.0,
            "滑缺距": 0, "消息": "不支持的图片格式",
        }

    return {
        "滑块": result["slider"],
        "滑块相似度": round(result["slider_confidence"], 4),
        "缺口": result["gap"],
        "缺口相似度": round(result["gap_confidence"], 4),
        "滑缺距": result["offset"],
        "消息": None,
    }


@app.post("/puzzle-base64")
async def puzzle_base64(data: dict):
    """
    上传 base64 图片，返回滑块位置 + 缺口位置 + 滑缺距

    请求格式：{"image": "base64字符串"}
    适合懒人精灵等脚本通过 HTTP POST 调用

    base64 字符串支持两种格式：
    - 带 data:image 前缀的：data:image/png;base64,iVBORw0KGgo...
    - 纯 base64：iVBORw0KGgo...
    """
    image_bytes = _decode_base64(data)

    if image_bytes is None:
        return {
            "滑块": [], "滑块相似度": 0.0,
            "缺口": [], "缺口相似度": 0.0,
            "滑缺距": 0, "消息": "缺少image参数或base64解码失败",
        }

    result = _recognize(image_bytes)

    if result is None:
        return {
            "滑块": [], "滑块相似度": 0.0,
            "缺口": [], "缺口相似度": 0.0,
            "滑缺距": 0, "消息": "不支持的图片格式",
        }

    return {
        "滑块": result["slider"],
        "滑块相似度": round(result["slider_confidence"], 4),
        "缺口": result["gap"],
        "缺口相似度": round(result["gap_confidence"], 4),
        "滑缺距": result["offset"],
        "消息": None,
    }


# ============================================================
# 启动入口
# ============================================================
if __name__ == "__main__":
    import uvicorn
    # HF Spaces 要求从环境变量读端口号，默认 7860
    # 本地开发时直接 python main.py 即可启动
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
