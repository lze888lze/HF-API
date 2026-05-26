import gc
import os
from typing import List

import cv2
import numpy as np
from captcha_recognizer.slider import Slider
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    docs_url=None,
    redoc_url=None,
)

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 启动时只加载一次模型
slider = Slider()


@app.get("/")
def hello_captcha():
    return {"Hello": "Captcha", "status": "running"}


@app.get("/health")
def health():
    return {"status": "ok"}


class DetectionResult(BaseModel):
    box: List[int]  # [x1, y1, x2, y2]
    confidence: float
    message: str = None


@app.post("/captcha", response_model=DetectionResult)
async def captcha(file: UploadFile = File(...)):
    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return DetectionResult(box=[], confidence=0, message="不支持的图片")

    box, confidence = slider.identify(source=image)
    box = [int(x) for x in box]

    del image, nparr, contents
    gc.collect()

    return DetectionResult(box=box, confidence=confidence)


@app.post("/captcha/base64", response_model=DetectionResult)
async def captcha_base64(data: dict):
    """支持base64图片上传，用于懒人精灵调用"""
    import base64

    image_b64 = data.get("image", "")
    if not image_b64:
        return DetectionResult(box=[], confidence=0, message="缺少image参数")

    # 去掉base64头（如果有）
    if "," in image_b64:
        image_b64 = image_b64.split(",")[1]

    try:
        image_bytes = base64.b64decode(image_b64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        return DetectionResult(box=[], confidence=0, message=f"base64解码失败: {str(e)}")

    if image is None:
        return DetectionResult(box=[], confidence=0, message="不支持的图片")

    box, confidence = slider.identify(source=image)
    box = [int(x) for x in box]

    del image, nparr, image_bytes
    gc.collect()

    return DetectionResult(box=box, confidence=confidence)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
