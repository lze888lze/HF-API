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
  POST /visualize  - 上传图片文件，返回坐标+带框图片URL
  POST /visualize-base64 - 上传base64图片，返回坐标+带框图片URL
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
  /visualize、/visualize-base64：
    {
      "滑块": [x1, y1, x2, y2],
      "滑块相似度": 0.92,
      "缺口": [x1, y1, x2, y2],
      "缺口相似度": 0.88,
      "滑缺距": 162,
      "image_url": "https://xxx.r2.dev/xxx.png",  // 带框图片URL
      "消息": null
    }
"""

import ipaddress  # IP 地址格式校验和 IPv4/IPv6 识别
import json      # 解析 JSON 请求体
import os       # 读取环境变量（端口号）
import time     # 时间戳生成文件名
import urllib.parse  # 解析 ip=xxx 表单字符串
import uuid     # 唯一ID
from pathlib import Path
from typing import Optional

import boto3   # AWS S3 SDK（兼容 R2）
import cv2      # OpenCV，图片解码
import numpy as np  # 数组操作

from captcha_recognizer.slider import Slider  # 核心：滑块识别模型
from fastapi import FastAPI, File, Query, Request, UploadFile  # Web 框架
from fastapi.middleware.cors import CORSMiddleware  # 跨域支持
import ip2region.searcher as ip2region_xdb  # ip2region xdb 查询器
import ip2region.util as ip2region_util  # ip2region 工具方法


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
# IP 归属地库加载（服务启动时验证并缓存 VectorIndex）
# ============================================================
# xdb 文件放在项目 data 目录，Docker 部署时会随 COPY . . 一起带到 HF Spaces
BASE_DIR = Path(__file__).resolve().parent
IPV4_XDB_PATH = str(BASE_DIR / "data" / "ip2region_v4.xdb")
IPV6_XDB_PATH = str(BASE_DIR / "data" / "ip2region_v6.xdb")

ip2region_error = None
ipv4_vector_index = None
ipv6_vector_index = None

try:
    # 启动时验证 xdb 文件可用，避免运行中才发现库版本或文件不匹配
    ip2region_util.verify_from_file(IPV4_XDB_PATH)
    ip2region_util.verify_from_file(IPV6_XDB_PATH)

    # 缓存 VectorIndex：比每次纯文件查询少一次固定 IO，同时比全量加载 xdb 更省内存
    ipv4_vector_index = ip2region_util.load_vector_index_from_file(IPV4_XDB_PATH)
    ipv6_vector_index = ip2region_util.load_vector_index_from_file(IPV6_XDB_PATH)
except Exception as e:
    ip2region_error = str(e)


# ============================================================
# Cloudflare R2 配置
# ============================================================

R2_CONFIG = {
    "endpoint": os.environ.get("R2_ENDPOINT"),  # 如 https://xxx.r2.cloudflarestorage.com
    "access_key": os.environ.get("R2_ACCESS_KEY_ID"),  # Access Key ID
    "secret_key": os.environ.get("R2_SECRET_ACCESS_KEY"),  # Secret Access Key
    "bucket_name": os.environ.get("R2_BUCKET_NAME", "lze"),  # Bucket 名称
    "public_url": os.environ.get("R2_PUBLIC_URL", "https://pub-7812368cee8345339c5c95b9768ad994.r2.dev"),  # 公开访问域名
}

# 初始化 R2 客户端（如果配置了）
r2_client = None
if R2_CONFIG["access_key"] and R2_CONFIG["secret_key"]:
    r2_client = boto3.client(
        "s3",
        endpoint_url=R2_CONFIG["endpoint"],
        aws_access_key_id=R2_CONFIG["access_key"],
        aws_secret_access_key=R2_CONFIG["secret_key"],
    )


def upload_to_r2(image_bytes: bytes, filename: str) -> Optional[str]:
    """
    上传图片到 Cloudflare R2，返回公开访问 URL
    
    参数：
      image_bytes: 图片字节数据
      filename: 文件名
    
    返回：
      成功 → 公开访问 URL
      失败 → None
    """
    if r2_client is None:
        return None
    
    try:
        r2_client.put_object(
            Bucket=R2_CONFIG["bucket_name"],
            Key=filename,
            Body=image_bytes,
            ContentType="image/png",
        )
        
        # 生成公开 URL
        if R2_CONFIG["public_url"]:
            # 自定义域名
            return f"{R2_CONFIG['public_url'].rstrip('/')}/{filename}"
        else:
            # 使用 R2.dev 默认域名（需要先在 R2 设置公开访问）
            return f"https://{R2_CONFIG['bucket_name']}.{R2_CONFIG['endpoint'].replace('https://', '')}/{filename}"
    except Exception as e:
        print(f"R2 上传失败: {e}")
        return None


def generate_image_filename() -> str:
    """生成唯一的图片文件名"""
    timestamp = int(time.time() * 1000)
    unique_id = uuid.uuid4().hex[:8]
    return f"captcha_{timestamp}_{unique_id}.png"


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
# IP 查询接口：/ip
# ============================================================

def _parse_region(region: str) -> dict:
    """
    ip2region 原始结果格式一般为：
      国家|省份/州|城市|运营商|国家代码

    这里保留原始字符串，同时拆成更方便脚本读取的字段。
    """
    parts = (region or "").split("|")
    parts = (parts + [""] * 5)[:5]

    def clean(value: str):
        return None if value in ("", "0") else value

    return {
        "国家": clean(parts[0]),
        "省份/州": clean(parts[1]),
        "城市": clean(parts[2]),
        "运营商": clean(parts[3]),
        "国家代码": clean(parts[4]),
    }


def _lookup_ip(ip: str) -> dict:
    """识别 IPv4/IPv6，并使用对应 xdb 离线库查询归属地"""
    ip_text = (ip or "").strip()
    if not ip_text:
        return {"ip": ip, "版本": None, "归属地": "", "数据": {}, "消息": "缺少ip参数"}

    try:
        ip_obj = ipaddress.ip_address(ip_text)
    except ValueError:
        return {"ip": ip_text, "版本": None, "归属地": "", "数据": {}, "消息": "无效IP地址"}

    if ip2region_error:
        return {
            "ip": ip_text,
            "版本": f"IPv{ip_obj.version}",
            "归属地": "",
            "数据": {},
            "消息": f"ip2region初始化失败：{ip2region_error}",
        }

    searcher = None
    try:
        if ip_obj.version == 4:
            searcher = ip2region_xdb.new_with_vector_index(
                ip2region_util.IPv4,
                IPV4_XDB_PATH,
                ipv4_vector_index,
            )
        else:
            searcher = ip2region_xdb.new_with_vector_index(
                ip2region_util.IPv6,
                IPV6_XDB_PATH,
                ipv6_vector_index,
            )

        region = searcher.search(ip_text) or ""
        return {
            "ip": ip_text,
            "版本": f"IPv{ip_obj.version}",
            "归属地": region,
            "数据": _parse_region(region),
            "消息": None,
        }
    except Exception as e:
        return {
            "ip": ip_text,
            "版本": f"IPv{ip_obj.version}",
            "归属地": "",
            "数据": {},
            "消息": f"查询失败：{str(e)}",
        }
    finally:
        if searcher is not None:
            searcher.close()


@app.get("/ip")
def ip_lookup(ip: str = Query(..., description="要查询的 IPv4 或 IPv6 地址")):
    """
    GET 查询 IP 归属地

    用法：
      GET /ip?ip=8.8.8.8
      GET /ip?ip=240e:3b7:3272:d8d0:db09:c067:8d59:539e
    """
    return _lookup_ip(ip)


@app.post("/ip")
async def ip_lookup_post(request: Request):
    """
    POST 查询 IP 归属地

    支持多种请求格式，方便不同客户端调用：
      {"ip": "8.8.8.8"}
      ip=8.8.8.8
      8.8.8.8
    """
    body = (await request.body()).decode("utf-8", errors="ignore").strip()
    ip = ""

    if body:
        # JSON：{"ip":"8.8.8.8"}
        try:
            data = json.loads(body)
            if isinstance(data, dict):
                ip = data.get("ip", "")
        except Exception:
            pass

        # 表单字符串：ip=8.8.8.8
        if not ip and "=" in body:
            form_data = urllib.parse.parse_qs(body)
            ip = (form_data.get("ip") or [""])[0]

        # 纯文本：8.8.8.8
        if not ip:
            ip = body

    return _lookup_ip(ip)


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
# 接口7 & 8：/visualize、/visualize-base64 — 返回带框图片URL
# ============================================================

def _draw_box_and_upload(image_bytes: bytes) -> tuple:
    """
    识别滑块并画框，上传到 R2，返回 (结果dict, 图片URL)
    
    返回：
      (识别结果dict, image_url) 或 (None, None) 如果失败
    """
    # 字节流 → numpy 数组 → OpenCV 图片
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return None, None

    # 调用模型获取识别结果
    result = slider_model.identify_both(source=image)

    # 绘制检测框
    output = image.copy()
    
    # 绘制滑块框（绿色）
    if result["slider"]:
        x1, y1, x2, y2 = result["slider"]
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(output, f"滑块:{result['slider_confidence']:.2f}", 
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # 绘制缺口框（红色）
    if result["gap"]:
        x1, y1, x2, y2 = result["gap"]
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(output, f"缺口:{result['gap_confidence']:.2f}", 
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # 绘制滑缺距文字
    if result["slider"] and result["gap"]:
        slider_x = result["slider"][0]
        gap_x = result["gap"][0]
        mid_y = (result["slider"][1] + result["gap"][1]) // 2
        cv2.putText(output, f"距离:{result['offset']}px", 
                    ((slider_x + gap_x) // 2, mid_y - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # 编码为 PNG
    _, buffer = cv2.imencode('.png', output)
    output_bytes = buffer.tobytes()

    # 上传到 R2
    filename = generate_image_filename()
    image_url = upload_to_r2(output_bytes, filename)

    return result, image_url


@app.post("/visualize")
async def puzzle_visualize(file: UploadFile = File(...)):
    """
    上传图片文件，返回滑块+缺口+滑缺距+带框图片URL
    用法：POST /puzzle-visualize，body 里传图片文件（form-data）
    返回示例：
    {
      "滑块": [91, 1045, 248, 1200],
      "滑块相似度": 0.9487,
      "缺口": [803, 1044, 957, 1201],
      "缺口相似度": 0.9474,
      "滑缺距": 712,
      "image_url": "https://xxx.r2.dev/captcha_xxx.png",
      "消息": null
    }
    """
    contents = await file.read()
    result, image_url = _draw_box_and_upload(contents)

    if result is None:
        return {
            "滑块": [], "滑块相似度": 0.0,
            "缺口": [], "缺口相似度": 0.0,
            "滑缺距": 0, 
            "image_url": None,
            "消息": "不支持的图片格式",
        }

    return {
        "滑块": result["slider"],
        "滑块相似度": round(result["slider_confidence"], 4),
        "缺口": result["gap"],
        "缺口相似度": round(result["gap_confidence"], 4),
        "滑缺距": result["offset"],
        "image_url": image_url,
        "消息": None,
    }


@app.post("/visualize-base64")
async def puzzle_visualize_base64(data: dict):
    """
    上传 base64 图片，返回滑块+缺口+滑缺距+带框图片URL
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
            "滑缺距": 0, 
            "image_url": None,
            "消息": "缺少image参数或base64解码失败",
        }

    result, image_url = _draw_box_and_upload(image_bytes)

    if result is None:
        return {
            "滑块": [], "滑块相似度": 0.0,
            "缺口": [], "缺口相似度": 0.0,
            "滑缺距": 0, 
            "image_url": None,
            "消息": "不支持的图片格式",
        }

    return {
        "滑块": result["slider"],
        "滑块相似度": round(result["slider_confidence"], 4),
        "缺口": result["gap"],
        "缺口相似度": round(result["gap_confidence"], 4),
        "滑缺距": result["offset"],
        "image_url": image_url,
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
