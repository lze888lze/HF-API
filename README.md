---
title: Slider Captcha API
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# 滑块验证码识别 API

基于 FastAPI + ONNX 的滑块验证码识别服务。

## 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | / | 健康检查 |
| GET | /health | 健康检查 |
| POST | /captcha | 文件上传识别 |
| POST | /captcha/base64 | Base64图片识别 |

## 调用示例

### 文件上传
```bash
curl -X POST https://your-space.hf.space/captcha \
  -F "file=@captcha.png"
```

### Base64（懒人精灵）
```lua
local http = require("http")
local json = require("json")
local base64 = require("base64")

local f = io.open("/sdcard/captcha.png", "rb")
local img = f:read("*a")
f:close()

local resp = http.post("https://your-domain.com/captcha/base64", {
    headers = {["Content-Type"] = "application/json"},
    body = '{"image":"' .. base64.encode(img) .. '"}'
})

local result = json.decode(resp.body)
-- result.box = [x1, y1, x2, y2]
-- result.confidence = 0.95
```
