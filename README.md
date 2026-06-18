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
| GET | /ip?ip=8.8.8.8 | IP 归属地查询，自动识别 IPv4/IPv6 |
| POST | /ip | IP 归属地查询，JSON 传入 `{"ip":"8.8.8.8"}` |
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

### IP 查询

GET 调用：
```bash
curl "https://your-space.hf.space/ip?ip=8.8.8.8"
```

POST 调用：
```bash
curl -X POST "https://your-space.hf.space/ip" \
  -H "Content-Type: application/json" \
  -d '{"ip":"240e:3b7:3272:d8d0:db09:c067:8d59:539e"}'
```

返回示例：
```json
{
  "ip": "8.8.8.8",
  "版本": "IPv4",
  "归属地": "United States|California|0|Google LLC|US",
  "数据": {
    "国家": "United States",
    "省份/州": "California",
    "城市": null,
    "运营商": "Google LLC",
    "国家代码": "US"
  },
  "消息": null
}
```
