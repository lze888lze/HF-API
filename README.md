---
title: Slider Captcha API
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# 滑块验证码识别 API

基于 FastAPI + ONNX 的滑块验证码识别服务，支持滑块位置识别、缺口位置识别、滑缺距计算、带框图片可视化上传，以及 IPv4/IPv6 离线归属地查询。

## 接口列表

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/` | 根路径健康检查 |
| `GET` | `/health` | 部署平台健康检查 |
| `GET` | `/ip?ip=8.8.8.8` | IP 归属地查询，自动识别 IPv4/IPv6 |
| `POST` | `/ip` | IP 归属地查询，JSON 传入 `{"ip":"8.8.8.8"}` |
| `POST` | `/slide` | 上传图片文件，只返回滑块位置 |
| `POST` | `/slide-base64` | 上传 base64 图片，只返回滑块位置 |
| `POST` | `/hole` | 上传图片文件，只返回缺口位置 |
| `POST` | `/hole-base64` | 上传 base64 图片，只返回缺口位置 |
| `POST` | `/puzzle` | 上传图片文件，返回滑块、缺口和滑缺距 |
| `POST` | `/puzzle-base64` | 上传 base64 图片，返回滑块、缺口和滑缺距 |
| `POST` | `/visualize` | 上传图片文件，返回识别结果和带框图片 URL |
| `POST` | `/visualize-base64` | 上传 base64 图片，返回识别结果和带框图片 URL |

## 基础地址

部署到 Hugging Face Spaces 后，把下面示例里的地址替换成你的 Space 地址：

```text
https://your-space.hf.space
```

## 健康检查

### 根路径

```bash
curl "https://your-space.hf.space/"
```

返回示例：

```json
{
  "Hello": "Captcha",
  "status": "running"
}
```

### health

```bash
curl "https://your-space.hf.space/health"
```

返回示例：

```json
{
  "status": "ok"
}
```

## IP 查询

接口会自动识别 IPv4 和 IPv6，并使用项目内的离线库查询：

```text
data/ip2region_v4.xdb
data/ip2region_v6.xdb
```

### GET 查询

```bash
curl "https://your-space.hf.space/ip?ip=8.8.8.8"
```

IPv6 示例：

```bash
curl "https://your-space.hf.space/ip?ip=240e:3b7:3272:d8d0:db09:c067:8d59:539e"
```

### POST 查询

```bash
curl -X POST "https://your-space.hf.space/ip" \
  -H "Content-Type: application/json" \
  -d '{"ip":"8.8.8.8"}'
```

### 返回示例

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

非法 IP 返回示例：

```json
{
  "ip": "not-an-ip",
  "版本": null,
  "归属地": "",
  "数据": {},
  "消息": "无效IP地址"
}
```

## 滑块位置识别

### 上传图片文件

```bash
curl -X POST "https://your-space.hf.space/slide" \
  -F "file=@captcha.png"
```

返回示例：

```json
{
  "滑块": [91, 1045, 248, 1200],
  "相似度": 0.9487,
  "消息": null
}
```

### 上传 base64

```bash
curl -X POST "https://your-space.hf.space/slide-base64" \
  -H "Content-Type: application/json" \
  -d '{"image":"base64字符串"}'
```

请求字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `image` | `string` | 图片 base64 字符串，支持纯 base64 或 `data:image/png;base64,` 前缀 |

## 缺口位置识别

### 上传图片文件

```bash
curl -X POST "https://your-space.hf.space/hole" \
  -F "file=@captcha.png"
```

返回示例：

```json
{
  "缺口": [803, 1044, 957, 1201],
  "相似度": 0.9474,
  "消息": null
}
```

### 上传 base64

```bash
curl -X POST "https://your-space.hf.space/hole-base64" \
  -H "Content-Type: application/json" \
  -d '{"image":"base64字符串"}'
```

## 滑块和缺口组合识别

### 上传图片文件

```bash
curl -X POST "https://your-space.hf.space/puzzle" \
  -F "file=@captcha.png"
```

返回示例：

```json
{
  "滑块": [91, 1045, 248, 1200],
  "滑块相似度": 0.9487,
  "缺口": [803, 1044, 957, 1201],
  "缺口相似度": 0.9474,
  "滑缺距": 712,
  "消息": null
}
```

`滑缺距` 的计算方式是：

```text
缺口 x1 - 滑块 x1
```

### 上传 base64

```bash
curl -X POST "https://your-space.hf.space/puzzle-base64" \
  -H "Content-Type: application/json" \
  -d '{"image":"base64字符串"}'
```

## 可视化识别

可视化接口会识别滑块和缺口，在图片上画框，然后把带框图片上传到 Cloudflare R2，最后返回图片 URL。

需要在 Hugging Face Spaces 配置这些环境变量：

| 环境变量 | 说明 |
|---|---|
| `R2_ENDPOINT` | Cloudflare R2 S3 兼容接口地址 |
| `R2_ACCESS_KEY_ID` | R2 Access Key ID |
| `R2_SECRET_ACCESS_KEY` | R2 Secret Access Key |
| `R2_BUCKET_NAME` | R2 Bucket 名称，默认 `lze` |
| `R2_PUBLIC_URL` | R2 公开访问域名 |

如果没有配置 R2，接口仍会返回识别坐标，但 `image_url` 会是 `null`。

### 上传图片文件

```bash
curl -X POST "https://your-space.hf.space/visualize" \
  -F "file=@captcha.png"
```

返回示例：

```json
{
  "滑块": [91, 1045, 248, 1200],
  "滑块相似度": 0.9487,
  "缺口": [803, 1044, 957, 1201],
  "缺口相似度": 0.9474,
  "滑缺距": 712,
  "image_url": "https://xxx.r2.dev/captcha_xxx.png",
  "消息": null
}
```

### 上传 base64

```bash
curl -X POST "https://your-space.hf.space/visualize-base64" \
  -H "Content-Type: application/json" \
  -d '{"image":"base64字符串"}'
```

## 错误返回

图片格式不支持时：

```json
{
  "消息": "不支持的图片格式"
}
```

base64 参数缺失或解码失败时：

```json
{
  "消息": "缺少image参数或base64解码失败"
}
```

IP 参数缺失时：

```json
{
  "消息": "缺少ip参数"
}
```

## 部署说明

项目使用 Docker SDK 部署到 Hugging Face Spaces，默认端口是：

```text
7860
```

运行时依赖在 `requirements.txt` 中声明。新增 IP 查询接口需要：

```text
py-ip2region
```

可视化上传 R2 需要：

```text
boto3
```
