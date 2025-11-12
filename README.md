## 云盘系统 - 智能存储与处理平台

### 功能特性

- ✅ **文件上传、下载与分享** - 支持图片、视频等多种格式
- ✅ **实时生成图像缩略图** - 自动为图片生成缩略图
- ✅ **视频转码服务** - 自动转码为通用格式，支持转码并下载
- ✅ **内容审核** - 敏感信息筛查（基于启发式算法）
- ✅ **图像内容分类** - 使用深度学习模型自动分类（可选）
- ✅ **图像文字识别（OCR）** - 支持中英文识别（可选）
- ✅ **语音识别（ASR）** - 视频语音转文字（可选）
- ✅ **智能检索** - 全文搜索文件名和提取的文字内容
- ✅ **美观的中文界面** - 现代化的Web界面，支持拖拽上传

### 快速开始

#### 1. 安装依赖

```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境（Windows）
.venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

#### 2. 启动服务

```bash
uvicorn app.main:app --reload --port 8000
```

#### 3. 访问界面

打开浏览器访问：**http://localhost:8000**

- **首页** (`/`) - 上传文件、搜索、查看最近文件
- **我的文件** (`/files`) - 管理所有文件，执行OCR、语音识别等操作

### 可选功能配置

如需使用以下高级功能，需要额外安装依赖：

#### OCR（文字识别）
```bash
# 方式1：使用pytesseract（需要先安装系统Tesseract）
pip install pytesseract

# 方式2：使用EasyOCR（纯Python，无需系统依赖）
pip install easyocr
```

#### 图像分类
```bash
pip install torch torchvision
```

#### 语音识别
```bash
# 方式1：使用Vosk（需要下载模型）
pip install vosk
# 将Vosk模型放到 app/models/vosk-model/ 目录

# 方式2：使用Whisper（OpenAI）
pip install openai-whisper
```

### API接口

- `POST /upload` - 上传文件
- `GET /download/{asset_id}` - 下载文件
- `POST /share/{asset_id}` - 生成分享链接
- `GET /s/{token}` - 通过分享链接下载
- `GET /search?q=关键词` - 搜索文件
- `POST /ocr/{asset_id}` - 图片OCR识别
- `POST /asr/{asset_id}` - 视频语音识别
- `GET /api/files` - 获取文件列表（JSON API）

### 数据存储

- 文件存储：`storage/original/` - 原始文件
- 衍生文件：`storage/derived/` - 缩略图、转码视频等
- 数据库：`cloud_drive.db` - SQLite数据库
- 搜索索引：`index/` - Whoosh全文索引

### 界面预览

- 🎨 **现代化设计** - 使用Tailwind CSS，响应式布局
- 📱 **移动端友好** - 自适应不同屏幕尺寸
- 🚀 **流畅交互** - 拖拽上传、实时搜索、进度显示
- 🌈 **渐变背景** - 紫色渐变主题，美观大方

### 技术栈

- **后端**: FastAPI + SQLModel + SQLite
- **前端**: Jinja2模板 + Tailwind CSS + Font Awesome
- **存储**: 本地文件系统
- **搜索**: Whoosh全文索引
- **图像处理**: Pillow
- **视频处理**: ffmpeg-python




