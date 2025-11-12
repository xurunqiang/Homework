from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query, Request, Header, Depends
from fastapi.responses import FileResponse, HTMLResponse, Response
from starlette.templating import Jinja2Templates
from pydantic import BaseModel
from sqlmodel import SQLModel, Field, Session, create_engine, select
from typing import Optional, List
from pathlib import Path
import shutil
import uuid
import datetime as dt
import warnings
import hashlib
import secrets

# 过滤PyTorch的pin_memory警告（当没有GPU时）
warnings.filterwarnings("ignore", message=".*pin_memory.*", category=UserWarning, module="torch.utils.data")

# Local services
from .services.thumbnails import generate_thumbnail_sync
from .services.transcode import transcode_video_bg
from .services.moderation import moderate_content
from .services.classify import classify_image
from .services.ocr_asr import run_ocr, run_asr
from .services.search import search_index, add_to_index, update_doc_text, remove_from_index

BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = BASE_DIR / "storage"
ORIGINAL_DIR = STORAGE_DIR / "original"
DERIVED_DIR = STORAGE_DIR / "derived"
TEMPLATES_DIR = BASE_DIR / "templates"

for p in [STORAGE_DIR, ORIGINAL_DIR, DERIVED_DIR, TEMPLATES_DIR]:
    p.mkdir(parents=True, exist_ok=True)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


class Asset(SQLModel, table=True):
    id: str = Field(primary_key=True, default_factory=lambda: uuid.uuid4().hex)
    filename: str
    content_type: str
    path: str
    kind: str  # image | video | other
    size_bytes: int
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.utcnow())
    shared_token: Optional[str] = None
    ocr_text: Optional[str] = None
    transcript_text: Optional[str] = None
    moderation_flags: Optional[str] = None
    labels: Optional[str] = None  # comma-separated labels


class User(SQLModel, table=True):
    id: str = Field(primary_key=True, default_factory=lambda: uuid.uuid4().hex)
    username: str = Field(index=True, unique=True)
    password_hash: str
    password_salt: str
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.utcnow())


class SessionToken(SQLModel, table=True):
    token: str = Field(primary_key=True, default_factory=lambda: secrets.token_hex(32))
    user_id: str = Field(index=True)
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.utcnow())
    expires_at: Optional[dt.datetime] = None


def _hash_password(password: str, salt: str) -> str:
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), 100_000)
    return dk.hex()


def _create_password_hash(password: str) -> tuple[str, str]:
    salt = secrets.token_hex(16)
    return _hash_password(password, salt), salt


DATABASE_PATH = BASE_DIR / "cloud_drive.db"
engine = create_engine(f"sqlite:///{DATABASE_PATH}")


def init_db() -> None:
    SQLModel.metadata.create_all(engine)
    search_index.initialize(BASE_DIR / "index")


app = FastAPI(title="云盘系统", description="智能云存储与处理平台")


@app.on_event("startup")
def on_startup() -> None:
    init_db()


def _detect_kind(content_type: str) -> str:
    if content_type.startswith("image/"):
        return "image"
    if content_type.startswith("video/"):
        return "video"
    return "other"


class ShareResponse(BaseModel):
    url: str


# 简单鉴权依赖：从Authorization头解析Bearer token并返回当前用户
def get_current_user(authorization: Optional[str] = Header(None)) -> Optional[User]:
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    token = authorization.split(" ", 1)[1].strip()
    with Session(engine) as session:
        st = session.exec(select(SessionToken).where(SessionToken.token == token)).first()
        if not st:
            return None
        if st.expires_at and st.expires_at < dt.datetime.utcnow():
            # 过期则清理
            session.delete(st)
            session.commit()
            return None
        user = session.get(User, st.user_id)
        return user


# 认证与用户相关模型
class RegisterRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class AuthResponse(BaseModel):
    token: str
    username: str


@app.post("/auth/register")
def register_user(req: RegisterRequest):
    username = req.username.strip()
    if not username or not req.password:
        raise HTTPException(status_code=400, detail="用户名与密码不能为空")
    if len(username) < 3 or len(req.password) < 6:
        raise HTTPException(status_code=400, detail="用户名或密码长度不符合要求")
    with Session(engine) as session:
        existed = session.exec(select(User).where(User.username == username)).first()
        if existed:
            raise HTTPException(status_code=409, detail="用户名已存在")
        pwd_hash, salt = _create_password_hash(req.password)
        user = User(username=username, password_hash=pwd_hash, password_salt=salt)
        session.add(user)
        session.commit()
        session.refresh(user)
        # 自动登录：创建会话token
        token = secrets.token_hex(32)
        session_token = SessionToken(token=token, user_id=user.id, expires_at=dt.datetime.utcnow() + dt.timedelta(days=7))
        session.add(session_token)
        session.commit()
        return AuthResponse(token=token, username=user.username)


@app.post("/auth/login")
def login(req: LoginRequest):
    with Session(engine) as session:
        user = session.exec(select(User).where(User.username == req.username.strip())).first()
        if not user:
            raise HTTPException(status_code=401, detail="用户名或密码错误")
        calc = _hash_password(req.password, user.password_salt)
        if calc != user.password_hash:
            raise HTTPException(status_code=401, detail="用户名或密码错误")
        # 签发新token（可并存多个设备）
        token = secrets.token_hex(32)
        session_token = SessionToken(token=token, user_id=user.id, expires_at=dt.datetime.utcnow() + dt.timedelta(days=7))
        session.add(session_token)
        session.commit()
        return AuthResponse(token=token, username=user.username)


@app.get("/auth/me")
def auth_me(current_user: Optional[User] = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(status_code=401, detail="未登录")
    return {"username": current_user.username, "id": current_user.id, "created_at": current_user.created_at}


@app.post("/upload", response_model=Asset)
async def upload(file: UploadFile = File(...), background: BackgroundTasks = None):
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    dst = ORIGINAL_DIR / f"{uuid.uuid4().hex}_{Path(file.filename).name}"
    with dst.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    kind = _detect_kind(file.content_type or "application/octet-stream")
    size = dst.stat().st_size

    # 上传时进行内容审核（同步，有问题的不允许上传）
    moderation_passed = True
    moderation_flags = []
    
    if kind == "image":
        # 图片直接审核（包括OCR文本审核）
        try:
            # 先进行图像审核
            flags = moderate_content(dst)
            
            # 如果图像审核通过，尝试OCR并审核文本（快速OCR，仅用于审核）
            if not flags:
                try:
                    ocr_text = run_ocr(dst)
                    if ocr_text:
                        # 使用OCR文本进行文本审核
                        text_flags = moderate_content(dst, ocr_text=ocr_text)
                        flags.extend(text_flags)
                except Exception:
                    # OCR失败不影响上传
                    pass
            
            if flags:
                moderation_flags = flags
                moderation_passed = False
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"内容审核失败: {e}")
            # 审核失败时允许上传（避免误判）
    
    elif kind == "video":
        # 视频提取第一帧进行审核
        try:
            import subprocess
            import tempfile
            # 提取视频第一帧
            frame_path = DERIVED_DIR / f"{dst.stem}_frame.jpg"
            result = subprocess.run(
                [
                    "ffmpeg", "-i", str(dst), "-vframes", "1", "-q:v", "2",
                    str(frame_path), "-y"
                ],
                capture_output=True,
                timeout=10
            )
            if result.returncode == 0 and frame_path.exists():
                flags = moderate_content(frame_path)
                if flags:
                    moderation_flags = flags
                    moderation_passed = False
                # 删除临时帧文件
                try:
                    frame_path.unlink()
                except Exception:
                    pass
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"视频内容审核失败: {e}")
            # 审核失败时允许上传（避免误判）
    
    # 如果检测到敏感内容，删除文件并拒绝上传
    if not moderation_passed:
        try:
            dst.unlink()
        except Exception:
            pass
        flags_str = ", ".join(moderation_flags)
        raise HTTPException(
            status_code=403,
            detail=f"文件包含敏感内容，不允许上传。检测到的标记: {flags_str}"
        )

    asset = Asset(
        filename=Path(file.filename).name,
        content_type=file.content_type or "application/octet-stream",
        path=str(dst),
        kind=kind,
        size_bytes=size,
    )
    with Session(engine) as session:
        session.add(asset)
        session.commit()
        session.refresh(asset)

    # index filename for search
    add_to_index(asset.id, asset.filename)

    # lightweight auto-jobs
    if kind == "image":
        try:
            thumb_path = generate_thumbnail_sync(dst, DERIVED_DIR)
            add_to_index(asset.id, f"thumbnail:{thumb_path.name}")
        except Exception:
            pass
        # classification and OCR in background to keep API snappy
        if background:
            background.add_task(_post_image_processing, str(dst), asset.id)
    elif kind == "video" and background:
        # 视频自动转码为720p mp4（后台任务）
        background.add_task(transcode_video_bg, str(dst), str(DERIVED_DIR), "mp4", "720p")

    return asset


def _post_image_processing(path: str, asset_id: str) -> None:
    flags = moderate_content(Path(path))
    labels = ",".join(classify_image(Path(path)) or [])
    text = run_ocr(Path(path)) or None
    with Session(engine) as session:
        asset = session.exec(select(Asset).where(Asset.id == asset_id)).one()
        asset.moderation_flags = ",".join(flags) if flags else None
        asset.labels = labels or None
        asset.ocr_text = text
        session.add(asset)
        session.commit()
    # update search index with derived text/labels
    update_doc_text(asset_id, " ".join([labels or "", text or ""]).strip())


@app.get("/download/{asset_id}")
def download(asset_id: str):
    with Session(engine) as session:
        asset = session.exec(select(Asset).where(Asset.id == asset_id)).first()
        if not asset:
            raise HTTPException(status_code=404, detail="文件未找到")
        return FileResponse(asset.path, filename=asset.filename, media_type=asset.content_type)


@app.post("/share/{asset_id}", response_model=ShareResponse)
def share(asset_id: str):
    token = uuid.uuid4().hex[:12]
    with Session(engine) as session:
        asset = session.exec(select(Asset).where(Asset.id == asset_id)).first()
        if not asset:
            raise HTTPException(status_code=404, detail="文件未找到")
        asset.shared_token = token
        session.add(asset)
        session.commit()
    return ShareResponse(url=f"/s/{token}")


@app.get("/s/{token}")
def shared_download(token: str):
    with Session(engine) as session:
        asset = session.exec(select(Asset).where(Asset.shared_token == token)).first()
        if not asset:
            raise HTTPException(status_code=404, detail="分享链接无效")
        return FileResponse(asset.path, filename=asset.filename, media_type=asset.content_type)


class SearchResult(BaseModel):
    id: str
    filename: str
    snippet: Optional[str] = None


@app.get("/search", response_model=List[SearchResult])
def search(q: str):
    ids_to_snippet = search_index.search(q)
    results: List[SearchResult] = []
    with Session(engine) as session:
        for asset_id, snippet in ids_to_snippet.items():
            asset = session.get(Asset, asset_id)
            if asset:
                results.append(SearchResult(id=asset.id, filename=asset.filename, snippet=snippet))
    return results


@app.post("/ocr/{asset_id}")
def ocr(asset_id: str):
    with Session(engine) as session:
        asset = session.get(Asset, asset_id)
        if not asset:
            raise HTTPException(status_code=404, detail="文件未找到")
        if asset.kind != "image":
            raise HTTPException(status_code=400, detail="OCR仅支持图片文件")
        text = run_ocr(Path(asset.path))
        asset.ocr_text = text or None
        session.add(asset)
        session.commit()
    if text:
        update_doc_text(asset_id, text)
    return {"text": text}


@app.post("/asr/{asset_id}")
def asr(asset_id: str, language: Optional[str] = Query(None, description="语言代码: zh(中文), en(英文), auto(自动检测)")):
    import logging
    logger = logging.getLogger(__name__)
    
    with Session(engine) as session:
        asset = session.get(Asset, asset_id)
        if not asset:
            raise HTTPException(status_code=404, detail="文件未找到")
        if asset.kind != "video":
            raise HTTPException(status_code=400, detail="语音识别仅支持视频文件")
        
        # 处理语言参数
        lang_code = None
        if language:
            lang_map = {"zh": "zh", "chinese": "zh", "中文": "zh", 
                       "en": "en", "english": "en", "英文": "en",
                       "auto": None, "自动": None}
            lang_code = lang_map.get(language.lower(), language)
        
        try:
            logger.info(f"开始语音识别: {asset.filename}, 语言: {lang_code or '自动检测'}")
            text = run_asr(Path(asset.path), language=lang_code)
            if text:
                asset.transcript_text = text
                session.add(asset)
                session.commit()
                update_doc_text(asset_id, text)
                logger.info(f"语音识别成功: {len(text)} 字符")
                return {"text": text}
            else:
                raise HTTPException(status_code=500, detail="语音识别未返回任何文本，请检查视频是否包含音频")
        except RuntimeError as e:
            logger.error(f"语音识别错误: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            logger.error(f"语音识别异常: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"语音识别失败: {str(e)}")


# 前端路由
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/files", response_class=HTMLResponse)
async def files_page(request: Request):
    return templates.TemplateResponse("files.html", {"request": request})


# Favicon路由
@app.get("/favicon.ico")
async def favicon():
    # 返回一个简单的SVG图标作为favicon
    svg_icon = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
        <defs>
            <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:#3b82f6;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#06b6d4;stop-opacity:1" />
            </linearGradient>
        </defs>
        <rect width="100" height="100" rx="20" fill="url(#grad)"/>
        <path d="M30 30 L50 50 L70 30 M30 70 L50 50 L70 70" stroke="white" stroke-width="8" fill="none" stroke-linecap="round"/>
    </svg>"""
    return Response(content=svg_icon, media_type="image/svg+xml")


# API: 获取文件列表
class AssetResponse(BaseModel):
    id: str
    filename: str
    content_type: str
    kind: str
    size_bytes: int
    created_at: dt.datetime
    shared_token: Optional[str] = None
    ocr_text: Optional[str] = None
    transcript_text: Optional[str] = None
    moderation_flags: Optional[str] = None
    labels: Optional[str] = None


@app.get("/api/files", response_model=List[AssetResponse])
def list_files(limit: Optional[int] = Query(None), offset: int = Query(0, ge=0)):
    with Session(engine) as session:
        query = select(Asset).order_by(Asset.created_at.desc())
        if limit:
            query = query.limit(limit).offset(offset)
        assets = session.exec(query).all()
        return [AssetResponse(**asset.model_dump()) for asset in assets]


@app.get("/api/files/{asset_id}", response_model=AssetResponse)
def get_file(asset_id: str):
    with Session(engine) as session:
        asset = session.get(Asset, asset_id)
        if not asset:
            raise HTTPException(status_code=404, detail="文件未找到")
        return AssetResponse(**asset.model_dump())


@app.delete("/api/files/{asset_id}")
def delete_file(asset_id: str):
    import logging
    logger = logging.getLogger(__name__)
    
    with Session(engine) as session:
        asset = session.get(Asset, asset_id)
        if not asset:
            raise HTTPException(status_code=404, detail="文件未找到")
        
        file_path = Path(asset.path)
        deleted_files = []
        errors = []
        
        # 删除物理文件
        if file_path.exists():
            try:
                file_path.unlink()
                deleted_files.append(f"原始文件: {file_path}")
            except Exception as e:
                errors.append(f"删除原始文件失败: {str(e)}")
                logger.error(f"删除文件失败 {file_path}: {e}")
        
        # 删除缩略图（如果存在）
        thumb_path = DERIVED_DIR / f"thumb_{file_path.stem}.jpg"
        if thumb_path.exists():
            try:
                thumb_path.unlink()
                deleted_files.append(f"缩略图: {thumb_path.name}")
            except Exception as e:
                errors.append(f"删除缩略图失败: {str(e)}")
        
        # 删除转码视频（如果存在）- 支持所有格式和分辨率
        for pattern in [f"{file_path.stem}_*p.*", f"{file_path.stem}_*.*"]:
            for derived_file in DERIVED_DIR.glob(pattern):
                # 确保是转码文件（包含分辨率标识）
                if any(res in derived_file.name for res in ['360p', '480p', '720p', '1080p']):
                    try:
                        derived_file.unlink()
                        deleted_files.append(f"转码文件: {derived_file.name}")
                    except Exception as e:
                        errors.append(f"删除转码文件失败: {str(e)}")
        
        # 从搜索索引中删除
        try:
            remove_from_index(asset_id)
            deleted_files.append("搜索索引")
        except Exception as e:
            errors.append(f"删除搜索索引失败: {str(e)}")
            logger.error(f"删除搜索索引失败 {asset_id}: {e}")
        
        # 从数据库删除
        try:
            session.delete(asset)
            session.commit()
            deleted_files.append("数据库记录")
        except Exception as e:
            session.rollback()
            errors.append(f"删除数据库记录失败: {str(e)}")
            logger.error(f"删除数据库记录失败 {asset_id}: {e}")
            raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")
    
    return {
        "message": "文件删除成功",
        "id": asset_id,
        "deleted": deleted_files,
        "warnings": errors if errors else None
    }


# 视频转码API
@app.post("/transcode/{asset_id}")
def transcode_video(asset_id: str, format: str = Query("mp4", description="格式: mp4, avi, mov, webm, mkv"), 
                    resolution: str = Query("720p", description="分辨率: 360p, 480p, 720p, 1080p")):
    from .services.transcode import transcode_video_bg
    import logging
    logger = logging.getLogger(__name__)
    
    with Session(engine) as session:
        asset = session.get(Asset, asset_id)
        if not asset:
            raise HTTPException(status_code=404, detail="文件未找到")
        if asset.kind != "video":
            raise HTTPException(status_code=400, detail="转码仅支持视频文件")
        
        try:
            logger.info(f"开始转码: {asset.filename} -> {format} {resolution}")
            result_path = transcode_video_bg(asset.path, str(DERIVED_DIR), format, resolution)
            
            if result_path and result_path.exists():
                logger.info(f"转码成功: {result_path}")
                return {
                    "message": "转码成功",
                    "file": str(result_path),
                    "download_url": f"/download/transcoded/{asset_id}?format={format}&resolution={resolution}"
                }
            else:
                raise HTTPException(status_code=500, detail="转码失败，请检查ffmpeg是否正确安装")
        except Exception as e:
            logger.error(f"转码异常: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"转码失败: {str(e)}")


@app.get("/download/transcoded/{asset_id}")
def download_transcoded(asset_id: str, format: str = Query("mp4"), resolution: str = Query("720p")):
    with Session(engine) as session:
        asset = session.get(Asset, asset_id)
        if not asset:
            raise HTTPException(status_code=404, detail="文件未找到")
        if asset.kind != "video":
            raise HTTPException(status_code=400, detail="转码仅支持视频文件")
        
        # 查找转码后的文件
        src_path = Path(asset.path)
        transcoded_path = DERIVED_DIR / f"{src_path.stem}_{resolution}.{format}"
        
        if not transcoded_path.exists():
            raise HTTPException(status_code=404, detail="转码文件不存在，请先进行转码")
        
        return FileResponse(
            transcoded_path, 
            filename=f"{src_path.stem}_{resolution}.{format}",
            media_type=f"video/{format}"
        )




