from pathlib import Path
import ffmpeg
from typing import Optional


def transcode_video_bg(src_path: str, out_dir: str, target_ext: str = "mp4", resolution: str = "720p") -> Optional[Path]:
    """转码视频到指定格式和分辨率
    
    Args:
        src_path: 源视频路径
        out_dir: 输出目录
        target_ext: 目标格式 (mp4, avi, mov, webm, mkv)
        resolution: 分辨率 (720p, 1080p, 480p, 360p)
    
    Returns:
        转码后的文件路径，失败返回None
    """
    src = Path(src_path)
    out_directory = Path(out_dir)
    out_directory.mkdir(parents=True, exist_ok=True)
    
    # 分辨率映射
    res_map = {
        "360p": 360,
        "480p": 480,
        "720p": 720,
        "1080p": 1080
    }
    height = res_map.get(resolution.lower(), 720)
    
    # 根据格式选择编码器
    vcodec_map = {
        "mp4": "libx264",
        "webm": "libvpx-vp9",
        "avi": "libx264",
        "mov": "libx264",
        "mkv": "libx264"
    }
    acodec_map = {
        "mp4": "aac",
        "webm": "libopus",
        "avi": "aac",
        "mov": "aac",
        "mkv": "aac"
    }
    
    vcodec = vcodec_map.get(target_ext.lower(), "libx264")
    acodec = acodec_map.get(target_ext.lower(), "aac")
    
    dst = out_directory / f"{src.stem}_{resolution}.{target_ext}"
    
    try:
        (
            ffmpeg
            .input(str(src))
            .filter("scale", -2, height)
            .output(str(dst), vcodec=vcodec, acodec=acodec, preset="veryfast", crf=23)
            .overwrite_output()
            .run(quiet=True)
        )
        return dst
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"转码失败: {e}")
        return None




