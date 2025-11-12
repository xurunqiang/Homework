from pathlib import Path
from PIL import Image


def generate_thumbnail_sync(src_path: Path, out_dir: Path, max_size: int = 320) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    src = Path(src_path)
    dst = out_dir / f"thumb_{src.stem}.jpg"
    with Image.open(src) as im:
        im.thumbnail((max_size, max_size))
        rgb = im.convert("RGB")
        rgb.save(dst, format="JPEG", quality=85)
    return dst




