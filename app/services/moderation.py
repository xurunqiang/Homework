from pathlib import Path
from PIL import Image
from typing import Optional


def moderate_content(image_path: Path, ocr_text: Optional[str] = None) -> list[str]:
    """Content moderation for images and text.

    - Flags image if average pixel in HSV has high saturation and skin-like hue window.
    - Flags text if contains sensitive keywords.
    - Intended only as placeholder; replace with proper model or SaaS.
    """
    flags: list[str] = []
    
    # 图像内容审核
    try:
        from colorsys import rgb_to_hsv
        with Image.open(image_path).convert("RGB") as im:
            small = im.resize((64, 64))
            pixels = list(small.getdata())
            skin_like = 0
            for r, g, b in pixels:
                h, s, v = rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
                # naive skin window: hue in [0.0, 0.1] or [0.9, 1.0), saturation moderate
                if (h < 0.1 or h > 0.9) and 0.2 < s < 0.8 and v > 0.35:
                    skin_like += 1
            ratio = skin_like / (64 * 64)
            if ratio > 0.45:
                flags.append("possible_nsfw_skin_ratio")
    except Exception:
        pass
    
    # 文本内容审核（如果提供了OCR文本）
    if ocr_text:
        sensitive_keywords = [
            # 暴力相关
            "暴力", "血腥", "杀戮", "死亡", "武器", "枪", "刀",
            # 色情相关
            "色情", "性", "裸体", "成人", "18禁",
            # 政治敏感
            "政治", "政府", "抗议", "示威",
            # 其他敏感词（可根据需要扩展）
        ]
        
        text_lower = ocr_text.lower()
        for keyword in sensitive_keywords:
            if keyword.lower() in text_lower:
                flags.append(f"sensitive_text_{keyword}")
                break  # 找到一个就足够标记了
    
    return flags




