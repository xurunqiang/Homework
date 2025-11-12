from pathlib import Path
from typing import Optional


def run_ocr(image_path: Path) -> Optional[str]:
    """Run OCR using pytesseract or EasyOCR if available.

    Returns extracted text or None when OCR not available.
    """
    # Try pytesseract first
    try:
        import pytesseract  # requires Tesseract installed on system
        from PIL import Image
        with Image.open(image_path) as im:
            return pytesseract.image_to_string(im)
    except Exception:
        pass
    # Try EasyOCR (pure python, heavy deps)
    try:
        import easyocr
        reader = easyocr.Reader(["en", "ch_sim"], gpu=False)
        result = reader.readtext(str(image_path), detail=0)
        return "\n".join(result)
    except Exception:
        return None


def run_asr(audio_or_video_path: Path, language: Optional[str] = None) -> Optional[str]:
    """Run speech recognition using Vosk or Whisper if available.

    Accepts a video path; implementation extracts audio internally if model supports it.
    language: "zh" for Chinese, "en" for English, None for auto-detect
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Try Whisper first (easier to use, no model download needed for base model)
    try:
        import whisper
        logger.info("尝试使用Whisper进行语音识别...")
        model = whisper.load_model("base")
        logger.info("Whisper模型加载成功，开始转录...")
        
        # 如果指定了语言，使用指定语言；否则自动检测
        if language:
            logger.info(f"使用指定语言: {language}")
            result = model.transcribe(str(audio_or_video_path), language=language)
        else:
            logger.info("自动检测语言...")
            result = model.transcribe(str(audio_or_video_path), language=None)  # None = auto-detect
        
        detected_lang = result.get("language", "unknown")
        logger.info(f"检测到的语言: {detected_lang}")
        
        text = result.get("text", "").strip()
        if text:
            logger.info(f"Whisper识别成功，文本长度: {len(text)}")
            return text
        else:
            logger.warning("Whisper识别结果为空")
    except ImportError:
        logger.info("Whisper未安装，尝试Vosk...")
    except Exception as e:
        logger.warning(f"Whisper识别失败: {e}，尝试Vosk...")
    
    # Try Vosk
    try:
        from vosk import Model, KaldiRecognizer
        import wave
        import json
        import subprocess
        import tempfile
        import os

        logger.info("尝试使用Vosk进行语音识别...")
        
        # Extract mono wav via ffmpeg
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_wav = Path(tmp.name)
        
        logger.info("正在提取音频...")
        cmd = [
            "ffmpeg", "-y", "-i", str(audio_or_video_path),
            "-ac", "1", "-ar", "16000", str(tmp_wav)
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        if not tmp_wav.exists():
            logger.error("音频提取失败")
            return None

        # Expect Vosk model directory under ./models/vosk-model
        model_dir = Path(__file__).resolve().parent.parent / "models" / "vosk-model"
        if not model_dir.exists():
            logger.warning(f"Vosk模型目录不存在: {model_dir}")
            return None
        
        logger.info(f"加载Vosk模型: {model_dir}")
        model = Model(str(model_dir))
        wf = wave.open(str(tmp_wav), "rb")
        rec = KaldiRecognizer(model, wf.getframerate())
        rec.SetWords(True)

        text_parts: list[str] = []
        logger.info("开始识别...")
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                j = json.loads(rec.Result())
                if "text" in j and j["text"]:
                    text_parts.append(j["text"])
        
        j = json.loads(rec.FinalResult())
        if "text" in j and j["text"]:
            text_parts.append(j["text"])
        
        # Clean up temp file
        try:
            os.unlink(tmp_wav)
        except:
            pass
        
        text = " ".join(t for t in text_parts if t).strip()
        if text:
            logger.info(f"Vosk识别成功，文本长度: {len(text)}")
            return text
        else:
            logger.warning("Vosk识别结果为空")
    except ImportError:
        logger.warning("Vosk未安装")
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg音频提取失败: {e.stderr}")
    except Exception as e:
        logger.error(f"Vosk识别失败: {e}")
    
    logger.warning("所有语音识别方法都失败或未安装")
    raise RuntimeError("语音识别失败：请安装Whisper (pip install openai-whisper) 或 Vosk (pip install vosk 并下载模型到 app/models/vosk-model/)")




