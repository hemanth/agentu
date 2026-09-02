"""Multi-modal utilities for image, audio, video, and document prompts.

Builds OpenAI/GenAI-compatible multi-part content arrays from
text and media sources (URLs, base64, local files, or explicit dicts).
"""

import base64
import logging
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Known video domains (YouTube, Vimeo, Loom, TikTok, Twitch, Dailymotion)
_VIDEO_DOMAINS = (
    "youtube.com",
    "youtu.be",
    "vimeo.com",
    "loom.com",
    "dailymotion.com",
    "tiktok.com",
    "twitch.tv",
)

# Known audio domains
_AUDIO_DOMAINS = (
    "soundcloud.com",
    "spotify.com",
    "podcasts.apple.com",
    "bandcamp.com",
)

_IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg",
    ".ico", ".tiff", ".tif", ".avif", ".heic",
}

_AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".ogg", ".m4a", ".aac", ".flac",
    ".wma", ".opus", ".aiff", ".alac", ".oga",
}

_VIDEO_EXTENSIONS = {
    ".mp4", ".webm", ".mov", ".avi", ".mkv", ".flv",
    ".m4v", ".wmv", ".ts", ".3gp", ".ogv",
}

_DOCUMENT_EXTENSIONS = {
    ".pdf",
}


def _sniff_magic_bytes(header: bytes) -> Optional[Tuple[str, str]]:
    """Sniff media category and MIME type from raw file header bytes.

    Args:
        header: Initial bytes of the file (typically first 32-64 bytes)

    Returns:
        Tuple of (category, mime_type) e.g. ("image", "image/png"), or None
    """
    if len(header) < 4:
        return None

    # PNG: \x89PNG\r\n\x1a\n
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image", "image/png"

    # JPEG: \xff\xd8\xff
    if header.startswith(b"\xff\xd8\xff"):
        return "image", "image/jpeg"

    # GIF: GIF87a or GIF89a
    if header.startswith((b"GIF87a", b"GIF89a")):
        return "image", "image/gif"

    # WEBP: RIFF....WEBP
    if header.startswith(b"RIFF") and len(header) >= 12 and header[8:12] == b"WEBP":
        return "image", "image/webp"

    # PDF: %PDF-
    if header.startswith(b"%PDF-"):
        return "document", "application/pdf"

    # WAV: RIFF....WAVE
    if header.startswith(b"RIFF") and len(header) >= 12 and header[8:12] == b"WAVE":
        return "audio", "audio/wav"

    # OGG: OggS
    if header.startswith(b"OggS"):
        return "audio", "audio/ogg"

    # FLAC: fLaC
    if header.startswith(b"fLaC"):
        return "audio", "audio/flac"

    # MP3: ID3 or sync frame \xff\xfb, \xff\xf3, \xff\xf2
    if header.startswith(b"ID3") or (
        len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0
    ):
        return "audio", "audio/mp3"

    # MP4 / MOV: ....ftyp
    if len(header) >= 8 and header[4:8] == b"ftyp":
        return "video", "video/mp4"

    # WebM / MKV: \x1a\x45\xdf\xa3
    if header.startswith(b"\x1a\x45\xdf\xa3"):
        return "video", "video/webm"

    return None


def detect_mime_type(source: str) -> str:
    """Infer MIME type from file extension, data URI prefix, or magic bytes.

    Args:
        source: File path, URL, or data URI

    Returns:
        MIME type string (defaults to image/png for image fallback)
    """
    if source.startswith("data:"):
        # data:image/png;base64,...
        try:
            return source.split(";")[0].split(":")[1]
        except (IndexError, ValueError):
            return "image/png"

    # Check local file magic bytes
    if not source.startswith(("http://", "https://")):
        path = Path(source).expanduser()
        if path.exists() and path.is_file():
            try:
                with open(path, "rb") as f:
                    sniffed = _sniff_magic_bytes(f.read(64))
                    if sniffed:
                        return sniffed[1]
            except Exception:
                pass

    # Try file extension
    mime, _ = mimetypes.guess_type(source)
    if mime:
        return mime

    # Check known video/audio domains
    source_lower = source.lower()
    if any(domain in source_lower for domain in _VIDEO_DOMAINS):
        return "video/mp4"
    if any(domain in source_lower for domain in _AUDIO_DOMAINS):
        return "audio/mp3"

    return "image/png"


def detect_media_kind(source: str) -> str:
    """Classify media into 'image', 'audio', 'video', or 'document'.

    Args:
        source: URL, data URI, or local file path

    Returns:
        One of 'image', 'audio', 'video', 'document'
    """
    source_lower = source.lower()

    # Data URI check
    if source_lower.startswith("data:"):
        if source_lower.startswith("data:video/"):
            return "video"
        if source_lower.startswith("data:audio/"):
            return "audio"
        if source_lower.startswith("data:application/pdf"):
            return "document"
        return "image"

    # Known domain check
    if any(domain in source_lower for domain in _VIDEO_DOMAINS):
        return "video"
    if any(domain in source_lower for domain in _AUDIO_DOMAINS):
        return "audio"

    # File extension check (extract path from URL if needed)
    path_part = source_lower.split("?")[0].split("#")[0]
    suffix = Path(path_part).suffix

    if suffix in _VIDEO_EXTENSIONS:
        return "video"
    if suffix in _AUDIO_EXTENSIONS:
        return "audio"
    if suffix in _IMAGE_EXTENSIONS:
        return "image"
    if suffix in _DOCUMENT_EXTENSIONS:
        return "document"

    # Local file magic bytes sniffing
    if not source.startswith(("http://", "https://")):
        path = Path(source).expanduser()
        if path.exists() and path.is_file():
            try:
                with open(path, "rb") as f:
                    sniffed = _sniff_magic_bytes(f.read(64))
                    if sniffed:
                        return sniffed[0]
            except Exception:
                pass

    # Default fallback
    mime, _ = mimetypes.guess_type(path_part)
    if mime:
        if mime.startswith("video/"):
            return "video"
        if mime.startswith("audio/"):
            return "audio"
        if mime == "application/pdf":
            return "document"

    return "image"


def resolve_media(source: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Convert a media source to an OpenAI/GenAI-compatible content part.

    Handles:
        - Explicit dicts: {"type": "video", "url": "..."} or {"type": "image_url", ...}
        - HTTP(S) URLs → auto-classified into image_url, video_url, or audio_url
        - data: URIs → auto-classified into image_url, video_url, or audio_url
        - Local file paths → read + base64 encoded into content parts

    Args:
        source: Media URL, data URI, local file path, or explicit dict

    Returns:
        Standard content part dictionary
    """
    if isinstance(source, dict):
        # Normalize explicit {"type": "video|image|audio", "url|uri": "..."}
        media_type = source.get("type", "")
        url = source.get("url") or source.get("uri")

        if media_type == "video" and url:
            rest = {k: v for k, v in source.items() if k not in ("type", "url", "uri")}
            return {"type": "video_url", "video_url": {"url": url, **rest}}
        if media_type == "image" and url:
            rest = {k: v for k, v in source.items() if k not in ("type", "url", "uri")}
            return {"type": "image_url", "image_url": {"url": url, **rest}}
        if media_type == "audio" and url:
            rest = {k: v for k, v in source.items() if k not in ("type", "url", "uri")}
            return {"type": "audio_url", "audio_url": {"url": url, **rest}}

        # Already standard (e.g. image_url, video_url, input_audio, or custom GenAI part)
        return source

    if not isinstance(source, str):
        raise TypeError(f"Media source must be str or dict, got {type(source).__name__}")

    kind = detect_media_kind(source)
    mime = detect_mime_type(source)

    if source.startswith(("http://", "https://", "data:")):
        url = source
    else:
        # Local file — read and base64 encode
        path = Path(source).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Media file not found: {path}")

        raw = path.read_bytes()
        encoded = base64.b64encode(raw).decode("utf-8")
        url = f"data:{mime};base64,{encoded}"

        # If audio, support standard input_audio part format as well
        if kind == "audio":
            # Extract format e.g. mp3, wav
            audio_format = mime.split("/")[-1].replace("mpeg", "mp3")
            return {
                "type": "input_audio",
                "input_audio": {"data": encoded, "format": audio_format},
            }

    if kind == "video":
        return {"type": "video_url", "video_url": {"url": url}}
    if kind == "audio":
        return {"type": "audio_url", "audio_url": {"url": url}}
    if kind == "document":
        return {"type": "document", "document": {"url": url, "mime_type": mime}}

    return {"type": "image_url", "image_url": {"url": url}}


def resolve_image(source: str) -> Dict[str, Any]:
    """Convert an image source to an OpenAI image_url content part (backward-compat).

    Args:
        source: Image URL, data URI, or local file path

    Returns:
        Dict with {"type": "image_url", "image_url": {"url": ...}}
    """
    return resolve_media(source)


def build_content_parts(
    text: str,
    images: Optional[List[str]] = None,
    media: Optional[List[Union[str, Dict[str, Any]]]] = None,
) -> Union[str, List[Dict[str, Any]]]:
    """Build OpenAI/GenAI-compatible content array.

    If no media or images provided, returns plain text string.
    If media or images provided, returns multi-part content list.

    Args:
        text: Text prompt
        images: Optional list of image sources (URL, data URI, or file path)
        media: Optional list of media sources (images, audios, videos, or explicit dicts)

    Returns:
        Plain string or list of content parts
    """
    all_media: List[Union[str, Dict[str, Any]]] = []
    if images:
        all_media.extend(images)
    if media:
        all_media.extend(media)

    if not all_media:
        return text

    parts: List[Dict[str, Any]] = [{"type": "text", "text": text}]
    for item in all_media:
        parts.append(resolve_media(item))

    return parts
