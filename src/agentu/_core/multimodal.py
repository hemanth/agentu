"""Multi-modal utilities for image + text prompts.

Builds OpenAI-compatible multi-part content arrays from
text and image sources (URLs, base64, local files).
"""

import base64
import logging
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def detect_mime_type(source: str) -> str:
    """Infer MIME type from file extension or data URI prefix.

    Args:
        source: File path, URL, or data URI

    Returns:
        MIME type string (defaults to image/png)
    """
    if source.startswith("data:"):
        # data:image/png;base64,...
        try:
            return source.split(";")[0].split(":")[1]
        except (IndexError, ValueError):
            return "image/png"

    # Try file extension
    mime, _ = mimetypes.guess_type(source)
    return mime or "image/png"


def resolve_image(source: str) -> Dict[str, Any]:
    """Convert an image source to an OpenAI image_url content part.

    Handles:
        - HTTP(S) URLs → pass through
        - data: URIs → pass through
        - Local file paths → read + base64 encode

    Args:
        source: Image URL, data URI, or local file path

    Returns:
        Dict with {"type": "image_url", "image_url": {"url": ...}}
    """
    if source.startswith(("http://", "https://", "data:")):
        url = source
    else:
        # Local file — read and base64 encode
        path = Path(source).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        mime = detect_mime_type(str(path))
        raw = path.read_bytes()
        encoded = base64.b64encode(raw).decode("utf-8")
        url = f"data:{mime};base64,{encoded}"

    return {"type": "image_url", "image_url": {"url": url}}


def build_content_parts(
    text: str, images: Optional[List[str]] = None
) -> Union[str, List[Dict[str, Any]]]:
    """Build OpenAI-compatible content array.

    If no images, returns plain text string (standard format).
    If images provided, returns multi-part content list.

    Args:
        text: Text prompt
        images: Optional list of image sources (URL, data URI, or file path)

    Returns:
        Plain string or list of content parts
    """
    if not images:
        return text

    parts: List[Dict[str, Any]] = [{"type": "text", "text": text}]
    for img in images:
        parts.append(resolve_image(img))

    return parts
