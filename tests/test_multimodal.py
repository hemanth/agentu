"""Tests for multimodal support (images, audios, videos, documents, mixed media)."""

import base64
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from agentu import (
    Agent,
    build_content_parts,
    resolve_image,
    resolve_media,
    detect_mime_type,
    detect_media_kind,
)


def _make_agent(**kwargs):
    return Agent("test-agent", model="test-model", auto_discover_rules=False, **kwargs)


class TestDetectMediaKind:
    def test_image_extensions(self):
        for ext in [".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp"]:
            assert detect_media_kind(f"https://example.com/asset{ext}") == "image"
            assert detect_media_kind(f"/tmp/local_file{ext}") == "image"

    def test_video_extensions(self):
        for ext in [".mp4", ".webm", ".mov", ".avi", ".mkv", ".flv", ".m4v"]:
            assert detect_media_kind(f"https://example.com/movie{ext}") == "video"
            assert detect_media_kind(f"./clip{ext}") == "video"

    def test_audio_extensions(self):
        for ext in [".mp3", ".wav", ".ogg", ".m4a", ".aac", ".flac", ".opus"]:
            assert detect_media_kind(f"https://example.com/track{ext}") == "audio"
            assert detect_media_kind(f"/var/log/recording{ext}") == "audio"

    def test_document_extensions(self):
        assert detect_media_kind("https://example.com/doc.pdf") == "document"
        assert detect_media_kind("./whitepaper.pdf") == "document"

    def test_known_video_domains(self):
        assert detect_media_kind("https://www.youtube.com/watch?v=7Z5Vy9JBANs") == "video"
        assert detect_media_kind("https://youtu.be/7Z5Vy9JBANs") == "video"
        assert detect_media_kind("https://vimeo.com/12345678") == "video"
        assert detect_media_kind("https://www.loom.com/share/abcdef") == "video"

    def test_known_audio_domains(self):
        assert detect_media_kind("https://soundcloud.com/artist/track") == "audio"
        assert detect_media_kind("https://spotify.com/episode/xyz") == "audio"

    def test_data_uris(self):
        assert detect_media_kind("data:image/png;base64,abc") == "image"
        assert detect_media_kind("data:video/mp4;base64,abc") == "video"
        assert detect_media_kind("data:audio/mp3;base64,abc") == "audio"
        assert detect_media_kind("data:application/pdf;base64,abc") == "document"

    def test_magic_bytes_detection(self, tmp_path):
        # PNG magic bytes without extension
        png_file = tmp_path / "mystery_png"
        png_file.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")
        assert detect_media_kind(str(png_file)) == "image"

        # JPEG magic bytes without extension
        jpg_file = tmp_path / "mystery_jpg"
        jpg_file.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")
        assert detect_media_kind(str(jpg_file)) == "image"

        # GIF magic bytes
        gif_file = tmp_path / "mystery_gif"
        gif_file.write_bytes(b"GIF89a\x01\x00\x01\x00")
        assert detect_media_kind(str(gif_file)) == "image"

        # MP4 ftyp magic bytes
        mp4_file = tmp_path / "mystery_video"
        mp4_file.write_bytes(b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00")
        assert detect_media_kind(str(mp4_file)) == "video"

        # MP3 ID3 magic bytes
        mp3_file = tmp_path / "mystery_audio"
        mp3_file.write_bytes(b"ID3\x03\x00\x00\x00\x00\x00\x7f")
        assert detect_media_kind(str(mp3_file)) == "audio"

        # PDF magic bytes
        pdf_file = tmp_path / "mystery_doc"
        pdf_file.write_bytes(b"%PDF-1.7\n")
        assert detect_media_kind(str(pdf_file)) == "document"


class TestResolveMedia:
    def test_image_url(self):
        res = resolve_media("https://example.com/chart.png")
        assert res == {"type": "image_url", "image_url": {"url": "https://example.com/chart.png"}}

    def test_video_url(self):
        res = resolve_media("https://youtu.be/7Z5Vy9JBANs")
        assert res == {"type": "video_url", "video_url": {"url": "https://youtu.be/7Z5Vy9JBANs"}}

    def test_audio_url(self):
        res = resolve_media("https://example.com/podcast.mp3")
        assert res == {"type": "audio_url", "audio_url": {"url": "https://example.com/podcast.mp3"}}

    def test_local_image_file(self, tmp_path):
        img = tmp_path / "pic.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n")
        res = resolve_media(str(img))
        assert res["type"] == "image_url"
        assert res["image_url"]["url"].startswith("data:image/png;base64,")

    def test_local_video_file(self, tmp_path):
        vid = tmp_path / "clip.mp4"
        vid.write_bytes(b"\x00\x00\x00\x20ftypmp42")
        res = resolve_media(str(vid))
        assert res["type"] == "video_url"
        assert res["video_url"]["url"].startswith("data:video/mp4;base64,")

    def test_local_audio_file(self, tmp_path):
        aud = tmp_path / "voice.mp3"
        aud.write_bytes(b"ID3\x03\x00\x00\x00")
        res = resolve_media(str(aud))
        assert res["type"] == "input_audio"
        assert res["input_audio"]["format"] == "mp3"
        assert "data" in res["input_audio"]

    def test_explicit_dicts(self):
        # Video dict
        v_dict = {"type": "video", "url": "https://example.com/stream", "processing": "agentic"}
        res = resolve_media(v_dict)
        assert res["type"] == "video_url"
        assert res["video_url"]["url"] == "https://example.com/stream"
        assert res["video_url"]["processing"] == "agentic"

        # Image dict
        i_dict = {"type": "image", "url": "https://example.com/img.jpg", "detail": "high"}
        res = resolve_media(i_dict)
        assert res["type"] == "image_url"
        assert res["image_url"]["url"] == "https://example.com/img.jpg"
        assert res["image_url"]["detail"] == "high"

        # Standard OpenAI dict pass-through
        std_dict = {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
        assert resolve_media(std_dict) == std_dict

    def test_resolve_image_backward_compatibility(self):
        res = resolve_image("https://example.com/test.png")
        assert res == {"type": "image_url", "image_url": {"url": "https://example.com/test.png"}}


class TestBuildContentParts:
    def test_text_only_returns_string(self):
        res = build_content_parts("Hello world")
        assert res == "Hello world"
        assert isinstance(res, str)

    def test_images_kwarg_backward_compatibility(self):
        res = build_content_parts("Hello", images=["https://example.com/a.png"])
        assert isinstance(res, list)
        assert len(res) == 2
        assert res[0] == {"type": "text", "text": "Hello"}
        assert res[1] == {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}

    def test_media_mixed_list(self):
        res = build_content_parts(
            "Analyze these files:",
            media=[
                "https://example.com/chart.png",
                "https://youtu.be/7Z5Vy9JBANs",
                {"type": "audio", "url": "https://example.com/voice.wav"},
            ],
        )
        assert isinstance(res, list)
        assert len(res) == 4
        assert res[0] == {"type": "text", "text": "Analyze these files:"}
        assert res[1]["type"] == "image_url"
        assert res[2]["type"] == "video_url"
        assert res[3]["type"] == "audio_url"


class TestAgentInferMedia:
    @pytest.mark.asyncio
    async def test_infer_with_media_propagates_to_raw_call(self):
        agent = _make_agent()
        mock_raw = AsyncMock(return_value='{"text_response": "I see the video"}')

        with patch.object(agent, "_raw_llm_call", mock_raw):
            res = await agent.infer(
                "What is this?",
                media=["https://youtu.be/7Z5Vy9JBANs"],
            )
            mock_raw.assert_awaited_once()
            _, kwargs = mock_raw.call_args
            assert kwargs["media"] == ["https://youtu.be/7Z5Vy9JBANs"]

    @pytest.mark.asyncio
    async def test_infer_with_images_backward_compatible(self):
        agent = _make_agent()
        mock_raw = AsyncMock(return_value='{"text_response": "I see the image"}')

        with patch.object(agent, "_raw_llm_call", mock_raw):
            res = await agent.infer(
                "Look at this",
                images=["https://example.com/pic.png"],
            )
            mock_raw.assert_awaited_once()
            _, kwargs = mock_raw.call_args
            assert kwargs["images"] == ["https://example.com/pic.png"]


class TestModelCapabilities:
    def test_gemini_capabilities(self):
        from agentu import detect_model_capabilities
        caps = detect_model_capabilities("gemini-2.5-flash")
        assert caps["image"] is True
        assert caps["video"] is True
        assert caps["audio"] is True
        assert caps["document"] is True

    def test_claude_capabilities(self):
        from agentu import detect_model_capabilities
        caps = detect_model_capabilities("claude-3-7-sonnet-20250219")
        assert caps["image"] is True
        assert caps["document"] is True
        assert caps["video"] is False
        assert caps["audio"] is False

    def test_text_only_capabilities(self):
        from agentu import detect_model_capabilities
        caps = detect_model_capabilities("deepseek-r1")
        assert caps["image"] is False
        assert caps["video"] is False
        assert caps["audio"] is False
        assert caps["document"] is False


class TestMediaToMarkdown:
    def test_custom_converter(self):
        from agentu import convert_media_to_markdown
        res = convert_media_to_markdown("https://example.com/clip.mp4", custom_converter=lambda s: f"# Transcribed {s}")
        assert res == "# Transcribed https://example.com/clip.mp4"

    def test_fallback_description(self):
        from agentu import convert_media_to_markdown
        res = convert_media_to_markdown("https://example.com/sound.mp3")
        assert "Audio Attachment: https://example.com/sound.mp3" in res

    def test_build_content_parts_with_text_model_degrades_to_string(self):
        from agentu import build_content_parts
        # When targeting text-only model (DeepSeek), video URL converts to Markdown text string
        res = build_content_parts(
            "Summarize this:",
            media=["https://youtu.be/7Z5Vy9JBANs"],
            model="deepseek-r1",
            custom_converter=lambda s: "- [00:01] Keynote introduction",
        )
        assert isinstance(res, str)
        assert "Summarize this:" in res
        assert "- [00:01] Keynote introduction" in res

    def test_build_content_parts_with_multimodal_model_keeps_parts(self):
        from agentu import build_content_parts
        # When targeting Gemini, video URL stays native video_url part
        res = build_content_parts(
            "Summarize this:",
            media=["https://youtu.be/7Z5Vy9JBANs"],
            model="gemini-2.5-flash",
        )
        assert isinstance(res, list)
        assert len(res) == 2
        assert res[0] == {"type": "text", "text": "Summarize this:"}
        assert res[1]["type"] == "video_url"

    def test_build_content_parts_claude_mixed_media(self):
        from agentu import build_content_parts
        # When targeting Claude: image stays native part, video degrades to markdown in prompt text
        res = build_content_parts(
            "Analyze both:",
            media=[
                "https://example.com/slide.png",
                "https://youtu.be/7Z5Vy9JBANs",
            ],
            model="claude-3-7-sonnet",
            custom_converter=lambda s: "- [00:00] Video content summary",
        )
        assert isinstance(res, list)
        assert len(res) == 2
        assert res[0]["type"] == "text"
        assert "- [00:00] Video content summary" in res[0]["text"]
        assert res[1]["type"] == "image_url"

