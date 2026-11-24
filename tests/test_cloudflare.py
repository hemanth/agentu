"""Tests for Cloudflare Python Workers support (FetchClientSession, InMemoryStorage, CloudflareAgentApp)."""

import json
import pytest

from agentu import Agent, CloudflareAgentApp, EdgeRequest, InMemoryStorage
from agentu._core.http_compat import FetchClientSession, StdlibClientResponseError


class _FakePyFetchResponse:
    def __init__(self, status=200, body=b"{}", headers=None, url="https://api.example/v1"):
        self.status = status
        self.status_text = "OK" if status < 400 else "Error"
        self.headers = headers or {"Content-Type": "application/json"}
        self.url = url
        self._body = body

    async def bytes(self):
        return self._body


@pytest.mark.asyncio
async def test_fetch_client_session_json_and_sse():
    calls = []

    async def fake_pyfetch(url, **kwargs):
        calls.append((url, kwargs))
        if "stream" in url:
            return _FakePyFetchResponse(
                status=200,
                body=b'data: {"delta":"hi"}\n\ndata: [DONE]\n',
                headers={"Content-Type": "text/event-stream"},
                url=url,
            )
        return _FakePyFetchResponse(status=200, body=b'{"ok":true,"answer":42}', url=url)

    async with FetchClientSession(fetcher=fake_pyfetch) as session:
        async with session.post("https://api.example/v1/chat", json={"msg": "hello"}) as resp:
            resp.raise_for_status()
            data = await resp.json()
            assert data == {"ok": True, "answer": 42}
            assert calls[0][1]["headers"]["Content-Type"] == "application/json"

        async with session.get("https://api.example/v1/stream") as resp:
            lines = [line async for line in resp.content]
            assert len(lines) == 3
            assert b"hi" in lines[0]


@pytest.mark.asyncio
async def test_fetch_client_session_raises_on_http_error():
    async def fake_pyfetch(url, **kwargs):
        return _FakePyFetchResponse(status=401, body=b'{"error":"bad key"}', url=url)

    async with FetchClientSession(fetcher=fake_pyfetch) as session:
        async with session.get("https://api.example/v1/models") as resp:
            with pytest.raises(StdlibClientResponseError) as exc:
                resp.raise_for_status()
            assert exc.value.status == 401


def test_in_memory_storage_roundtrip():
    store = InMemoryStorage()
    store.save([{"content": "Prefers oat milk", "importance": 0.9}])
    snap = store.export_json()

    store2 = InMemoryStorage()
    store2.import_json(snap)
    loaded = store2.load()
    assert len(loaded) == 1
    assert loaded[0]["content"] == "Prefers oat milk"


@pytest.mark.asyncio
async def test_cloudflare_agent_app_passcode_chat_and_whatsapp():
    class _FakeAgent:
        name = "personal-ops"
        tools = {}

        async def infer(self, prompt: str):
            class _R:
                text = f"echo:{prompt}"

            return _R()

    app = CloudflareAgentApp(
        _FakeAgent(),
        passcode="secret-123",
        private=True,
        whatsapp_verify_token="wa-verify-99",
    )

    # 1. Health is public and non-indexable
    h = await app.handle_edge_request(EdgeRequest("GET", "https://h3manth.com/api/health"))
    assert h.status == 200
    assert "noindex" in h.headers["X-Robots-Tag"]

    # 2. Chat without cookie is 401
    unauth = await app.handle_edge_request(
        EdgeRequest("POST", "https://h3manth.com/api/chat", body=json.dumps({"message": "hi"}))
    )
    assert unauth.status == 401

    # 3. Wrong passcode returns 401
    bad_login = await app.handle_edge_request(
        EdgeRequest("POST", "https://h3manth.com/api/auth/login", body=json.dumps({"passcode": "wrong"}))
    )
    assert bad_login.status == 401

    # 4. Valid passcode issues HttpOnly cookie and unlocks /api/chat
    login = await app.handle_edge_request(
        EdgeRequest("POST", "https://h3manth.com/api/auth/login", body=json.dumps({"passcode": "secret-123"}))
    )
    assert login.status == 200
    cookie = login.headers["Set-Cookie"].split(";", 1)[0]

    chat = await app.handle_edge_request(
        EdgeRequest(
            "POST",
            "https://h3manth.com/api/chat",
            headers={"Cookie": cookie},
            body=json.dumps({"message": "check coffee deals"}),
        )
    )
    assert chat.status == 200
    assert json.loads(chat.body)["reply"] == "echo:check coffee deals"

    # 5. WhatsApp webhook verification challenge
    wa = await app.handle_edge_request(
        EdgeRequest(
            "GET",
            "https://h3manth.com/api/webhook/whatsapp?hub.mode=subscribe&hub.verify_token=wa-verify-99&hub.challenge=CHALLENGE_OK",
        )
    )
    assert wa.status == 200
    assert wa.body == "CHALLENGE_OK"
