"""Zero-cost Cloudflare Python Workers adapter for always-on ``agentu`` agents.

Wraps any ``agentu.Agent`` with:
- Optional HMAC-SHA256 passcode authentication (``HttpOnly; Secure; SameSite=Strict`` cookie)
- Strict ``X-Robots-Tag: noindex, nofollow, noarchive, nosnippet`` headers for private agents
- REST endpoints: ``/api/health``, ``/api/auth/*``, ``/api/chat``, ``/api/tools``,
  ``/api/memory``, and ``/api/webhook/whatsapp`` (Meta WhatsApp Cloud API)
- Cloudflare Cron Trigger hook (``handle_scheduled``)
- Optional Cloudflare KV persistence for memory and chat history
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple


SESSION_COOKIE = "agentu_session"
SESSION_TTL_SEC = 30 * 24 * 60 * 60

NOINDEX_HEADERS = {
    "X-Robots-Tag": "noindex, nofollow, noarchive, nosnippet",
    "Cache-Control": "no-store",
}


@dataclass
class EdgeRequest:
    """Framework-agnostic HTTP request representation (works in Workers & pytest)."""

    method: str
    url: str
    headers: Dict[str, str] = field(default_factory=dict)
    body: str = ""

    @property
    def path(self) -> str:
        return urllib.parse.urlsplit(self.url).path or "/"

    @property
    def query(self) -> Dict[str, str]:
        qs = urllib.parse.parse_qs(urllib.parse.urlsplit(self.url).query)
        return {k: v[0] for k, v in qs.items() if v}

    def header(self, name: str, default: str = "") -> str:
        low = name.lower()
        for k, v in self.headers.items():
            if k.lower() == low:
                return v
        return default

    def json(self) -> Dict[str, Any]:
        if not self.body:
            return {}
        return json.loads(self.body)


@dataclass
class EdgeResponse:
    """Framework-agnostic HTTP response convertible to a Cloudflare Worker Response."""

    body: str
    status: int = 200
    headers: Dict[str, str] = field(default_factory=dict)

    def to_worker_response(self) -> Any:
        """Convert to a Cloudflare Python Workers ``Response`` (or return self on CPython)."""
        try:
            from workers import Response as WorkerResponse  # type: ignore

            return WorkerResponse(self.body, status=self.status, headers=self.headers)
        except ImportError:
            return self


def _json_response(
    data: Any,
    status: int = 200,
    extra_headers: Optional[Dict[str, str]] = None,
    private: bool = True,
) -> EdgeResponse:
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if private:
        headers.update(NOINDEX_HEADERS)
    if extra_headers:
        headers.update(extra_headers)
    return EdgeResponse(body=json.dumps(data), status=status, headers=headers)


def _sign_token(secret: str, exp: int) -> str:
    sig = hmac.new(
        secret.encode("utf-8"),
        f"agentu-edge-v1:{exp}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"{exp}.{sig}"


def _parse_cookie(cookie_header: str, name: str) -> Optional[str]:
    for part in (cookie_header or "").split(";"):
        part = part.strip()
        if "=" in part:
            k, v = part.split("=", 1)
            if k.strip() == name:
                return urllib.parse.unquote(v.strip())
    return None


def _verify_token(cookie_header: str, secret: str) -> bool:
    if not secret:
        return True
    tok = _parse_cookie(cookie_header, SESSION_COOKIE)
    if not tok or "." not in tok:
        return False
    exp_str, sig = tok.split(".", 1)
    try:
        exp = int(exp_str)
    except ValueError:
        return False
    if int(time.time()) > exp:
        return False
    expected = _sign_token(secret, exp).split(".", 1)[1]
    return hmac.compare_digest(sig, expected)


class CloudflareAgentApp:
    """Turn any ``agentu.Agent`` into an always-on Cloudflare Python Worker."""

    def __init__(
        self,
        agent: Any,
        *,
        passcode: Optional[str] = None,
        private: bool = True,
        whatsapp_verify_token: Optional[str] = None,
        on_cron: Optional[Callable[[Any, Any], Awaitable[Any]]] = None,
        ui_html: Optional[str] = None,
        kv_namespace_binding: str = "AGENT_KV",
    ):
        self.agent = agent
        self.passcode = (passcode or "").strip() or None
        self.private = private
        self.whatsapp_verify_token = (whatsapp_verify_token or "").strip() or None
        self.on_cron = on_cron
        self.ui_html = ui_html
        self.kv_namespace_binding = kv_namespace_binding
        self._fail_counts: Dict[str, Tuple[int, float]] = {}

    def _get_passcode(self, env: Any = None) -> Optional[str]:
        if self.passcode:
            return self.passcode
        if env is not None:
            val = getattr(env, "AGENT_PASSCODE", None)
            if val:
                return str(val).strip()
        return None

    def _get_kv(self, env: Any = None) -> Any:
        if env is None:
            return None
        return getattr(env, self.kv_namespace_binding, None)

    def _check_rate_limit(self, ip: str, ok: bool) -> bool:
        now = time.time()
        count, reset_at = self._fail_counts.get(ip, (0, now + 300))
        if now > reset_at:
            count, reset_at = 0, now + 300
        if ok:
            self._fail_counts.pop(ip, None)
            return False
        count += 1
        self._fail_counts[ip] = (count, reset_at)
        return count >= 10

    async def _coerce_request(self, request: Any) -> EdgeRequest:
        if isinstance(request, EdgeRequest):
            return request
        method = str(getattr(request, "method", "GET")).upper()
        url = str(getattr(request, "url", "https://edge/"))
        raw_headers = getattr(request, "headers", None)
        headers: Dict[str, str] = {}
        if raw_headers is not None:
            if hasattr(raw_headers, "items"):
                headers = {str(k): str(v) for k, v in raw_headers.items()}
            elif hasattr(raw_headers, "entries"):
                for pair in raw_headers.entries():
                    headers[str(pair[0])] = str(pair[1])
        body = ""
        if method not in ("GET", "HEAD") and hasattr(request, "text"):
            body = str(await request.text())
        return EdgeRequest(method=method, url=url, headers=headers, body=body)

    async def handle_edge_request(self, req: EdgeRequest, env: Any = None) -> EdgeResponse:
        path = req.path
        secret = self._get_passcode(env)

        # Public health check
        if req.method == "GET" and path == "/api/health":
            return _json_response(
                {"ok": True, "agent": getattr(self.agent, "name", "agentu")},
                private=self.private,
            )

        # Auth status
        if req.method == "GET" and path == "/api/auth/status":
            authed = _verify_token(req.header("cookie"), secret or "")
            return _json_response(
                {"authenticated": authed, "passcodeRequired": bool(secret)},
                private=self.private,
            )

        # Passcode login
        if req.method == "POST" and path == "/api/auth/login":
            if not secret:
                return _json_response({"ok": True}, private=self.private)
            ip = req.header("cf-connecting-ip", "client")
            try:
                body = req.json()
            except Exception:
                return _json_response({"error": "Invalid JSON"}, status=400, private=self.private)
            attempt = str(body.get("passcode", "")).strip()
            matched = hmac.compare_digest(attempt, secret)
            if self._check_rate_limit(ip, matched):
                return _json_response(
                    {"error": "Too many attempts. Try again in 5 minutes."},
                    status=429,
                    private=self.private,
                )
            if not matched:
                return _json_response({"error": "Incorrect passcode."}, status=401, private=self.private)
            exp = int(time.time()) + SESSION_TTL_SEC
            token = _sign_token(secret, exp)
            cookie = (
                f"{SESSION_COOKIE}={urllib.parse.quote(token)}; Path=/; "
                f"HttpOnly; Secure; SameSite=Strict; Max-Age={SESSION_TTL_SEC}"
            )
            return _json_response({"ok": True}, extra_headers={"Set-Cookie": cookie}, private=self.private)

        # Logout
        if req.method == "POST" and path == "/api/auth/logout":
            cookie = f"{SESSION_COOKIE}=; Path=/; HttpOnly; Secure; SameSite=Strict; Max-Age=0"
            return _json_response({"ok": True}, extra_headers={"Set-Cookie": cookie}, private=self.private)

        # Meta WhatsApp Cloud API Webhook (uses its own verify token)
        if path == "/api/webhook/whatsapp":
            verify_tok = self.whatsapp_verify_token or (
                str(getattr(env, "WHATSAPP_VERIFY_TOKEN", "") or "").strip() if env else ""
            )
            if req.method == "GET":
                mode = req.query.get("hub.mode", "")
                token = req.query.get("hub.verify_token", "")
                challenge = req.query.get("hub.challenge", "")
                if mode == "subscribe" and verify_tok and hmac.compare_digest(token, verify_tok):
                    return EdgeResponse(
                        body=challenge,
                        status=200,
                        headers={"Content-Type": "text/plain", **(NOINDEX_HEADERS if self.private else {})},
                    )
                return EdgeResponse(body="Forbidden", status=403, headers=NOINDEX_HEADERS if self.private else {})

            if req.method == "POST":
                try:
                    payload = req.json()
                except Exception:
                    return _json_response({"error": "Invalid JSON"}, status=400, private=self.private)
                entries = payload.get("entry") or []
                replies: List[Dict[str, str]] = []
                for entry in entries:
                    for change in entry.get("changes") or []:
                        val = change.get("value") or {}
                        for msg in val.get("messages") or []:
                            text = (msg.get("text") or {}).get("body", "").strip()
                            sender = str(msg.get("from", ""))
                            if text:
                                res = await self.agent.infer(text)
                                reply_text = getattr(res, "text", str(res))
                                replies.append({"to": sender, "reply": reply_text})
                return _json_response({"ok": True, "replies": replies}, private=self.private)

        # Root UI page (returns minimal passcode-gated UI if configured)
        if req.method == "GET" and path in ("/", "/index.html"):
            if self.ui_html:
                headers = {"Content-Type": "text/html; charset=utf-8"}
                if self.private:
                    headers.update(NOINDEX_HEADERS)
                return EdgeResponse(body=self.ui_html, status=200, headers=headers)
            return _json_response(
                {"ok": True, "agent": getattr(self.agent, "name", "agentu")},
                private=self.private,
            )

        # All other /api/* endpoints require valid session when passcode is set
        if path.startswith("/api/"):
            if secret and not _verify_token(req.header("cookie"), secret):
                return _json_response({"error": "Unauthorized"}, status=401, private=self.private)

            if req.method == "GET" and path == "/api/tools":
                tools = getattr(self.agent, "tools", {}) or {}
                items = [
                    {
                        "name": name,
                        "description": getattr(t, "description", ""),
                    }
                    for name, t in tools.items()
                ]
                return _json_response({"tools": items}, private=self.private)

            if req.method == "POST" and path == "/api/chat":
                try:
                    body = req.json()
                except Exception:
                    return _json_response({"error": "Invalid JSON"}, status=400, private=self.private)
                prompt = str(body.get("message") or body.get("prompt") or "").strip()
                if not prompt:
                    return _json_response({"error": "Empty message"}, status=400, private=self.private)
                res = await self.agent.infer(prompt)
                text = getattr(res, "text", str(res))
                return _json_response(
                    {
                        "ok": True,
                        "reply": text,
                        "agent": getattr(self.agent, "name", "agentu"),
                    },
                    private=self.private,
                )

            if path == "/api/memory":
                kv = self._get_kv(env)
                if req.method == "GET":
                    notes = []
                    if kv is not None:
                        raw = await kv.get("agentu:notes")
                        if raw:
                            try:
                                notes = json.loads(raw)
                            except Exception:
                                notes = []
                    return _json_response({"notes": notes}, private=self.private)
                if req.method == "POST":
                    try:
                        body = req.json()
                    except Exception:
                        return _json_response({"error": "Invalid JSON"}, status=400, private=self.private)
                    note = str(body.get("note") or "").strip()
                    if not note:
                        return _json_response({"error": "Empty note"}, status=400, private=self.private)
                    if hasattr(self.agent, "remember"):
                        self.agent.remember(note)
                    if kv is not None:
                        raw = await kv.get("agentu:notes")
                        notes = json.loads(raw) if raw else []
                        notes.append({"note": note, "ts": int(time.time())})
                        await kv.put("agentu:notes", json.dumps(notes[-100:]))
                    return _json_response({"ok": True}, private=self.private)

        return _json_response({"error": "Not Found"}, status=404, private=self.private)

    async def handle_fetch(self, request: Any, env: Any = None) -> Any:
        edge_req = await self._coerce_request(request)
        edge_res = await self.handle_edge_request(edge_req, env=env)
        return edge_res.to_worker_response()

    async def handle_scheduled(self, controller: Any, env: Any = None) -> Any:
        if self.on_cron is not None:
            return await self.on_cron(self.agent, env)
        return None
