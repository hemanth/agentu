"""Zero-dependency async HTTP/1.1 + TLS + SSE client with progressive aiohttp acceleration.

When ``aiohttp`` is installed in the environment, ``aiohttp`` resolves directly to
the native C/Cython-accelerated ``aiohttp`` package. When ``aiohttp`` is absent
(base ``pip install agentu`` with zero dependencies), ``aiohttp`` resolves to a
pure Python standard-library async implementation built on ``asyncio`` streams
and ``ssl``.
"""

from __future__ import annotations

import asyncio
import json as _json
import ssl
import urllib.parse
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Iterator, Mapping, Optional, Tuple


class StdlibClientError(Exception):
    """Base exception for stdlib HTTP transport errors."""


class StdlibClientResponseError(StdlibClientError):
    """HTTP status error raised by ``response.raise_for_status()``."""

    def __init__(
        self,
        request_info: Any = None,
        history: Tuple[Any, ...] = (),
        *,
        status: int = 0,
        message: str = "",
        headers: Optional[Mapping[str, str]] = None,
    ):
        super().__init__(f"{status}, message={message!r}")
        self.request_info = request_info
        self.history = history
        self.status = status
        self.message = message
        self.headers = headers or {}


@dataclass
class StdlibClientTimeout:
    """Drop-in replacement for ``aiohttp.ClientTimeout``."""

    total: Optional[float] = None
    connect: Optional[float] = None
    sock_read: Optional[float] = None
    sock_connect: Optional[float] = None


class CaseInsensitiveHeaders(dict):
    """Case-insensitive HTTP header mapping."""

    def __init__(self, initial: Optional[Mapping[str, str]] = None):
        super().__init__()
        self._lower_map: Dict[str, str] = {}
        if initial:
            for k, v in initial.items():
                self[k] = v

    def __setitem__(self, key: str, value: str) -> None:
        low = key.lower()
        if low in self._lower_map:
            super().__delitem__(self._lower_map[low])
        self._lower_map[low] = key
        super().__setitem__(key, value)

    def __getitem__(self, key: str) -> str:
        actual = self._lower_map.get(key.lower(), key)
        return super().__getitem__(actual)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return key.lower() in self._lower_map

    def get(self, key: str, default: Any = None) -> Any:
        actual = self._lower_map.get(key.lower())
        if actual is None:
            return default
        return super().get(actual, default)


class StdlibStreamContent:
    """Async line/byte stream over an HTTP/1.1 response (supports chunked + Content-Length + EOF)."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        *,
        chunked: bool,
        content_length: Optional[int],
        read_timeout: Optional[float],
    ):
        self._reader = reader
        self._writer = writer
        self._chunked = chunked
        self._remaining = content_length
        self._timeout = read_timeout
        self._closed = False
        self._buffer = bytearray()

    async def _with_timeout(self, coro: Any) -> Any:
        if self._timeout is not None and self._timeout > 0:
            return await asyncio.wait_for(coro, timeout=self._timeout)
        return await coro

    async def _read_next_block(self) -> bytes:
        if self._closed:
            return b""

        if self._chunked:
            size_line = await self._with_timeout(self._reader.readline())
            if not size_line:
                self._closed = True
                return b""
            size_str = size_line.split(b";", 1)[0].strip()
            if not size_str:
                return b""
            chunk_size = int(size_str, 16)
            if chunk_size == 0:
                # Consume trailing CRLF after terminal 0 chunk
                await self._with_timeout(self._reader.readline())
                self._closed = True
                return b""
            data = await self._with_timeout(self._reader.readexactly(chunk_size))
            # Consume CRLF after chunk payload
            await self._with_timeout(self._reader.readexactly(2))
            return data

        if self._remaining is not None:
            if self._remaining <= 0:
                self._closed = True
                return b""
            to_read = min(65536, self._remaining)
            data = await self._with_timeout(self._reader.read(to_read))
            if not data:
                self._closed = True
                return b""
            self._remaining -= len(data)
            if self._remaining <= 0:
                self._closed = True
            return data

        data = await self._with_timeout(self._reader.read(65536))
        if not data:
            self._closed = True
        return data

    async def read_all(self) -> bytes:
        parts = [bytes(self._buffer)] if self._buffer else []
        self._buffer.clear()
        while not self._closed:
            blk = await self._read_next_block()
            if not blk:
                break
            parts.append(blk)
        return b"".join(parts)

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._iter_lines()

    async def _iter_lines(self) -> AsyncIterator[bytes]:
        while True:
            nl = self._buffer.find(b"\n")
            if nl != -1:
                line = bytes(self._buffer[: nl + 1])
                del self._buffer[: nl + 1]
                yield line
                continue
            if self._closed:
                if self._buffer:
                    tail = bytes(self._buffer)
                    self._buffer.clear()
                    yield tail
                break
            blk = await self._read_next_block()
            if not blk:
                if self._buffer:
                    tail = bytes(self._buffer)
                    self._buffer.clear()
                    yield tail
                break
            self._buffer.extend(blk)


class StdlibClientResponse:
    """Async context-managed HTTP response compatible with ``aiohttp.ClientResponse``."""

    def __init__(
        self,
        url: str,
        status: int,
        reason: str,
        headers: CaseInsensitiveHeaders,
        content: StdlibStreamContent,
        writer: asyncio.StreamWriter,
    ):
        self.url = url
        self.status = status
        self.reason = reason
        self.headers = headers
        self.content = content
        self._writer = writer
        self._body_cache: Optional[bytes] = None

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise StdlibClientResponseError(
                self.url,
                status=self.status,
                message=self.reason or f"HTTP {self.status}",
                headers=self.headers,
            )

    async def read(self) -> bytes:
        if self._body_cache is None:
            self._body_cache = await self.content.read_all()
        return self._body_cache

    async def text(self, encoding: str = "utf-8") -> str:
        raw = await self.read()
        return raw.decode(encoding, errors="replace")

    async def json(self) -> Any:
        txt = await self.text()
        return _json.loads(txt)

    async def release(self) -> None:
        try:
            self._writer.close()
            await self._writer.wait_closed()
        except Exception:
            pass

    async def __aenter__(self) -> "StdlibClientResponse":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.release()


class _RequestContextManager:
    """Allows ``async with session.post(...) as resp:`` as well as ``resp = await session.post(...)``."""

    def __init__(self, coro: Any):
        self._coro = coro
        self._resp: Optional[StdlibClientResponse] = None

    def __await__(self) -> Iterator[Any]:
        return self._coro.__await__()

    async def __aenter__(self) -> StdlibClientResponse:
        self._resp = await self._coro
        return self._resp

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._resp is not None:
            await self._resp.release()


class StdlibClientSession:
    """Pure standard-library async HTTP/1.1 + TLS + SSE client session."""

    def __init__(
        self,
        *,
        timeout: Optional[StdlibClientTimeout] = None,
        headers: Optional[Mapping[str, str]] = None,
        **_: Any,
    ):
        self._default_timeout = timeout
        self._default_headers = dict(headers) if headers else {}
        self.closed = False
        self._ssl_ctx: Optional[ssl.SSLContext] = None

    def _get_ssl_context(self) -> ssl.SSLContext:
        if self._ssl_ctx is None:
            self._ssl_ctx = ssl.create_default_context()
        return self._ssl_ctx

    def request(
        self,
        method: str,
        url: str,
        *,
        json: Any = None,
        data: Any = None,
        headers: Optional[Mapping[str, str]] = None,
        timeout: Optional[Any] = None,
        max_redirects: int = 5,
    ) -> _RequestContextManager:
        return _RequestContextManager(
            self._send(
                method=method.upper(),
                url=url,
                json_body=json,
                data=data,
                headers=headers,
                timeout=timeout,
                max_redirects=max_redirects,
            )
        )

    def get(self, url: str, **kwargs: Any) -> _RequestContextManager:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> _RequestContextManager:
        return self.request("POST", url, **kwargs)

    def put(self, url: str, **kwargs: Any) -> _RequestContextManager:
        return self.request("PUT", url, **kwargs)

    def delete(self, url: str, **kwargs: Any) -> _RequestContextManager:
        return self.request("DELETE", url, **kwargs)

    async def _send(
        self,
        *,
        method: str,
        url: str,
        json_body: Any,
        data: Any,
        headers: Optional[Mapping[str, str]],
        timeout: Optional[Any],
        max_redirects: int,
    ) -> StdlibClientResponse:
        if self.closed:
            raise StdlibClientError("Session is closed")

        eff_timeout: Optional[float] = None
        if isinstance(timeout, (int, float)):
            eff_timeout = float(timeout)
        elif timeout is not None and hasattr(timeout, "total"):
            eff_timeout = timeout.total
        elif self._default_timeout is not None:
            eff_timeout = self._default_timeout.total

        merged_headers = CaseInsensitiveHeaders(self._default_headers)
        if headers:
            for k, v in headers.items():
                merged_headers[k] = v

        body_bytes = b""
        if json_body is not None:
            body_bytes = _json.dumps(json_body).encode("utf-8")
            if "Content-Type" not in merged_headers:
                merged_headers["Content-Type"] = "application/json"
        elif data is not None:
            body_bytes = data if isinstance(data, bytes) else str(data).encode("utf-8")

        current_url = url
        for _ in range(max_redirects + 1):
            parsed = urllib.parse.urlsplit(current_url)
            scheme = (parsed.scheme or "http").lower()
            host = parsed.hostname
            if not host:
                raise StdlibClientError(f"Invalid URL: {current_url!r}")
            use_tls = scheme == "https"
            port = parsed.port or (443 if use_tls else 80)
            path = parsed.path or "/"
            if parsed.query:
                path = f"{path}?{parsed.query}"

            host_header = f"{host}:{port}" if parsed.port else host
            req_headers = CaseInsensitiveHeaders(merged_headers)
            req_headers["Host"] = host_header
            req_headers["Connection"] = "close"
            if "User-Agent" not in req_headers:
                req_headers["User-Agent"] = "agentu-stdlib-http/2.11"
            if body_bytes or method in ("POST", "PUT", "PATCH"):
                req_headers["Content-Length"] = str(len(body_bytes))

            try:
                conn_coro = asyncio.open_connection(
                    host,
                    port,
                    ssl=self._get_ssl_context() if use_tls else None,
                    server_hostname=host if use_tls else None,
                    limit=1024 * 1024,
                )
                reader, writer = (
                    await asyncio.wait_for(conn_coro, timeout=eff_timeout)
                    if eff_timeout
                    else await conn_coro
                )
            except asyncio.TimeoutError:
                raise
            except OSError as exc:
                raise StdlibClientError(f"Connection failed to {host}:{port}: {exc}") from exc

            try:
                lines = [f"{method} {path} HTTP/1.1"]
                for hk, hv in req_headers.items():
                    lines.append(f"{hk}: {hv}")
                raw_head = ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1")
                writer.write(raw_head)
                if body_bytes:
                    writer.write(body_bytes)
                if eff_timeout:
                    await asyncio.wait_for(writer.drain(), timeout=eff_timeout)
                else:
                    await writer.drain()

                status_line_bytes = (
                    await asyncio.wait_for(reader.readline(), timeout=eff_timeout)
                    if eff_timeout
                    else await reader.readline()
                )
                if not status_line_bytes:
                    raise StdlibClientError(f"Empty HTTP status line from {current_url}")
                status_line = status_line_bytes.decode("latin-1", errors="replace").strip()
                parts = status_line.split(" ", 2)
                status_code = int(parts[1]) if len(parts) >= 2 else 500
                reason = parts[2] if len(parts) >= 3 else ""

                resp_headers = CaseInsensitiveHeaders()
                while True:
                    hline = (
                        await asyncio.wait_for(reader.readline(), timeout=eff_timeout)
                        if eff_timeout
                        else await reader.readline()
                    )
                    if not hline or hline in (b"\r\n", b"\n"):
                        break
                    decoded = hline.decode("latin-1", errors="replace").rstrip("\r\n")
                    if ":" in decoded:
                        k, v = decoded.split(":", 1)
                        resp_headers[k.strip()] = v.strip()

                if status_code in (301, 302, 303, 307, 308) and "Location" in resp_headers:
                    writer.close()
                    try:
                        await writer.wait_closed()
                    except Exception:
                        pass
                    current_url = urllib.parse.urljoin(current_url, resp_headers["Location"])
                    if status_code == 303:
                        method = "GET"
                        body_bytes = b""
                    continue

                te = resp_headers.get("Transfer-Encoding", "").lower()
                is_chunked = "chunked" in te
                cl_str = resp_headers.get("Content-Length")
                content_len = int(cl_str) if (cl_str is not None and cl_str.isdigit() and not is_chunked) else None

                stream = StdlibStreamContent(
                    reader,
                    writer,
                    chunked=is_chunked,
                    content_length=content_len,
                    read_timeout=eff_timeout,
                )
                return StdlibClientResponse(
                    url=current_url,
                    status=status_code,
                    reason=reason,
                    headers=resp_headers,
                    content=stream,
                    writer=writer,
                )
            except Exception as exc:
                writer.close()
                try:
                    await writer.wait_closed()
                except Exception:
                    pass
                if isinstance(exc, (StdlibClientError, asyncio.TimeoutError)):
                    raise
                raise StdlibClientError(str(exc)) from exc

        raise StdlibClientError(f"Too many redirects for {url}")

    async def close(self) -> None:
        self.closed = True

    async def __aenter__(self) -> "StdlibClientSession":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()


class _StdlibAiohttpModule:
    """Namespace shim mirroring the ``aiohttp`` classes used by ``agentu``."""

    ClientSession = StdlibClientSession
    ClientResponse = StdlibClientResponse
    ClientTimeout = StdlibClientTimeout
    ClientError = StdlibClientError
    ClientResponseError = StdlibClientResponseError


stdlib_aiohttp = _StdlibAiohttpModule()

try:
    import aiohttp as _native_aiohttp  # type: ignore
    if _native_aiohttp is None:  # e.g. sys.modules['aiohttp'] = None in zero-dep tests
        raise ImportError
    aiohttp: Any = _native_aiohttp
    HAS_NATIVE_AIOHTTP = True
except ImportError:
    aiohttp = stdlib_aiohttp
    HAS_NATIVE_AIOHTTP = False
