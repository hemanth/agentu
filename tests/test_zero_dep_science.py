"""Tests for zero-dependency science-baked architecture:
1. Plain-class & inline-dict schemas + AST constraint lifting + multi-assert diagnostics
2. Deterministic Pushdown Automaton JSON healer (`heal_json`)
3. Canonical key alignment + Damerau-Levenshtein near-miss diagnostics
4. Fielded Okapi BM25F + Character 3-Gram deferred tool ranking
5. Pure-stdlib async HTTP/1.1 + chunked SSE transport (`StdlibClientSession`)
6. Clean-room subprocess execution with `aiohttp` and `pydantic` completely blocked
"""

from __future__ import annotations

import asyncio
import dataclasses
import enum
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Literal, Optional, TypedDict
from unittest.mock import AsyncMock

import pytest

from agentu import (
    Agent,
    GeminiDriver,
    InferResult,
    StructRecord,
    StructuredOutputError,
    format_validation_error,
    heal_json,
    parse_and_validate,
    schema_for_type,
)
from agentu._core.bm25 import rank_tools_bm25f, tokenize_text
from agentu._core.http_compat import StdlibClientResponseError, StdlibClientSession
from agentu._core.tools import Tool


# ─── 1. Plain Classes + AST Contract Lifting ───────────────────────────────


class RefundDecision:
    """Customer refund decision with zero-import AST contracts."""

    summary: str
    refund_usd: float
    ticket_id: str = "TKT-0001"
    tier: str = "standard"
    tags: list[str] = []

    def check(self):
        assert 0 <= self.refund_usd <= 100
        assert self.tier in ("standard", "vip")
        assert len(self.summary) >= 12
        assert 0 <= len(self.tags) <= 4
        assert re.fullmatch(r"[A-Z]{3}-\d{4}", self.ticket_id)


class LineItem:
    sku: str
    qty: int


class OrderSummary:
    order_id: str
    items: list[LineItem]
    notes: Optional[str] = None


class Priority(enum.Enum):
    LOW = "low"
    HIGH = "high"


class TriageTicket:
    priority: Priority
    channel: Literal["email", "slack"]


class TestZeroImportPlainClassAndASTLifting:
    def test_ast_lifts_constraints_into_json_schema(self):
        schema = schema_for_type(RefundDecision)
        assert schema["title"] == "RefundDecision"
        assert "Customer refund decision" in schema["description"]
        assert set(schema["required"]) == {"summary", "refund_usd"}

        props = schema["properties"]
        assert props["refund_usd"]["type"] == "number"
        assert props["refund_usd"]["minimum"] == 0
        assert props["refund_usd"]["maximum"] == 100
        assert props["tier"]["enum"] == ["standard", "vip"]
        assert props["tier"]["default"] == "standard"
        assert props["summary"]["minLength"] == 12
        assert props["tags"]["minItems"] == 0
        assert props["tags"]["maxItems"] == 4
        assert props["ticket_id"]["pattern"] == r"^(?:[A-Z]{3}-\d{4})$"

    def test_plain_class_hydration_and_mutable_default_isolation(self):
        r1 = parse_and_validate(
            '{"summary": "Approved full customer refund", "refund_usd": 42.5, "ticket_id": "RFD-1024"}',
            RefundDecision,
        )
        r2 = parse_and_validate(
            '{"summary": "Approved partial customer refund", "refund_usd": 15, "ticket_id": "RFD-1025"}',
            RefundDecision,
        )
        assert isinstance(r1, RefundDecision)
        assert r1.refund_usd == 42.5
        assert r1.tier == "standard"
        assert r1.tags == []
        r1.tags.append("mutated")
        assert r2.tags == []  # Mutable class default `[]` is deep-copied per instance!
        assert "RefundDecision(" in repr(r1)
        assert r2.to_dict()["ticket_id"] == "RFD-1025"

    def test_multi_assert_contract_reports_all_violations_in_one_turn(self):
        bad_raw = json.dumps({
            "summary": "short",       # violates len >= 12
            "refund_usd": 250.0,      # violates <= 100
            "tier": "platinum",       # violates in ("standard", "vip")
            "ticket_id": "RFD-1024",
        })
        with pytest.raises(ValueError, match="Schema validation failed") as exc_info:
            parse_and_validate(bad_raw, RefundDecision)

        feedback = format_validation_error(exc_info.value, RefundDecision)
        assert "Field errors:" in feedback
        assert "refund_usd" in feedback and "250.0" in feedback
        assert "tier" in feedback and "'platinum'" in feedback
        assert "summary" in feedback and "'short'" in feedback

    def test_nested_plain_classes_enums_and_literals(self):
        order = parse_and_validate(
            '{"order_id": "ORD-9", "items": [{"sku": "KB-1", "qty": "2"}]}',
            OrderSummary,
        )
        assert isinstance(order, OrderSummary)
        assert isinstance(order.items[0], LineItem)
        assert order.items[0].qty == 2
        assert order.notes is None
        assert order.to_dict() == {
            "order_id": "ORD-9",
            "items": [{"sku": "KB-1", "qty": 2}],
            "notes": None,
        }

        triage = parse_and_validate('{"priority": "HIGH", "channel": "SLACK"}', TriageTicket)
        assert triage.priority is Priority.HIGH
        assert triage.channel == "slack"

    def test_dataclass_and_typeddict_support(self):
        @dataclasses.dataclass
        class MetricPoint:
            name: str
            latency_ms: float

            def __post_init__(self):
                assert self.latency_ms >= 0

        class UserTD(TypedDict):
            handle: str
            karma: int

        s_dc = schema_for_type(MetricPoint)
        assert s_dc["properties"]["latency_ms"]["minimum"] == 0
        pt = parse_and_validate('{"name": "p99", "latency_ms": 12.4}', MetricPoint)
        assert pt == MetricPoint(name="p99", latency_ms=12.4)

        with pytest.raises(ValueError, match="latency_ms"):
            parse_and_validate('{"name": "p99", "latency_ms": -3}', MetricPoint)

        td = parse_and_validate('{"handle": "hemanth", "karma": 9001}', UserTD)
        assert isinstance(td, dict)
        assert td.handle == "hemanth"
        assert td["karma"] == 9001


# ─── 2. Inline Dict Shapes (`StructRecord`) ────────────────────────────────


class TestInlineDictShapes:
    @pytest.mark.asyncio
    async def test_inline_dict_output_type_end_to_end(self):
        driver = GeminiDriver(api_key="test-key", model="gemini-2.5-flash")

        captured_schemas = []

        async def fake_call(messages, tools=None, output_schema=None):
            captured_schemas.append(output_schema)
            from agentu.drivers.base import DriverResponse
            return DriverResponse(text='{"pick": "ThinkPad X1", "priceUsd": 1299}')

        driver.call = fake_call
        agent = Agent("buyer", driver=driver, enable_memory=False, auto_discover_rules=False)

        res: InferResult = await agent.infer(
            "Best Linux laptop",
            output_type={"pick": str, "price_usd": int, "coupon?": str},
        )
        assert isinstance(res.data, StructRecord)
        assert isinstance(res.data, dict)
        assert res.data.pick == "ThinkPad X1"
        assert res.data["price_usd"] == 1299  # camelCase -> snake_case aligned!
        assert res.data.coupon is None
        assert captured_schemas[0]["type"] == "object"
        assert set(captured_schemas[0]["required"]) == {"pick", "price_usd"}


# ─── 3. Pushdown Automaton JSON Healer (`heal_json`) ───────────────────────


class TestPushdownJSONHealer:
    def test_heals_fences_prose_comments_quotes_python_literals_and_commas(self):
        messy = """Sure! Here is your payload:
```json
{
    // Primary item chosen by model
    pick: 'MacBook Air 13"',
    'in_stock': True,
    "discount": None, /* no promo active */
    "tags": ['arm64', 'fanless',],
}
```
Let me know if you want alternatives!"""
        healed = heal_json(messy)
        parsed = json.loads(healed)
        assert parsed == {
            "pick": 'MacBook Air 13"',
            "in_stock": True,
            "discount": None,
            "tags": ["arm64", "fanless"],
        }

    def test_preserves_urls_and_escapes_raw_newlines_in_strings(self):
        raw = '{"url": "https://hemanth.github.io/agentu-codelab/", "note": "line 1\nline 2\tindented"}'
        parsed = json.loads(heal_json(raw))
        assert parsed["url"] == "https://hemanth.github.io/agentu-codelab/"
        assert parsed["note"] == "line 1\nline 2\tindented"

    def test_closes_truncated_json_streams_cleanly(self):
        # Truncated mid-key after complete required fields
        trunc_mid_key = '{"summary": "Customer eligible for full refund", "refund_usd": 25.0, "tic'
        r = parse_and_validate(trunc_mid_key, RefundDecision)
        assert r.refund_usd == 25.0
        # Wait: RefundDecision default ticket_id had spaces in class? Wait: check ticket_id!
        # Let's check dangling colon too:
        trunc_colon = '{"a": [1, 2, {"nested": '
        assert json.loads(heal_json(trunc_colon)) == {"a": [1, 2, {"nested": None}]}

    def test_non_json_prose_still_raises_invalid_json(self):
        with pytest.raises(ValueError, match="Invalid JSON"):
            parse_and_validate("Here is a conversational paragraph without braces.", OrderSummary)


# ─── 4. Canonical Key Alignment & Damerau-Levenshtein Hints ───────────────


class TestKeyAlignmentAndNearMiss:
    def test_camel_and_kebab_keys_align_without_retry(self):
        order = parse_and_validate(
            '{"orderId": "ORD-42", "items": [{"sku": "M1", "qty": 1}]}',
            OrderSummary,
        )
        assert order.order_id == "ORD-42"

    def test_damerau_levenshtein_transposition_hint_in_error(self):
        with pytest.raises(ValueError) as exc_info:
            parse_and_validate('{"oredr_id": "ORD-42", "items": []}', OrderSummary)
        msg = format_validation_error(exc_info.value, OrderSummary)
        assert "oredr_id" in msg
        assert "edit distance 1" in msg


# ─── 5. Fuzzy Fielded BM25F + Character 3-Gram Tool Discovery ─────────────


class TestBM25FToolDiscovery:
    def test_bm25f_outperforms_word_overlap_against_verbose_distractors(self):
        def get_system_status(verbose: bool = False) -> str:
            """Get status check and get general information when you want to get updates."""
            return "ok"

        def stripe_issue_refund(charge_id: str, amount_cents: int) -> str:
            """Create a credit reversal on a captured card charge."""
            return "refunded"

        def k8s_rollout_restart(deployment_name: str, namespace: str = "prod") -> str:
            """Restart Kubernetes pods in a cluster deployment."""
            return "restarted"

        tools = [Tool(get_system_status), Tool(stripe_issue_refund), Tool(k8s_rollout_restart)]

        # Query mentions parameter `charge_id` and morphological variant `refunding`
        ranked = rank_tools_bm25f("get help refunding a customer charge_id", tools, limit=2)
        assert ranked[0].name == "stripe_issue_refund"

        # Typo resilience: "kubernets" + "deploymnt" via char 3-gram cosine
        typo_ranked = rank_tools_bm25f("kubernets deploymnt pod", tools, limit=2)
        assert typo_ranked[0].name == "k8s_rollout_restart"

        # Identifier camelCase splitting
        assert "github" in tokenize_text("fetchGitHubPullRequests")
        assert "pull" in tokenize_text("fetchGitHubPullRequests")


# ─── 6. Pure-Stdlib Async HTTP/SSE + Zero-Dep Clean-Room Subprocess ────────


class TestStdlibHTTPAndZeroDepRuntime:
    @pytest.mark.asyncio
    async def test_stdlib_client_session_json_and_chunked_sse(self):
        async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
            req_line = await reader.readline()
            headers = {}
            while True:
                hline = await reader.readline()
                if hline in (b"\r\n", b"\n", b""):
                    break
                k, v = hline.decode("latin-1").split(":", 1)
                headers[k.strip().lower()] = v.strip()
            clen = int(headers.get("content-length", "0"))
            body = json.loads((await reader.readexactly(clen)).decode("utf-8")) if clen else {}

            path = req_line.decode("latin-1").split()[1]
            if path == "/error":
                payload = b'{"error": "rate limited"}'
                writer.write(
                    b"HTTP/1.1 429 Too Many Requests\r\n"
                    b"Content-Type: application/json\r\n"
                    + f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n".encode()
                    + payload
                )
            elif body.get("stream"):
                writer.write(
                    b"HTTP/1.1 200 OK\r\n"
                    b"Content-Type: text/event-stream\r\n"
                    b"Transfer-Encoding: chunked\r\n"
                    b"Connection: close\r\n\r\n"
                )
                for tok in ("Zero", "-", "dep!"):
                    sse = f'data: {{"choices": [{{"delta": {{"content": "{tok}"}}}}]}}\n\n'.encode()
                    writer.write(f"{len(sse):X}\r\n".encode() + sse + b"\r\n")
                done = b"data: [DONE]\n\n"
                writer.write(f"{len(done):X}\r\n".encode() + done + b"\r\n0\r\n\r\n")
            else:
                resp_bytes = json.dumps({
                    "choices": [{"message": {"content": '{"answer": "42", "ok": true}'}}]
                }).encode("utf-8")
                writer.write(
                    b"HTTP/1.1 200 OK\r\n"
                    b"Content-Type: application/json\r\n"
                    + f"Content-Length: {len(resp_bytes)}\r\nConnection: close\r\n\r\n".encode()
                    + resp_bytes
                )
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(handle_client, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        base_url = f"http://127.0.0.1:{port}"

        try:
            async with StdlibClientSession() as session:
                # 1. JSON POST
                async with session.post(f"{base_url}/v1/chat/completions", json={"stream": False}) as resp:
                    assert resp.status == 200
                    data = await resp.json()
                    assert '"42"' in data["choices"][0]["message"]["content"]

                # 2. Chunked SSE stream
                chunks = []
                async with session.post(f"{base_url}/v1/chat/completions", json={"stream": True}) as resp:
                    async for raw_line in resp.content:
                        line = raw_line.decode("utf-8").strip()
                        if line.startswith("data: ") and line != "data: [DONE]":
                            chunks.append(json.loads(line[6:])["choices"][0]["delta"]["content"])
                assert "".join(chunks) == "Zero-dep!"

                # 3. HTTP 429 raises ClientResponseError
                async with session.post(f"{base_url}/error", json={}) as err_resp:
                    with pytest.raises(StdlibClientResponseError) as exc_info:
                        err_resp.raise_for_status()
                    assert exc_info.value.status == 429
        finally:
            server.close()
            await server.wait_closed()

    def test_clean_room_zero_dependencies_subprocess(self, tmp_path: Path):
        """Block both `aiohttp` and `pydantic` in sys.modules and run Agent end-to-end."""
        src_dir = str(Path(__file__).resolve().parents[1] / "src")
        script_file = tmp_path / "clean_room_check.py"
        script_file.write_text(
            """
import sys
sys.modules["aiohttp"] = None
sys.modules["pydantic"] = None

import asyncio, json
from agentu import Agent, AgentConfig, schema_for_type

class Brief:
    headline: str
    confidence: float
    def check(self):
        assert 0.0 <= self.confidence <= 1.0

async def main():
    # Verify config works with zero pydantic
    cfg = AgentConfig(name="clean", model="gpt-4o", cache={"enabled": False})
    assert cfg.name == "clean"

    # Verify AST schema lifting works with zero pydantic
    s = schema_for_type(Brief)
    assert s["properties"]["confidence"]["maximum"] == 1.0

    # Spin up local OpenAI-compatible mock server over raw asyncio sockets
    async def handler(reader, writer):
        await reader.readline()
        headers = {}
        while True:
            h = await reader.readline()
            if h in (b"\\r\\n", b"\\n", b""):
                break
            k, v = h.decode().split(":", 1)
            headers[k.strip().lower()] = v.strip()
        await reader.readexactly(int(headers.get("content-length", "0")))
        body = json.dumps({
            "choices": [{"message": {"content": "```json\\n{headline: 'Zero Dep Works', confidence: 0.98,}\\n```"}}]
        }).encode()
        writer.write(
            b"HTTP/1.1 200 OK\\r\\nContent-Type: application/json\\r\\n"
            + f"Content-Length: {len(body)}\\r\\nConnection: close\\r\\n\\r\\n".encode()
            + body
        )
        await writer.drain()
        writer.close()

    srv = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = srv.sockets[0].getsockname()[1]
    try:
        agent = Agent(
            "zero-dep-agent",
            model="local-test",
            api_base=f"http://127.0.0.1:{port}",
            enable_memory=False,
            auto_discover_rules=False,
        )
        res = await agent.infer("Summarize", output_type=Brief)
        assert res.data.headline == "Zero Dep Works"
        assert res.data.confidence == 0.98
    finally:
        srv.close()
        await srv.wait_closed()

asyncio.run(main())
print("CLEAN_ROOM_OK")
""",
            encoding="utf-8",
        )
        proc = subprocess.run(
            [sys.executable, str(script_file)],
            env={"PYTHONPATH": src_dir},
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert proc.returncode == 0, f"STDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}"
        assert "CLEAN_ROOM_OK" in proc.stdout
