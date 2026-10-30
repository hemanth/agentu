#!/usr/bin/env python3
"""Reproducible empirical benchmarks for zero-dependency science-baked agentu.

Measures:
1. Pushdown Automaton JSON Healer + Canonical Key Alignment vs Baseline JSON parser
   across a 24-case malformed LLM output corpus (0-retry recovery rate + us latency).
2. Fuzzy Fielded Okapi BM25F + Char 3-Gram vs Naive Word-Overlap baseline
   across a 50-tool enterprise catalog x 30 queries (Recall@1, Recall@3, MRR, us/query).
3. Zero-dependency cold-start import latency & module count vs Pydantic + aiohttp.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agentu._core.bm25 import rank_tools_bm25f
from agentu._core.structured import heal_json, parse_and_validate, schema_for_type
from agentu._core.tools import Tool


# ─── 1. Structured Output Corpus & Target Class ────────────────────────────


class RefundDecision:
    """Customer refund decision."""

    summary: str
    refund_usd: float
    tier: str = "standard"
    approved: bool = True

    def check(self):
        assert 0 <= self.refund_usd <= 500
        assert self.tier in ("standard", "vip", "enterprise")
        assert len(self.summary) >= 8


def _legacy_parse(raw: str) -> Dict[str, Any]:
    """Baseline parser from agentu <= 2.10.1 (code-fence strip + strict json.loads + exact keys)."""
    text = raw.strip()
    if text.startswith("```"):
        lines = [ln for ln in text.split("\n") if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()
    data = json.loads(text)
    if "summary" not in data or "refund_usd" not in data:
        raise ValueError("Missing required exact snake_case fields")
    return data


MALFORMED_CORPUS: List[Tuple[str, str]] = [
    ("clean_json", '{"summary": "Approved standard refund", "refund_usd": 45.0, "tier": "standard", "approved": true}'),
    ("markdown_fenced", '```json\n{"summary": "Approved standard refund", "refund_usd": 45.0}\n```'),
    ("prose_wrapped", 'Sure! Here is the JSON payload:\n{"summary": "Approved standard refund", "refund_usd": 45.0}\nLet me know if you need edits.'),
    ("fenced_plus_prose", 'Analysis complete:\n```json\n{"summary": "Approved VIP credit", "refund_usd": 120.0, "tier": "vip"}\n```\nDone!'),
    ("trailing_comma_obj", '{"summary": "Approved standard refund", "refund_usd": 45.0, "tier": "standard",}'),
    ("trailing_comma_nested", '{\n  "summary": "Approved standard refund",\n  "refund_usd": 45.0,\n}'),
    ("line_comments", '{\n  // Evaluated against policy v4\n  "summary": "Approved standard refund",\n  "refund_usd": 45.0 // full amount\n}'),
    ("block_comments", '{/* header */ "summary": "Approved standard refund", /* inline */ "refund_usd": 45.0}'),
    ("single_quoted_all", "{'summary': 'Approved standard refund', 'refund_usd': 45.0, 'tier': 'vip'}"),
    ("single_quoted_inner_dbl", "{'summary': 'Approved \"priority\" refund', 'refund_usd': 80.0}"),
    ("unquoted_keys", '{summary: "Approved standard refund", refund_usd: 45.0, tier: "standard"}'),
    ("python_booleans", '{"summary": "Approved standard refund", "refund_usd": 45.0, "approved": True}'),
    ("python_false_mixed", '{summary: "Declined zero refund", refund_usd: 0.0, approved: False,}'),
    ("raw_newline_in_str", '{"summary": "Approved refund\nVerified receipt", "refund_usd": 39.99}'),
    ("raw_tab_in_str", '{"summary": "Approved:\tfull item refund", "refund_usd": 19.0}'),
    ("invalid_single_quote_esc", '{"summary": "Customer\\\'s item arrived damaged", "refund_usd": 64.5}'),
    ("camel_case_keys", '{"summary": "Approved camelCase payload", "refundUsd": 55.0, "tier": "vip"}'),
    ("kebab_case_keys", '{"summary": "Approved kebab-case payload", "refund-usd": 55.0}'),
    ("case_folded_literal_num_str", '{"summary": "Numeric string coerced cleanly", "refund_usd": "72.5", "approved": "true"}'),
    ("truncated_after_fields", '{"summary": "Approved before max_tokens", "refund_usd": 30.0'),
    ("truncated_trailing_comma", '{"summary": "Approved before max_tokens", "refund_usd": 30.0, '),
    ("truncated_mid_optional_key", '{"summary": "Approved before max_tokens", "refund_usd": 30.0, "tie'),
    ("truncated_mid_string_val", '{"refund_usd": 30.0, "summary": "Approved customer refund before cut'),
    ("prose_comments_python_combo", 'Result:\n{ // hybrid defect\n  summary: \'Refunded order #91\', refundUsd: 99.0, approved: True,\n}\nThanks!'),
]


def bench_pda_repair(iters: int = 1000) -> Dict[str, Any]:
    legacy_ok = 0
    pda_ok = 0
    for _name, sample in MALFORMED_CORPUS:
        try:
            _legacy_parse(sample)
            legacy_ok += 1
        except Exception:
            pass
        try:
            obj = parse_and_validate(sample, RefundDecision)
            assert isinstance(obj, RefundDecision)
            pda_ok += 1
        except Exception as exc:
            raise AssertionError(f"Corpus case {_name} failed: {exc}") from exc

    latencies_us: List[float] = []
    for _ in range(iters):
        for _, sample in MALFORMED_CORPUS:
            t0 = time.perf_counter_ns()
            parse_and_validate(sample, RefundDecision)
            latencies_us.append((time.perf_counter_ns() - t0) / 1000.0)

    latencies_us.sort()
    p50 = latencies_us[len(latencies_us) // 2]
    p95 = latencies_us[int(len(latencies_us) * 0.95)]

    t0 = time.perf_counter_ns()
    for _ in range(iters):
        schema_for_type(RefundDecision)
    schema_us = ((time.perf_counter_ns() - t0) / 1000.0) / iters

    total = len(MALFORMED_CORPUS)
    return {
        "total_cases": total,
        "legacy_recovered": legacy_ok,
        "legacy_rate_pct": round(100.0 * legacy_ok / total, 1),
        "pda_recovered": pda_ok,
        "pda_rate_pct": round(100.0 * pda_ok / total, 1),
        "p50_us": round(p50, 2),
        "p95_us": round(p95, 2),
        "ast_schema_compile_us": round(schema_us, 2),
    }


# ─── 2. 50-Tool Enterprise Catalog x 30 Queries (BM25F vs Word Overlap) ────


def _make_stub_tool(name: str, desc: str, params: Dict[str, str]) -> Tool:
    def _fn(**kwargs):
        return kwargs

    t = Tool(_fn, name=name, description=desc)
    t.parameters = {
        "type": "object",
        "properties": {k: {"type": "string", "description": v} for k, v in params.items()},
    }
    return t


ENTERPRISE_TOOLS: List[Tuple[str, str, Dict[str, str]]] = [
    # Verbose distractors rich in common verbs ("get", "fetch", "create", "update", "status", "user", "data")
    ("get_service_health_status", "Get current system status, fetch service status data, and check user API status updates.", {"service_name": "Target microservice"}),
    ("fetch_audit_trail_events", "Fetch user activity data and get compliance status logs from the central event bus.", {"actor_id": "User identifier", "since_iso": "Start timestamp"}),
    ("list_workspace_members", "Get workspace user list, fetch member profile data, and check user status.", {"workspace_id": "Workspace UUID"}),
    ("update_account_preferences", "Update user settings data, get notification status, and save profile preferences.", {"user_id": "Account ID", "timezone": "IANA zone"}),
    ("create_support_escalation", "Create a new customer support ticket and update user escalation status.", {"customer_email": "Contact email", "severity": "P0-P3"}),
    # Payments & Billing
    ("stripe_issue_refund", "Reverse a captured credit card charge and credit funds back to the buyer.", {"charge_id": "Stripe ch_ identifier", "amount_cents": "Refund in cents"}),
    ("stripe_create_invoice", "Draft and finalize a recurring B2B customer billing statement.", {"customer_id": "Stripe cus_ ID", "due_days": "Net terms"}),
    ("stripe_list_disputes", "Retrieve open chargebacks and cardholder fraud inquiries.", {"status_filter": "Dispute state"}),
    ("calculate_vat_withholding", "Compute European value-added tax rates by buyer country code.", {"country_iso": "Two-letter ISO country", "net_eur": "Amount in EUR"}),
    ("wire_transfer_payout", "Initiate an international SWIFT or ACH bank settlement.", {"iban": "Destination IBAN", "routing_number": "ABA routing"}),
    # Kubernetes & Cloud Infra
    ("k8s_rollout_restart", "Gracefully cycle pods inside a Kubernetes Deployment.", {"deployment_name": "Deployment resource", "namespace": "Cluster namespace"}),
    ("k8s_pod_tail_logs", "Stream stdout container logs from a running pod replica.", {"pod_name": "Pod identifier", "tail_lines": "Number of lines"}),
    ("k8s_scale_replicas", "Adjust horizontal replica count on a StatefulSet or Deployment.", {"resource_name": "Target workload", "replicas": "Desired pod count"}),
    ("terraform_plan_diff", "Preview infrastructure-as-code state changes without applying.", {"module_path": "HCL directory", "var_file": "tfvars path"}),
    ("aws_s3_presign_url", "Mint a temporary signed download link for a private bucket object.", {"bucket_name": "S3 bucket", "object_key": "Key path"}),
    ("aws_iam_rotate_keys", "Deactivate expiring IAM access keys and provision a fresh secret pair.", {"iam_username": "Principal name"}),
    ("cloudflare_purge_cdn", "Invalidate cached edge assets across global POPs by URL prefix.", {"zone_id": "Cloudflare zone", "url_prefix": "Purge path"}),
    # Databases & Analytics
    ("postgresql_explain_query", "Run EXPLAIN ANALYZE on a read-only SQL statement to inspect index scans.", {"sql_statement": "SELECT query", "dsn_alias": "Replica alias"}),
    ("redis_evict_pattern", "Scan and delete cached keys matching a glob prefix using UNLINK.", {"key_glob": "Redis pattern", "db_index": "Logical DB"}),
    ("bigquery_cost_dryrun", "Estimate bytes scanned and slot cost before executing a warehouse SQL job.", {"project_id": "GCP project", "query_sql": "Standard SQL"}),
    ("clickhouse_merge_parts", "Inspect merge tree partition mutations and replica lag.", {"table_name": "MergeTree table"}),
    ("snowflake_warehouse_resume", "Wake a suspended virtual compute warehouse.", {"warehouse_name": "Compute cluster"}),
    # Git & CI/CD (camelCase / PascalCase / snake_case mix)
    ("fetchGitHubPullRequest", "Retrieve diff stats, review comments, and mergeability for a repository PR.", {"repo_slug": "owner/name", "pull_number": "PR integer"}),
    ("triggerGitHubWorkflowRun", "Dispatch a GitHub Actions workflow YAML on a target git branch.", {"workflow_file": "YAML filename", "git_ref": "Branch or tag"}),
    ("gitCherryPickCommit", "Apply a specific commit SHA onto a release stabilization branch.", {"commit_sha": "40-char hash", "target_branch": "Release branch"}),
    ("sentryResolveIssue", "Mark anproduction exception group as resolved in Sentry.", {"issue_short_id": "Sentry group ID"}),
    ("datadogQueryTimeseries", "Fetch p99 latency or CPU metric series over a time window.", {"metric_expr": "Datadog query", "window_mins": "Lookback"}),
    ("pagerdutyAckIncident", "Acknowledge an active on-call alert to halt escalation pages.", {"incident_id": "PD incident"}),
    # Messaging, Docs & Security
    ("slackPostThreadReply", "Send a formatted Block Kit reply inside an existing channel thread.", {"channel_id": "C-prefixed channel", "thread_ts": "Parent timestamp"}),
    ("sendgrid_transactional_email", "Deliver an HTML email template via SMTP relay.", {"recipient_address": "To email", "template_id": "Dynamic template"}),
    ("jira_transition_issue", "Move an engineering issue card across workflow board columns.", {"issue_key": "PROJ-123 key", "target_column": "Status name"}),
    ("notion_append_callout", "Insert a markdown block at the bottom of a Notion wiki page.", {"page_uuid": "32-char page ID", "markdown_body": "Content"}),
    ("linear_assign_cycle", "Attach an engineering task to the upcoming two-week sprint cycle.", {"ticket_id": "LIN identifier", "cycle_number": "Sprint number"}),
    ("vault_read_kv_secret", "Fetch an encrypted secret version from HashiCorp Vault KV v2.", {"mount_path": "KV engine mount", "secret_path": "Path"}),
    ("trivy_scan_container", "Audit an OCI Docker image manifest for CVE vulnerabilities.", {"image_digest": "sha256 image ref"}),
    ("semgrep_sast_audit", "Run static taint analysis rules over modified source files.", {"ruleset": "OWASP pack", "target_dir": "Source folder"}),
    ("whois_domain_expiry", "Check registrar expiration date and nameserver delegation for a domain.", {"domain_name": "FQDN"}),
    ("tls_cert_chain_check", "Verify X.509 certificate expiration and SAN hostnames on port 443.", {"hostname": "TLS host"}),
    ("dns_dig_records", "Resolve A, AAAA, MX, and TXT records from authoritative nameservers.", {"fqdn": "Domain", "record_type": "RR type"}),
    ("pdf_extract_tables", "Parse tabular financial rows out of a multi-page PDF filing.", {"pdf_path": "Local path"}),
    ("ffmpeg_transcode_h264", "Re-encode raw video footage into web-optimized MP4 H.264.", {"input_media": "Source file", "crf": "Quality"}),
    ("whisper_transcribe_audio", "Convert spoken WAV or MP3 recordings into timestamped text.", {"audio_file": "Path to audio"}),
    ("playwright_capture_har", "Record browser network waterfall Archive from a headless page load.", {"page_url": "Target URL"}),
    ("lighthouse_cwv_score", "Measure Largest Contentful Paint and CLS performance metrics.", {"target_url": "Webpage URL"}),
    ("openapi_lint_spec", "Validate a Swagger / OpenAPI 3.1 YAML schema for breaking changes.", {"spec_file": "YAML path"}),
    ("graphql_introspect_schema", "Fetch query and mutation types from a GraphQL endpoint.", {"endpoint_url": "GraphQL URL"}),
    ("cron_next_occurrences", "Compute the next five UTC fire timestamps for a 5-field cron expression.", {"cron_expr": "Cron string"}),
    ("jwt_verify_rs256", "Validate token signature, issuer, and exp claim against a JWKS keyset.", {"id_token": "Bearer JWT", "jwks_uri": "Keyset URL"}),
    ("oauth_exchange_pkce", "Swap an authorization code and PKCE verifier for access tokens.", {"auth_code": "Callback code", "code_verifier": "S256 verifier"}),
    ("csv_pivot_aggregate", "Group a CSV dataset by column and compute sum/median statistics.", {"csv_file": "Dataset path", "group_col": "Column"}),
]


TOOL_QUERIES: List[Tuple[str, str]] = [
    # Conversational queries with high-frequency verbs/nouns ("get", "user", "status", "update", "data", "fetch")
    ("please get user data and status to refund charge_id", "stripe_issue_refund"),
    ("get current status and fetch review updates on pull_number", "fetchGitHubPullRequest"),
    ("fetch user workflow_file and get status on git_ref", "triggerGitHubWorkflowRun"),
    ("update release target_branch by cherry-picking commit_sha", "gitCherryPickCommit"),
    ("fetch current metric_expr data for window_mins", "datadogQueryTimeseries"),
    ("update status to acknowledge incident_id on call", "pagerdutyAckIncident"),
    ("post user update message to channel_id thread_ts", "slackPostThreadReply"),
    ("update status of production issue_short_id", "sentryResolveIssue"),
    # Parameter-schema driven queries (where domain term lives in parameter schema)
    ("initiate bank payout to destination iban and routing_number", "wire_transfer_payout"),
    ("get temporary access for bucket_name and object_key", "aws_s3_presign_url"),
    ("clear cached entries matching key_glob in db_index", "redis_evict_pattern"),
    ("get cost estimate for project_id and query_sql", "bigquery_cost_dryrun"),
    ("update issue_key to done target_column", "jira_transition_issue"),
    ("check bearer id_token signature using jwks_uri", "jwt_verify_rs256"),
    ("exchange callback auth_code and s256 code_verifier", "oauth_exchange_pkce"),
    ("compute tax for country_iso and net_eur", "calculate_vat_withholding"),
    # Typos & character transpositions (character 3-gram resilience)
    ("kubernets deploymnt pod restart", "k8s_rollout_restart"),
    ("postgersql explian index scan", "postgresql_explain_query"),
    ("terrafrom infrastructur state diff", "terraform_plan_diff"),
    ("cloudflar purg cdn assets", "cloudflare_purge_cdn"),
    ("snowflak resum warehous cluster", "snowflake_warehouse_resume"),
    ("whispr transcrib audio file", "whisper_transcribe_audio"),
    ("lighthous contentful paint cwv", "lighthouse_cwv_score"),
    ("trivy containr vulnerabilites cve", "trivy_scan_container"),
    # Direct engineering intents with morphological inflections
    ("tailing container logs from running pods", "k8s_pod_tail_logs"),
    ("rotating expiring iam_username keys", "aws_iam_rotate_keys"),
    ("checking x509 certificate hostname expiry", "tls_cert_chain_check"),
    ("resolving domain fqdn dns record_type", "dns_dig_records"),
    ("fetching secret_path from mount_path", "vault_read_kv_secret"),
    ("computing upcoming timestamps for cron_expr", "cron_next_occurrences"),
]


def _naive_word_overlap(query: str, tools: List[Tool], limit: int = 5) -> List[Tool]:
    """Exact baseline from agentu <= 2.10.1 (`Agent._find_matching_tools`)."""
    query_terms = query.lower().split()
    scored = []
    for idx, tool in enumerate(tools):
        text = f"{tool.name} {tool.description}".lower()
        score = sum(1 for term in query_terms if term in text)
        if score > 0:
            scored.append((score, -idx, tool))
    scored.sort(reverse=True, key=lambda x: (x[0], x[1]))
    return [t for _, _, t in scored[:limit]]


def _eval_ranker(rank_fn: Callable[[str, List[Tool], int], List[Tool]], tools: List[Tool]) -> Dict[str, float]:
    # Warm cache once (mirrors steady-state agent loop)
    for query, _ in TOOL_QUERIES[:3]:
        rank_fn(query, tools, 5)

    hits_at_1 = 0
    hits_at_3 = 0
    rr_sum = 0.0
    latencies_us: List[float] = []

    for query, expected_name in TOOL_QUERIES:
        t0 = time.perf_counter_ns()
        ranked = rank_fn(query, tools, 5)
        latencies_us.append((time.perf_counter_ns() - t0) / 1000.0)

        names = [t.name for t in ranked]
        if names and names[0] == expected_name:
            hits_at_1 += 1
        if expected_name in names[:3]:
            hits_at_3 += 1
        if expected_name in names:
            rr_sum += 1.0 / (names.index(expected_name) + 1)

    n = len(TOOL_QUERIES)
    return {
        "recall_at_1_pct": round(100.0 * hits_at_1 / n, 1),
        "recall_at_3_pct": round(100.0 * hits_at_3 / n, 1),
        "mrr": round(rr_sum / n, 3),
        "median_us": round(statistics.median(latencies_us), 1),
    }


def bench_tool_discovery() -> Dict[str, Any]:
    catalog = [_make_stub_tool(n, d, p) for n, d, p in ENTERPRISE_TOOLS]
    baseline = _eval_ranker(_naive_word_overlap, catalog)
    bm25f = _eval_ranker(lambda q, ts, lim: rank_tools_bm25f(q, ts, limit=lim), catalog)
    return {
        "catalog_size": len(catalog),
        "num_queries": len(TOOL_QUERIES),
        "naive_word_overlap": baseline,
        "fuzzy_bm25f": bm25f,
    }


# ─── 3. Cold-Start Import Latency & Module Footprint ───────────────────────


def _measure_subprocess_import(code: str, runs: int = 7) -> Tuple[float, int]:
    samples_ms: List[float] = []
    mod_count = 0
    for _ in range(runs):
        out = subprocess.check_output(
            [sys.executable, "-c", code],
            env={"PYTHONPATH": str(SRC)},
            text=True,
        ).strip()
        ms_str, cnt_str = out.split(",")
        samples_ms.append(float(ms_str))
        mod_count = int(cnt_str)
    return round(statistics.median(samples_ms), 2), mod_count


def bench_cold_start() -> Dict[str, Any]:
    zero_dep_probe = (
        "import sys, time; "
        "sys.modules['aiohttp'] = None; sys.modules['pydantic'] = None; "
        "t0 = time.perf_counter(); "
        "import agentu; from agentu import Agent, schema_for_type, heal_json; "
        "ms = (time.perf_counter() - t0) * 1000; "
        "print(f'{ms:.3f},{len(sys.modules)}')"
    )
    heavy_stack_probe = (
        "import sys, time; "
        "t0 = time.perf_counter(); "
        "import pydantic, aiohttp, agentu; from agentu import Agent; "
        "ms = (time.perf_counter() - t0) * 1000; "
        "print(f'{ms:.3f},{len(sys.modules)}')"
    )
    zero_ms, zero_mods = _measure_subprocess_import(zero_dep_probe)
    heavy_ms, heavy_mods = _measure_subprocess_import(heavy_stack_probe)
    return {
        "zero_dep_import_ms": zero_ms,
        "zero_dep_loaded_modules": zero_mods,
        "pydantic_aiohttp_import_ms": heavy_ms,
        "pydantic_aiohttp_loaded_modules": heavy_mods,
        "speedup_x": round(heavy_ms / max(zero_ms, 0.01), 2),
    }


def main() -> None:
    pda = bench_pda_repair()
    bm25 = bench_tool_discovery()
    cold = bench_cold_start()

    report = {
        "structured_pda_repair": pda,
        "deferred_tool_discovery": bm25,
        "cold_start_footprint": cold,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
