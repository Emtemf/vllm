# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavioral tests for the opt-in TTFT diagnostic helper.

These cover the contract other modules rely on: the off-path is inert, the
key=value log line is greppable and well-formed, spans pair start/end with
matching ids and a non-negative duration, exceptions are recorded and
re-raised without leaking context, and point events never inherit the
enclosing span_id.
"""

import logging

import pytest

from vllm.v1.metrics import ttft_diagnostics as diag


def _parse_kv(body: str) -> dict[str, str]:
    """Split `k=v k2="a b"` into a dict of raw string values.

    Mirrors the helper's own renderer: bare tokens are space-delimited, quoted
    tokens may contain spaces and use backslash escapes.
    """
    out: dict[str, str] = {}
    i, n = 0, len(body)
    while i < n:
        eq = body.index("=", i)
        key = body[i:eq]
        j = eq + 1
        if j < n and body[j] == '"':
            j += 1
            buf = []
            while j < n and body[j] != '"':
                if body[j] == "\\":
                    j += 1
                if j < n:
                    buf.append(body[j])
                j += 1
            j += 1  # closing quote
            out[key] = "".join(buf)
        else:
            k = j
            while k < n and body[k] != " ":
                k += 1
            out[key] = body[j:k]
            j = k
        i = j + 1  # skip the separating space
    return out


@pytest.fixture
def captured(monkeypatch):
    """Capture helper log records and force the enabled path deterministically.

    ``_ENABLED`` is read once at import; the helper's public surface keys off
    it, so patch the module attribute rather than the process environment.
    """
    records: list[dict] = []

    class Handler(logging.Handler):
        def emit(self, record):
            text = record.getMessage()
            assert text.startswith("[TTFT_DIAG] "), text
            records.append(_parse_kv(text[len("[TTFT_DIAG] "):]))

    handler = Handler()
    monkeypatch.setattr(diag, "_ENABLED", True)
    monkeypatch.setattr(diag, "_HOST", "test-host")
    monkeypatch.setattr(diag, "_RUN_ID", "test-run")
    logger = logging.getLogger("vllm.ttft_diagnostics")
    logger.handlers[:] = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return records


def test_disabled_path_is_inert(monkeypatch):
    """With the flag off there is no output and no retained context.

    ``new_id`` has no enabled guard of its own; callers gate it behind
    ``enabled()``. Off, it still returns a str (with an empty run id) and must
    not raise, but nothing is logged and ``current()`` stays empty.
    """
    seen = []

    class Handler(logging.Handler):
        def emit(self, record):
            seen.append(record.getMessage())

    monkeypatch.setattr(diag, "_ENABLED", False)
    monkeypatch.setattr(diag, "_RUN_ID", "")
    logger = logging.getLogger("vllm.ttft_diagnostics")
    logger.handlers[:] = [Handler()]
    logger.setLevel(logging.INFO)
    with diag.scope(request_id="r"), diag.span("never"):
        diag.emit("never")
    assert seen == []
    assert diag.current() == {}
    # new_id stays safe to call when disabled; it just carries an empty run id.
    value = diag.new_id("step")
    assert isinstance(value, str) and value.startswith("step--")


def test_key_value_line_is_wellformed(captured):
    diag.emit("request_wait", request_id="req-1", tp_rank=3, some_none=None,
              flag=True, ratio=0.5, name="has space", empty=())
    rec = captured[-1]
    assert rec["event"] == "request_wait"
    assert rec["request_id"] == "req-1"
    assert rec["tp_rank"] == "3"
    assert rec["some_none"] == "null"
    assert rec["flag"] == "true"
    assert rec["ratio"] == "0.5"
    assert rec["name"] == "has space"
    assert rec["empty"] == "[]"
    # Common envelope fields are always present.
    for field in ("version", "host", "pid", "run_id", "mono_ns", "wall_ns"):
        assert field in rec, field
    assert rec["host"] == "test-host"
    assert rec["run_id"] == "test-run"


def test_sequence_is_comma_joined_without_spaces(captured):
    diag.emit("step", request_ids=("A", "B", "C"))
    rec = next(r for r in captured if r["event"] == "step")
    assert rec["request_ids"] == "A,B,C"


def test_span_pairs_start_and_end_with_matching_id(captured):
    with diag.span("engine_execute_wait", request_id="req-1"):
        pass
    starts = [r for r in captured if r["event"] == "engine_execute_wait_start"]
    ends = [r for r in captured if r["event"] == "engine_execute_wait"]
    assert len(starts) == 1 and len(ends) == 1
    assert starts[0]["span_id"] == ends[0]["span_id"]
    start_ns, end_ns = int(starts[0]["start_ns"]), int(ends[0]["end_ns"])
    duration = int(ends[0]["duration_ns"])
    assert duration == end_ns - start_ns >= 0
    assert ends[0]["status"] == "ok"


def test_nested_span_records_parent(captured):
    with diag.span("outer"), diag.span("inner"):
        pass
    outer_start = next(r for r in captured if r["event"] == "outer_start")
    inner_start = next(r for r in captured if r["event"] == "inner_start")
    assert inner_start["parent_span_id"] == outer_start["span_id"]


def test_point_event_does_not_inherit_span_id(captured):
    with diag.span("outer"):
        diag.emit("point")
    point = next(r for r in captured if r["event"] == "point")
    outer = next(r for r in captured if r["event"] == "outer")
    # A point event must not masquerade as the enclosing span's end record.
    assert "span_id" not in point or point.get("span_id") != outer["span_id"]


def test_span_records_error_and_reraises_without_leaking_context(captured):
    with pytest.raises(ValueError, match="secret"), diag.span("failure"):
        raise ValueError("secret")
    end = next(r for r in captured if r["event"] == "failure")
    # The span contract guarantees a status on error; the concrete exception
    # type is recorded by the caller's own business emit, not by the span.
    assert end["status"] == "error"
    # The exception message itself is never serialized into the record.
    assert "secret" not in str(end)
    # Context is restored even on the exception path.
    assert diag.current() == {}


def test_scope_restores_previous_context(captured):
    with diag.scope(request_id="outer"):
        with diag.scope(tp_rank=2):
            assert diag.current()["request_id"] == "outer"
            assert diag.current()["tp_rank"] == 2
        assert diag.current() == {"request_id": "outer"}
    assert diag.current() == {}


def test_new_id_is_bounded_ascii_and_unique(captured):
    ids = {diag.new_id("span") for _ in range(50)}
    assert len(ids) == 50, "ids must be unique"
    for value in ids:
        assert value.isascii()
        assert len(value) <= 256
