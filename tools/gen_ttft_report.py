#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate a step report retaining the v0.23 TTFT and tokenizer columns.

Usage:
    python gen_ttft_report.py log_prefill.log -o report.html
    python gen_ttft_report.py log_prefill.log --req REQUEST_ID --tp-rank 7

The default prefill scope ends at BREAKDOWN. Missing associations are displayed
as dashes. Execution and task identities, native thread mappings and explicit
task intervals are used; log ordering never manufactures an execution identity.

Deploy matching vLLM and vllm-ascend diagnostics and retain startup TTFT_THREAD
records. Mooncake's original batch_put_from_multi_buffers API and native phase
logs are sufficient; no trace API is required. A build exposing optional trace
APIs may remain installed. Old logs without execution/thread identity stay
incomplete. Save details and volume default to TP0, not summed parallel ranks.
"""

from __future__ import annotations

import argparse
import ast
import html
import math
import re
from collections import defaultdict
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from pathlib import Path

RE_KV = re.compile(r"(\w+)=(\[[^\]\n]*\]|\([^\)\n]*\)|\"[^\"\n]*\"|'[^'\n]*'|[^\s]+)")
RE_STORE = re.compile(r"\[TTFT_DEBUG\]\[AscendStoreSave[^\]]*\]\s*(.*)$")
RE_WORKER = re.compile(r"\(Worker[^)]*?_TP(\d+)[^)]*?\bpid=(\d+)[^)]*\)")
RE_PID = re.compile(r"\bpid=(\d+)")
RE_GLOG = re.compile(
    r"\b[IWEF](\d{2})(\d{2})\s+(\d{2}):(\d{2}):(\d{2})\.(\d{1,9})\s+(\d+)\s+"
)
RE_LOG_WALL = re.compile(
    r"\b(\d{2})-(\d{2})[ T](\d{2}):(\d{2}):(\d{2})(?:[.,](\d{1,9}))?"
)
PHASES = (
    "phase1_start_batch_put_us",
    "phase2_submit_transfers_us",
    "phase3_wait_transfers_us",
    "phase4_finalize_batch_put_us",
)
SCHEDULE = ("hash", "ext_lookup", "alloc", "overhead")
ENGINE = ("sched_hash", "sched_ext", "sched_alloc", "sched_overhead", "load", "forward")
SAVE = ("queue_wait", "build_key", "lookup", "prepare", "sync")


def pf(text):
    result = {}
    for match in RE_KV.finditer(text):
        value = match[2].rstrip(",")
        if value[:1] in ('"', "'"):
            with suppress(SyntaxError, ValueError):
                value = ast.literal_eval(value)
        result[match[1]] = value
    return result


def number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def integer(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def total(records, field):
    if not records:
        return None
    values = [number(record.get(field)) for record in records]
    return None if any(value is None for value in values) else sum(values)


def ids(record):
    for key in ("req_id", "req", "request_id", "req_ids", "request_ids"):
        value = record.get(key)
        if value is None or str(value) in ("", "None", "null"):
            continue
        try:
            parsed = ast.literal_eval(str(value))
        except (SyntaxError, ValueError):
            parsed = value
        if isinstance(parsed, (tuple, list)):
            return {str(item) for item in parsed if str(item)}
        return {item.strip() for item in str(parsed).split(",") if item.strip()}
    return set()


def parse(path: Path):
    records = {
        kind: []
        for kind in (
            "bd",
            "queue",
            "tok",
            "steps",
            "saves",
            "wfs",
            "timing",
            "diag",
            "threads",
        )
    }
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line_no, line in enumerate(stream, 1):
            store_match = RE_STORE.search(line)
            if store_match:
                tail = store_match[1]
                fields = pf(tail)
                if "keys" in fields:
                    kind = "saves"
                elif "wait_for_save" in fields:
                    kind = "wfs"
                else:
                    continue
            else:
                for marker, kind in (
                    ("[TTFT][BREAKDOWN]", "bd"),
                    ("[TTFT][QUEUE]", "queue"),
                    ("[TTFT_DEBUG][Tokenizer]", "tok"),
                    ("[TTFT_DEBUG][EngineCore]", "steps"),
                    ("[BatchPutTiming]", "timing"),
                    ("[TTFT_THREAD]", "threads"),
                    ("[TTFT_DIAG]", "diag"),
                ):
                    if marker in line:
                        tail = line.split(marker, 1)[1]
                        fields = pf(tail)
                        break
                else:
                    continue
            worker = RE_WORKER.search(line)
            pid = integer(fields.get("pid"))
            if pid is None:
                pid_match = RE_PID.search(line)
                pid = int(pid_match[1]) if pid_match else None
            rank = integer(fields.get("tp_rank"))
            if rank is None and worker:
                rank = int(worker[1])
            glog = RE_GLOG.search(line)
            wall = None
            if glog:
                month, day, hour, minute, second = map(int, glog.groups()[:5])
                wall = (
                    (month, day),
                    ((hour * 60 + minute) * 60 + second) * 10**9
                    + int(glog[6].ljust(9, "0")),
                )
            else:
                stamp = RE_LOG_WALL.search(line)
                if stamp:
                    month, day, hour, minute, second = map(int, stamp.groups()[:5])
                    wall = (
                        (month, day),
                        ((hour * 60 + minute) * 60 + second) * 10**9
                        + int((stamp[6] or "0").ljust(9, "0")),
                    )
            fields.update(
                _line=line_no,
                _source=str(path),
                _raw=line.rstrip(),
                _worker=(rank, pid) if rank is not None and pid is not None else None,
                _pid=pid,
                _prefer="[prefer_same_node]" in tail,
                _native_tid=integer(
                    fields.get(
                        "native_tid", fields.get("native_thread_id", fields.get("tid"))
                    )
                ),
                _wall=wall,
            )
            if glog:
                fields["_native_tid"] = int(glog[7])
            records[kind].append(fields)
    resolve_threads(records)
    return records


def resolve_threads(data):
    """Resolve native glog TIDs only through a unique explicit thread mapping."""
    mappings = defaultdict(set)
    for record in data["threads"]:
        tid, pid = record["_native_tid"], record["_pid"]
        if tid is None or pid is None:
            continue
        contexts = {
            (r.get("host"), pid, r.get("run_id"), r["_worker"] or record["_worker"])
            for r in data["diag"]
            if r["_native_tid"] == tid
            and r["_pid"] == pid
            and r.get("host")
            and r.get("run_id")
        }
        mappings[tid].update(
            contexts
            or {
                (
                    record.get("host"),
                    pid,
                    record.get("run_id"),
                    record["_worker"],
                )
            }
        )
    for records in data.values():
        for record in records:
            tid = record["_native_tid"]
            candidates = [
                item
                for item in mappings.get(tid, ())
                if (record["_pid"] is None or item[1] == record["_pid"])
                and (not record.get("host") or item[0] == record["host"])
                and (not record.get("run_id") or item[2] == record["run_id"])
            ]
            if len(candidates) == 1:
                host, pid, run, worker = candidates[0]
                record["_pid"] = pid
                record["_worker"] = record["_worker"] or worker
                record["_thread"] = (host, pid, run, tid)
            elif not candidates and tid is not None and record["_pid"] is not None:
                record["_thread"] = (
                    record.get("host"),
                    record["_pid"],
                    record.get("run_id"),
                    tid,
                )
            else:
                record["_thread"] = None


def combine_phases(records):
    """Pair split native phases on the same resolved native thread, never TP alone."""
    pending = defaultdict(list)
    batches = []
    for record in records:
        thread = record["_thread"]
        if record.get(PHASES[1]) is None:
            if record["_prefer"] and record.get(PHASES[0]) is not None:
                pending[thread].append(record)
            continue
        batch = dict(record)
        batch["_first_line"] = record["_line"]
        batch["_first_wall"] = record["_wall"]
        batch["_phase_lines"] = [record["_line"]]
        batch["_phase_status"] = (
            "complete"
            if all(number(record.get(field)) is not None for field in PHASES)
            else "incomplete"
        )
        if record["_prefer"] and record.get(PHASES[0]) is None:
            starts = pending.pop(thread, [])
            if (
                thread is not None
                and len(starts) == 1
                and starts[0].get("keys") == record.get("keys")
                and (
                    starts[0].get("bytes") is None
                    or starts[0].get("bytes") == record.get("bytes")
                )
            ):
                batch[PHASES[0]] = starts[0][PHASES[0]]
                batch["_first_line"] = starts[0]["_line"]
                batch["_first_wall"] = starts[0]["_wall"]
                batch["_phase_lines"].insert(0, starts[0]["_line"])
                batch["_phase_status"] = (
                    "complete"
                    if all(number(batch.get(field)) is not None for field in PHASES)
                    else "incomplete"
                )
            else:
                for start in starts:
                    orphan = dict(start)
                    orphan.update(
                        _first_line=start["_line"],
                        _phase_lines=[start["_line"]],
                        _first_wall=start["_wall"],
                        _phase_status="incomplete",
                    )
                    batches.append(orphan)
        else:
            for start in pending.pop(thread, []):
                orphan = dict(start)
                orphan.update(
                    _first_line=start["_line"],
                    _phase_lines=[start["_line"]],
                    _first_wall=start["_wall"],
                    _phase_status="incomplete",
                )
                batches.append(orphan)
        batches.append(batch)
    for starts in pending.values():
        for start in starts:
            batch = dict(start)
            batch.update(
                _first_line=start["_line"],
                _phase_lines=[start["_line"]],
                _phase_status="incomplete",
            )
            batches.append(batch)
    return sorted(batches, key=lambda record: record["_first_line"])


def choose_request(data, requested):
    breakdowns = data["bd"]
    if not breakdowns:
        raise ValueError("没有找到 [TTFT][BREAKDOWN]，无法确定请求总量和 step 数。")
    if len(breakdowns) != 1:
        raise ValueError(
            "本脚本用于单请求实验；日志有多条 BREAKDOWN，请先截取一次请求的日志。"
        )
    breakdown = breakdowns[0]
    names = ids(breakdown)
    if len(names) != 1:
        raise ValueError("BREAKDOWN 缺少唯一请求 ID。")
    request = next(iter(names))
    if requested and requested != request:
        raise ValueError(f"--req={requested} 与日志请求 {request} 不一致。")
    return request, breakdown


def execution(record):
    return record.get("execution_id") or record.get("ttft_execution_id") or ""


def select_steps(data, request, breakdown, mode, warnings):
    end = breakdown["_line"] if mode == "prefill" else math.inf
    records = [
        record
        for record in data["steps"]
        if record["_line"] < end
        and integer(record.get("num_total")) not in (None, 0)
        and integer(record.get("num_tokens")) not in (None, 0)
        and (
            mode != "prefill"
            or "prefill_request_ids" not in record
            or request in ids({"request_ids": record["prefill_request_ids"]})
        )
    ]
    expected = integer(breakdown.get("steps"))
    if expected is None or expected <= 0:
        raise ValueError("BREAKDOWN.steps 必须为正整数。")
    identified = [r for r in records if execution(r) and ids(r) == {request}]
    if identified:
        if mode == "prefill" and any(
            "prefill_request_ids" not in r for r in identified
        ):
            warnings.append(
                "执行日志缺少 prefill_request_ids；不能独立证明每轮均在首 token 前。"
            )
        grouped = defaultdict(list)
        for record in identified:
            grouped[execution(record)].append(record)
        steps = [group[0] for group in grouped.values() if len(group) == 1]
        if len(steps) != len(grouped):
            warnings.append("重复 execution_id 的 EngineCore 记录有歧义，未认领。")
        if mode == "prefill" and len(steps) != expected:
            warnings.append(
                f"BREAKDOWN.steps={expected}；实际唯一请求 execution={len(steps)}。"
                "保留真实 execution，不截取或补配其他空批次/decode。"
            )
    else:
        legacy = [r for r in records if not ids(r) or ids(r) == {request}]
        if (
            (len(legacy) == expected or (mode == "all" and len(legacy) > expected))
            and len({r["_pid"] for r in legacy}) == 1
            and not any(execution(r) for r in legacy)
        ):
            steps = legacy
            warnings.append(
                "旧 EngineCore 日志没有 execution_id；只展示原始行，不关联保存任务。"
            )
        else:
            steps = [{} for _ in range(expected)]
            warnings.append(
                "EngineCore 缺少唯一请求执行身份；保留缺失行，不按顺序猜测。"
            )
    if mode == "all":
        warnings.append("mode=all 包含首 token 后记录；总计行仍为 TTFT 请求总量。")
    return steps, end


def task_key(record):
    task = record.get("task_id")
    if task and execution(record) and record["_pid"] is not None:
        return record["_pid"], execution(record), task
    return None


def same_clock(left, right):
    return all(
        left.get(key) is not None and left.get(key) == right.get(key)
        for key in ("host", "pid", "run_id")
    )


def interval(record):
    start, end = integer(record.get("start_ns")), integer(record.get("end_ns"))
    duration = integer(record.get("duration_ns"))
    if start is None or end is None or duration is None or end < start:
        return None
    return (start, end) if duration == end - start else None


def span_start(data, completion):
    starts = [
        r
        for r in data["diag"]
        if r.get("event") == completion.get("event", "") + "_start"
        and same_clock(r, completion)
        and execution(r) == execution(completion)
        and task_key(r) == task_key(completion)
        and r.get("group_id") == completion.get("group_id")
        and r.get("span_id") == completion.get("span_id")
        and integer(r.get("start_ns")) == integer(completion.get("start_ns"))
    ]
    return starts[0] if len(starts) == 1 else None


def wall_key(value, reference):
    stamp = integer(value)
    if stamp is None or reference is None:
        return None
    # glog has no zone. Infer it from this source log, never this workstation.
    utc = datetime.fromtimestamp(stamp // 10**9, timezone.utc)
    (month, day), local_ns = reference
    offsets = []
    for year in (utc.year - 1, utc.year, utc.year + 1):
        try:
            local = datetime(year, month, day, tzinfo=timezone.utc)
        except ValueError:
            continue
        difference = int(local.timestamp()) * 10**9 + local_ns - stamp
        minutes = round(difference / (60 * 10**9))
        if (
            abs(minutes) <= 14 * 60
            and abs(difference - minutes * 60 * 10**9) < 2 * 10**9
        ):
            offsets.append(minutes)
    if len(offsets) != 1:
        return None
    date = utc + timedelta(minutes=offsets[0])
    return (
        (date.month, date.day),
        ((date.hour * 60 + date.minute) * 60 + date.second) * 10**9 + stamp % 10**9,
    )


def inside_put(batch, start, finish):
    if (
        batch["_thread"] is None
        or batch["_thread"] != finish["_thread"]
        or start["_thread"] != finish["_thread"]
        or interval(finish) is None
    ):
        return False
    lower = wall_key(start.get("wall_ns"), start["_wall"])
    upper = wall_key(finish.get("wall_ns"), finish["_wall"])
    first_wall = batch.get("_first_wall", batch["_wall"])
    if lower is not None and upper is not None and batch["_wall"] is not None:
        if first_wall is None:
            return False
        return lower <= first_wall <= batch["_wall"] <= upper
    return start["_line"] < batch["_first_line"] <= batch["_line"] < finish["_line"]


def associate_puts(data, saves):
    batches = combine_phases(data["timing"])
    windows = []
    for finish in data["diag"]:
        if finish.get("event") != "store_backend_put" or task_key(finish) is None:
            continue
        start = span_start(data, finish)
        if start is not None:
            windows.append((start, finish))
    matches = defaultdict(list)
    unassigned = []
    for batch in batches:
        candidates = [
            finish for start, finish in windows if inside_put(batch, start, finish)
        ]
        if len(candidates) == 1:
            finish = candidates[0]
            batch["_put_span"] = finish
            matches[task_key(finish)].append(batch)
        else:
            unassigned.append(batch)
    result = {}
    counts = defaultdict(int)
    for save in saves:
        counts[task_key(save)] += 1
    for save in saves:
        key = task_key(save)
        candidates = (
            matches.get(key, []) if key is not None and counts[key] == 1 else []
        )
        keys, byte_count = integer(save.get("keys")), integer(save.get("bytes"))
        phases = [None] * 4
        status = "缺失/歧义：没有唯一同线程 put 区间及 keys/bytes 对账"
        puts = [
            r
            for r in data["diag"]
            if r.get("event") in ("store_backend_put", "store_backend_put_start")
            and key is not None
            and task_key(r) == key
        ]
        group_valid = all(
            total([b for b in candidates if b["_put_span"] is finish], "keys")
            == number(finish.get("count"))
            and total([b for b in candidates if b["_put_span"] is finish], "bytes")
            == number(finish.get("bytes"))
            for _, finish in windows
            if key is not None and task_key(finish) == key
        )
        if (
            keys == 0
            and byte_count == 0
            and not candidates
            and key is not None
            and counts[key] == 1
        ):
            status = (
                "真实零提交：keys=0 bytes=0；put 调用已执行/区间待完成"
                if puts
                else "真实零提交：keys=0 bytes=0；无 put 调用记录"
            )
        elif (
            candidates
            and keys is not None
            and byte_count is not None
            and total(candidates, "keys") == keys
            and total(candidates, "bytes") == byte_count
            and group_valid
        ):
            phases = [
                None
                if total(candidates, field) is None
                else total(candidates, field) / 1e6
                for field in PHASES
            ]
            status = "完整" if all(v is not None for v in phases) else "阶段日志不完整"
        result[save["_line"]] = {
            "phases": phases,
            "batches": candidates,
            "status": status,
        }
    return result, unassigned


def save_rows(data, request, end, steps, requested_rank, warnings):
    executions = {execution(step) for step in steps if execution(step)}
    saves = [
        s
        for s in data["saves"]
        if ids(s) == {request} and (execution(s) in executions or s["_line"] < end)
    ]
    waits = [
        s
        for s in data["wfs"]
        if execution(s) in executions and (not ids(s) or ids(s) == {request})
    ]
    workers = sorted(
        {
            s["_worker"]
            for s in saves
            + waits
            + [r for r in data["diag"] if execution(r) in executions]
            if s["_worker"] is not None
        }
    )
    rank = 0 if requested_rank is None else requested_rank
    candidates = [w for w in workers if w[0] == rank]
    selected = candidates[0] if candidates else None
    if selected and sum(w[0] == selected[0] for w in candidates) > 1:
        raise ValueError("同一个 TP 有多个 Worker PID，保存时间来源有歧义。")
    if requested_rank is not None and selected is None:
        raise ValueError(f"没有找到 TP{requested_rank} 的 Worker 保存日志。")
    if selected is None:
        warnings.append("没有已关联 Worker 保存记录，保存明细缺失，不代表零耗时。")
    phase_map, unassigned = associate_puts(data, saves)
    if unassigned:
        warnings.append(
            f"{len(unassigned)} 条原生 put 未能唯一关联 task/put 区间；"
            "未按批次顺序消费。"
        )
    if any(not execution(s) or not task_key(s) for s in saves):
        warnings.append(
            "旧保存记录缺少 execution/task 身份；仅统计真实数据量，"
            "不按 born_step 或 join 序号关联。"
        )
    rows = []
    for step in steps:
        exec_id = execution(step)
        all_tasks = [
            s for s in saves if exec_id and execution(s) == exec_id and task_key(s)
        ]
        tasks = [s for s in all_tasks if s["_worker"] == selected]
        boundaries = [
            w for w in waits if execution(w) == exec_id and w["_worker"] == selected
        ]
        empty = (
            not tasks
            and len(boundaries) == 1
            and number(boundaries[0].get("join")) == 0
            and number(boundaries[0].get("enqueue")) is not None
            and number(boundaries[0].get("enqueue")) >= 0
        )
        phases = []
        for phase in range(4):
            values = [phase_map[s["_line"]]["phases"][phase] for s in tasks]
            phases.append(
                sum(values) if values and all(v is not None for v in values) else None
            )
        keys, byte_count = (
            (0, 0) if empty else (total(tasks, "keys"), total(tasks, "bytes"))
        )
        rows.append(
            {
                "values": [total(tasks, field) for field in SAVE]
                + phases
                + [keys, None if byte_count is None else byte_count / 1048576],
                "put": total(tasks, "put"),
                "empty": empty,
                "tasks": all_tasks,
                "phase_map": phase_map,
                "waits": boundaries,
                "unassigned_puts": unassigned,
            }
        )
    return rows, selected, saves


def tokenizer_for_request(data, request, end, warnings):
    records = [
        r
        for r in data["tok"]
        if r["_line"] < end
        and ids(r) == {request}
        and r.get("purpose") == "business"
        and integer(r.get("n")) not in (None, 0)
    ]
    if len(records) == 1:
        result = dict(records[0])
        sources = [records[0]["_line"]]
        for name in ("render", "tokenize"):
            spans = [
                r
                for r in data["diag"]
                if r.get("event") == name
                and ids(r) == {request}
                and r.get("purpose") == "business"
                and r["_line"] < end
            ]
            if len(spans) == 1 and interval(spans[0]) is not None:
                result[name] = integer(spans[0]["duration_ns"]) / 1e9
                sources.append(spans[0]["_line"])
            elif spans:
                result.pop(name, None)
                warnings.append(f"Tokenizer {name} 结构化区间不完整/歧义，未扣减。")
        result["_sources"] = sources
        return result
    # Structured intervals can be present even when the compatibility line was lost.
    result = {}
    for name in ("render", "tokenize"):
        spans = [
            r
            for r in data["diag"]
            if r.get("event") == name
            and ids(r) == {request}
            and r.get("purpose") == "business"
            and r["_line"] < end
        ]
        if len(spans) == 1 and interval(spans[0]) is not None:
            result[name] = integer(spans[0]["duration_ns"]) / 1e9
            result.setdefault("_sources", []).append(spans[0]["_line"])
            result["_line"] = spans[0]["_line"]
    if not result:
        warnings.append(
            "Tokenizer 缺少唯一 business 请求身份；"
            "warmup/unscoped 不认领，旧日志保持缺失。"
        )
    return result


def escape(value):
    return html.escape(str(value), quote=True)


def fmt(value, digits=6):
    return "—" if value is None else f"{value:.{digits}f}"


CSS = """
:root{color-scheme:light;--fg:#182635;--muted:#526374;--border:#becad3;
--surface:#f5f8fa;--ec:#edf4ff;--wk:#eef7f1;--store:#f4f2f9;--put:#fff4df}
*{box-sizing:border-box}body{margin:0;padding:22px;color:var(--fg);
font:15px/1.65 'Segoe UI','Microsoft YaHei',sans-serif}h1{font-size:24px;margin:0 0 6px}
.meta,.note{color:var(--muted);font-size:14px}.request{padding:12px 16px;
border:1px solid var(--border);background:#f0f7fc;margin:12px 0}
.request b{margin-right:18px}
.scroll{overflow-x:auto;max-width:100%;border:1px solid var(--border)}
table{border-collapse:separate;border-spacing:0;width:100%;min-width:1650px;
font:14px/1.6 ui-monospace,Consolas,monospace;white-space:nowrap}
th,td{border-right:1px solid var(--border);border-bottom:1px solid var(--border);
padding:8px;text-align:right}
thead th{text-align:center;font-family:'Segoe UI',sans-serif}
th:first-child{position:sticky;left:0;text-align:left;background:var(--surface);z-index:1}
.ec{background:var(--ec)}.wk{background:var(--wk)}.store{background:var(--store)}
.put{background:var(--put)}.total td{text-align:center;font-weight:600}
.missing{color:#697b89}
details{margin:12px 0;border:1px solid var(--border);padding:9px 14px}
summary{cursor:pointer}li{margin:5px 0}.warning{border-left:4px solid #b77719;
background:#fff8e8;padding:10px 16px;margin:12px 0;font-size:14px}
code{overflow-wrap:anywhere}a{color:#215b8e}
"""


def evidence(record):
    return (
        f"<details><summary>来源 {escape(record['_source'])}:"
        f"{record['_line']}</summary>"
        f"<code>{escape(record['_raw'])}</code></details>"
    )


def span_description(record):
    bounds = interval(record)
    duration = None if bounds is None else (bounds[1] - bounds[0]) / 1e9
    status = "完整区间" if bounds is not None else "区间缺失/不一致"
    identity = " · ".join(
        f"{key}={record[key]}"
        for key in (
            "timing_scope",
            "host",
            "pid",
            "run_id",
            "native_tid",
            "tp_rank",
            "span_id",
            "parent_span_id",
            "start_ns",
            "end_ns",
            "status",
        )
        if key in record
    )
    return (
        f"<li><b>{escape(record.get('event', 'span'))}</b>: {fmt(duration)} s · "
        f"{status} · {escape(identity)}{evidence(record)}</li>"
    )


def visible_span(record, records):
    if not record.get("event", "").endswith("_start"):
        return True
    return not any(
        other.get("span_id") == record.get("span_id")
        and record.get("span_id")
        and same_clock(record, other)
        and not other.get("event", "").endswith("_start")
        for other in records
    )


def task_exclusive(data, parent, puts, save):
    """Subtract complete same-task put descendants on the same clock/thread."""
    bounds = interval(parent)
    if bounds is None or parent["_thread"] is None or span_start(data, parent) is None:
        return None
    starts = [
        r
        for r in data["diag"]
        if r.get("event") == "store_backend_put_start"
        and task_key(r) == task_key(parent)
        and same_clock(r, parent)
    ]
    if len(starts) != len(puts):
        return None
    if not puts:
        return (
            (bounds[1] - bounds[0]) / 1e9
            if (integer(save.get("keys")) == 0 and integer(save.get("bytes")) == 0)
            else None
        )
    if total(puts, "count") != number(save.get("keys")):
        return None
    if total(puts, "bytes") != number(save.get("bytes")):
        return None
    windows = []
    for child in puts:
        window = interval(child)
        if (
            window is None
            or not same_clock(parent, child)
            or child["_thread"] != parent["_thread"]
            or task_key(child) != task_key(parent)
            or not parent.get("span_id")
            or window[0] < bounds[0]
            or window[1] > bounds[1]
            or span_start(data, child) is None
        ):
            return None
        windows.append(window)
    covered = 0
    cursor = bounds[0]
    for start, end in sorted(windows):
        covered += max(0, end - max(cursor, start))
        cursor = max(cursor, end)
    return (bounds[1] - bounds[0] - covered) / 1e9


def worker_uncovered(data, parent):
    """Return only proven same-thread gaps inside a complete Worker call."""
    bounds = interval(parent)
    if (
        bounds is None
        or parent["_thread"] is None
        or span_start(data, parent) is None
        or not execution(parent)
    ):
        return None
    children = [
        r
        for r in data["diag"]
        if execution(r) == execution(parent)
        and same_clock(r, parent)
        and r["_thread"] == parent["_thread"]
        and r.get("event") in ("worker_load", "worker_forward", "worker_save")
        and interval(r) is not None
        and bounds[0] <= interval(r)[0] <= interval(r)[1] <= bounds[1]
    ]
    if any(
        sum(r.get("event") == event for r in children) != 1
        for event in ("worker_load", "worker_forward")
    ):
        return None
    for start in data["diag"]:
        if (
            start.get("event") in ("worker_load_start", "worker_save_start")
            and execution(start) == execution(parent)
            and same_clock(start, parent)
            and start["_thread"] == parent["_thread"]
            and bounds[0] <= (integer(start.get("start_ns")) or -1) <= bounds[1]
            and not any(span_start(data, child) is start for child in children)
        ):
            return None
    covered, cursor = 0, bounds[0]
    for start, end in sorted(interval(r) for r in children):
        covered += max(0, end - max(cursor, start))
        cursor = max(cursor, end)
    return (bounds[1] - bounds[0] - covered) / 1e9


def hierarchy(data, request, steps, rows, raw_saves, end):
    output = [
        (
            "<details open><summary>请求 → execution → TP / save task → put · "
            f"{escape(request)}</summary>"
        )
    ]
    associated = set()
    for step, row in zip(steps, rows):
        exec_id = execution(step)
        if not exec_id:
            continue
        output.append(f"<details><summary>execution {escape(exec_id)}</summary>")
        output.append(evidence(step))
        spans = [
            r
            for r in data["diag"]
            if execution(r) == exec_id and (not ids(r) or ids(r) == {request})
        ]
        engine_spans = [
            r
            for r in spans
            if visible_span(r, spans)
            and not r.get("event", "").startswith(("worker_", "store_"))
        ]
        output.append("<h3>Engine · shared_batch_host</h3><ul>")
        output.extend(span_description(r) for r in engine_spans)
        if not engine_spans:
            output.append("<li>缺少 Engine 区间；不由总耗时均分。</li>")
        output.append("</ul>")
        workers = sorted(
            {
                r["_worker"]
                for r in spans
                + row["tasks"]
                + [r for r in data["wfs"] if execution(r) == exec_id]
                if r["_worker"] is not None
            }
        )
        for worker in workers:
            output.append(
                f"<details><summary>TP{worker[0]} / PID {worker[1]}</summary><ul>"
            )
            worker_spans = [
                r
                for r in spans
                if r["_worker"] == worker
                and r.get("event", "").startswith("worker_")
                and visible_span(r, spans)
            ]
            output.extend(span_description(r) for r in worker_spans)
            for parent in worker_spans:
                if parent.get("event") == "worker_execute":
                    uncovered = worker_uncovered(data, parent)
                    output.append(
                        "<li>Worker 外层未被已观测 load/forward/save 区间覆盖："
                        f"{fmt(uncovered)} s；仅限该方法内，不直接加到 residual。</li>"
                    )
            output.append("</ul>")
            worker_waits = [
                r
                for r in data["wfs"]
                if execution(r) == exec_id
                and r["_worker"] == worker
                and (not ids(r) or ids(r) == {request})
            ]
            for wait in worker_waits:
                status = (
                    "未执行保存：显式 enqueue=0/join=0"
                    if (
                        number(wait.get("enqueue")) == 0
                        and number(wait.get("join")) == 0
                        and not any(s["_worker"] == worker for s in row["tasks"])
                    )
                    else "已记录同步 wait_for_save；不可视为任务数/成本加总"
                )
                output.append(f"<p>{status}</p>{evidence(wait)}")
            tasks = [s for s in row["tasks"] if s["_worker"] == worker]
            if not tasks:
                output.append(
                    "<p>无已关联任务；仅显式无任务 join=0 边界可证明未执行，"
                    "缺日志不等于零。</p>"
                )
            orphan_spans = [
                r
                for r in spans
                if r["_worker"] == worker
                and r.get("event", "").startswith("store_")
                and not any(task_key(s) == task_key(r) for s in tasks)
            ]
            if orphan_spans:
                output.append(
                    "<details><summary>缺少保存汇总的 task/put 区间</summary><ul>"
                )
                output.extend(span_description(r) for r in orphan_spans)
                output.append("</ul></details>")
            for save in tasks:
                associated.add(save["_line"])
                match = row["phase_map"][save["_line"]]
                task_spans = [
                    r
                    for r in spans
                    if r.get("event") == "store_task" and task_key(r) == task_key(save)
                ]
                puts = [
                    r
                    for r in spans
                    if r.get("event") == "store_backend_put"
                    and task_key(r) == task_key(save)
                ]
                exclusive = (
                    task_exclusive(data, task_spans[0], puts, save)
                    if len(task_spans) == 1
                    else None
                )
                output.append(
                    f"<details><summary>save task {escape(save.get('task_id'))} · "
                    f"{escape(match['status'])} · "
                    f"keys={escape(save.get('keys', '—'))} · "
                    f"bytes={escape(save.get('bytes', '—'))}</summary>"
                    f"<p>put={fmt(number(save.get('put')))} s；"
                    "task 扣除完整同线程 put 区间并集后 "
                    f"{fmt(exclusive)} s（不可计算时为 —，不跨进程减时间戳）。</p>"
                )
                output.append(evidence(save))
                output.append("<ul>")
                output.extend(span_description(r) for r in task_spans)
                output.extend(
                    span_description(r)
                    for r in spans
                    if task_key(r) == task_key(save)
                    and r.get("event", "").startswith("store_")
                    and r.get("event", "").endswith("_start")
                    and visible_span(r, spans)
                )
                output.append("</ul>")
                for put in puts:
                    batches = [b for b in match["batches"] if b.get("_put_span") is put]
                    output.append(
                        "<details><summary>put "
                        f"group={escape(put.get('group_id', '—'))} · "
                        f"native_tid={escape(put.get('native_tid', '—'))} · "
                        f"{len(batches)} 条原生 BatchPutTiming</summary><ul>"
                    )
                    output.append(span_description(put))
                    for batch in batches:
                        phase_text = " / ".join(
                            fmt(number(batch.get(p))) for p in PHASES
                        )
                        output.append(
                            f"<li>原生阶段 μs: {phase_text} · "
                            f"{escape(batch['_phase_status'])} · "
                            f"源行 {escape(batch['_phase_lines'])} · "
                            f"keys={escape(batch.get('keys', '—'))} "
                            f"bytes={escape(batch.get('bytes', '—'))}"
                            + "".join(
                                evidence(source)
                                for source in data["timing"]
                                if source["_line"] in batch["_phase_lines"]
                            )
                            + "</li>"
                        )
                    if not batches:
                        output.append(
                            "<li>原生阶段缺失/歧义；"
                            "已执行 put span 不等于原生耗时为零。</li>"
                        )
                    output.append("</ul></details>")
                output.append("</details>")
            output.append("</details>")
        output.append("</details>")
    unmatched = [s for s in raw_saves if s["_line"] not in associated]
    output.append(
        "<details><summary>未归属 execution/task 的保存证据："
        f"{len(unmatched)}</summary>"
    )
    output.extend(evidence(r) for r in unmatched)
    output.append("</details>")
    request_spans = [
        r for r in data["diag"] if ids(r) == {request} and not execution(r)
    ]
    output.append(
        "<details><summary>请求级核心区间 / residual 诊断（不可任意相加）</summary><ul>"
    )
    output.extend(
        span_description(r) for r in request_spans if visible_span(r, data["diag"])
    )
    if not request_spans:
        output.append("<li>旧日志缺少请求级区间，不据 log 顺序制造独占时间。</li>")
    output.append("</ul></details>")
    unassigned = (
        rows[0]["unassigned_puts"] if rows else associate_puts(data, raw_saves)[1]
    )
    output.append(
        f"<details><summary>未唯一归属的原生 put：{len(unassigned)}</summary>"
    )
    output.extend(evidence(r) for r in unassigned)
    output.append("</details>")
    output.append(
        "<p class='note'>TP 时间并行，仅逐 rank 展示，绝不相加为请求成本；"
        "数据量可累加。区间 start_ns/end_ns 为同机单调时钟；"
        "wall_ns 仅作原生日志关联桥。"
        "独占值仅由相同 host/PID/run/原生线程和 task 的完整区间并集计算；"
        "未完整覆盖时为缺失。所有来源可展开查看。</p></details>"
    )
    return "".join(output)


def render(
    data, out: Path, req_filter=None, mode="prefill", tp_rank=None, tokenizer_line=None
):
    warnings = []
    request, bd = choose_request(data, req_filter)
    steps, end = select_steps(data, request, bd, mode, warnings)
    saves, worker, raw_saves = save_rows(data, request, end, steps, tp_rank, warnings)
    tok = tokenizer_for_request(data, request, bd["_line"], warnings)
    if tokenizer_line is not None:
        candidates = [t for t in data["tok"] if t["_line"] == tokenizer_line]
        if (
            len(candidates) != 1
            or tokenizer_line >= bd["_line"]
            or candidates[0].get("purpose") in ("warmup", "unscoped")
            or (ids(candidates[0]) and ids(candidates[0]) != {request})
        ):
            raise ValueError("--tokenizer-line 不是目标请求的可用 Tokenizer 证据。")
        if not tok:
            tok = dict(candidates[0], _manual=True)
            warnings.append(
                "手工旧 Tokenizer 行仅供查看，没有业务请求身份，不用于 residual 扣减。"
            )
        elif tokenizer_line not in tok.get("_sources", [tok["_line"]]):
            raise ValueError("--tokenizer-line 与已自动关联的业务分词记录不一致。")
    queues = [q for q in data["queue"] if ids(q) == {request}]
    queue = queues[0] if len(queues) == 1 else {}
    queue_valid = (
        queue.get("initial_queue_valid") == "true"
        and queue.get("reason") == "available"
        and integer(queue.get("preemptions")) == 0
        and integer(queue.get("queued_events")) == 1
        and integer(queue.get("scheduled_events")) == 1
    )
    queue_time = number(queue.get("queue_wait_s")) if queue_valid else None
    if queue_time is not None and queue_time < 0:
        queue_time = None
    if queue_time is None:
        warnings.append(
            "首次 QUEUE 无有效完整证据或发生 preemption/requeue；"
            "不声称总排队，不从 residual 扣减。"
        )
    schedule_values = [number(bd.get(k)) for k in SCHEDULE]
    schedule = None if any(v is None for v in schedule_values) else sum(schedule_values)
    known_keys = SCHEDULE + ("load_kv", "forward", "save_kv", "update")
    if "kv_transfer" in bd:
        known_keys += ("kv_transfer",)
    known_values = [number(bd.get(k)) for k in known_keys]
    ttft = number(bd.get("ttft"))
    residual = number(bd.get("residual"))
    delta = None
    if (
        ttft is not None
        and residual is not None
        and all(v is not None for v in known_values)
    ):
        delta = ttft - sum(known_values) - residual
    body = []
    for index, (step, save) in enumerate(zip(steps, saves)):
        cells = [
            (
                f'<th scope="row" title="{escape(execution(step) or "执行身份缺失")}">'
                f"step{index}</th>"
            ),
            '<td title="请求级 TTFT 仅在总计行展示"></td>',
            '<td title="请求级 tokenizer 仅在总计行展示"></td>',
        ]
        values = [number(step.get(k)) for k in ENGINE] + save["values"]
        for column, value in enumerate(values):
            if column == 6:
                wait = save["waits"][0] if len(save["waits"]) == 1 else {}
                selected_tasks = [
                    task for task in save["tasks"] if task["_worker"] == worker
                ]
                interface_values = [
                    number(step.get("save")),
                    number(wait.get("wait_for_save")),
                    number(wait.get("enqueue")),
                    number(wait.get("join")),
                    total(selected_tasks, "total"),
                ]
                cells.extend(f"<td>{fmt(item)}</td>" for item in interface_values)
            if 11 <= column <= 14 and (
                save["empty"]
                or (
                    save["put"] is not None
                    and any(v is None for v in save["values"][5:9])
                )
            ):
                if column == 11:
                    label = (
                        "未执行保存（无任务、join=0）"
                        if save["empty"]
                        else f"{fmt(save['put'])} s（put 总耗时；四阶段未关联）"
                    )
                    cells.append(f'<td class="put" colspan="4">{label}</td>')
                continue
            if save["empty"] and 6 <= column <= 10:
                cells.append('<td class="missing">未执行</td>')
                continue
            classes = "put" if 11 <= column <= 14 else ""
            if value is None:
                classes += " missing"
            digits = 0 if column == 15 else 1 if column == 16 else 6
            title = f"EngineCore 日志行 {step['_line']}" if column < 6 and step else ""
            cells.append(
                f'<td class="{classes.strip()}" title="{escape(title)}">'
                f"{fmt(value, digits)}</td>"
            )
        body.append("<tr>" + "".join(cells) + "</tr>")
    worker_text = (
        "未知 Worker" if worker is None else f"TP{worker[0]} / PID {worker[1]}"
    )
    key_total = total(raw_saves, "keys")
    byte_total = total(raw_saves, "bytes")
    data_summary = (
        f"真实保存记录 {len(raw_saves)} 条；全部 TP 的 key 提交数 {fmt(key_total, 0)}；"
        f"数据量 {fmt(None if byte_total is None else byte_total / 1048576, 1)} MiB"
    )
    queue_source = "—" if not queue else str(queue["_line"])
    tokenizer_source = (
        "—" if not tok else ", ".join(map(str, tok.get("_sources", [tok["_line"]])))
    )
    residual_parts = [
        number(tok.get("render")),
        number(tok.get("tokenize")),
        queue_time,
    ]
    remaining = None
    if (
        residual is not None
        and not tok.get("_manual")
        and all(v is not None for v in residual_parts)
    ):
        remaining = residual - sum(residual_parts)
    warning_html = ""
    tokenizer_control = (
        "Tokenizer 依据唯一 business 请求身份自动关联；warmup/unscoped 不参与。"
    )
    hierarchy_html = hierarchy(data, request, steps, saves, raw_saves, end)
    queue_evidence = evidence(queue) if queue else "<p>QUEUE 证据缺失或多条歧义。</p>"
    tokenizer_evidence = "".join(
        evidence(r)
        for r in data["tok"] + data["diag"]
        if r["_line"] in tok.get("_sources", [tok.get("_line")])
    )
    if warnings:
        warning_html = (
            '<div class="warning"><b>日志完整性说明</b><ul>'
            + "".join(
                f"<li>{escape(message)}</li>" for message in dict.fromkeys(warnings)
            )
            + "</ul></div>"
        )
    reconciliation = "对账所需字段不完整，未计算误差。"
    if delta is not None:
        reconciliation = (
            f"原始 BREAKDOWN 对账误差 {delta:+.6f} s（保留日志舍入误差，不自动改数）。"
        )
    transfer_summary = ""
    if "kv_transfer" in bd:
        transfer_summary = f"、kv_transfer={fmt(number(bd['kv_transfer']))} s"
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TTFT 逐 step 统计 · {escape(request)}</title><style>{CSS}</style></head><body>
<h1>TTFT 逐 step 统计</h1>
<div class="meta">请求 <code>{escape(request)}</code> · mode={escape(mode)} ·
{len(steps)} steps ·
保存耗时固定 {escape(worker_text)} · 时间 s，数据量 MiB</div>
<div class="request">render
<b><span id="render-value">{fmt(number(tok.get("render")))}</span> s</b>
首次 Scheduler 排队 <b>{fmt(queue_time)} s</b>
update <b>{fmt(number(bd.get("update")))} s</b>
剩余 residual <b id="remaining-value">{fmt(remaining)} s</b><br>{data_summary}</div>
<p>{tokenizer_control}</p>
<p class="note">请求分解：TTFT = render + tokenize + 首次排队 + schedule
+ load_kv + forward + save_kv + update
{" + kv_transfer" if "kv_transfer" in bd else ""} + 剩余 residual。
仅对唯一关联的预处理与有效首次排队作扣减；缺失时剩余值显示 —。
原始 BREAKDOWN 及其对账保留在下方来源中。</p>
{warning_html}
<div class="scroll" tabindex="0" aria-label="逐 step 统计表，可横向滚动"><table><thead>
<tr><th>行</th><th>TTFT (s)</th><th>tokenizer (s)</th>
<th colspan="4" class="ec">schedule (s)</th><th class="wk">start_load_kv (s)</th>
<th class="wk">forward (s)</th><th class="wk">EngineCore save (s)</th>
<th colspan="15" class="store">所选 TP 保存明细（嵌套计时，不另加到 TTFT）</th></tr>
<tr class="total"><th>总计</th><td>{fmt(ttft)}</td>
<td id="tokenizer-value">{fmt(number(tok.get("tokenize")))}</td>
<td colspan="4">{fmt(schedule)}</td>
<td>{fmt(number(bd.get("load_kv")))}</td><td>{fmt(number(bd.get("forward")))}</td>
<td>{fmt(number(bd.get("save_kv")))}</td>
<td colspan="15" class="store">BREAKDOWN.save_kv 是各轮 EngineCore save 累计；
右侧明细固定 {escape(worker_text)}，不是请求分项之和。</td></tr>
<tr><th rowspan="2">执行轮次</th><th rowspan="2"></th><th rowspan="2"></th>
<th rowspan="2" class="ec">hash</th>
<th rowspan="2" class="ec">ext_lookup</th><th rowspan="2" class="ec">alloc</th>
<th rowspan="2" class="ec">other</th><th rowspan="2" class="wk">load</th>
<th rowspan="2" class="wk">forward</th><th rowspan="2" class="wk">save</th>
<th rowspan="2">wait_for_save</th><th rowspan="2">enqueue</th>
<th rowspan="2">join</th><th rowspan="2">task_total</th>
<th rowspan="2" class="store">q_wait</th>
<th rowspan="2" class="store">build_key</th><th rowspan="2" class="store">lookup</th>
<th rowspan="2" class="store">prepare</th><th rowspan="2" class="store">sync</th>
<th colspan="4" class="put">Mooncake put 阶段</th>
<th rowspan="2" class="store">keys (所选TP)</th>
<th rowspan="2" class="store">MiB (所选TP)</th></tr>
<tr><th class="put">start</th><th class="put">submit</th><th class="put">wait</th>
<th class="put">finalize</th></tr></thead><tbody>{"".join(body)}</tbody></table></div>
<p class="note">TTFT/tokenizer 仅在总计行展示。save 累计取 EngineCore；
保存接口、任务与 Mooncake 阶段是嵌套计时，不相互累加。
保存时间和 keys/MiB 均固定 {escape(worker_text)}（默认 TP0）；
只累计同 TP 的串行 put，不跨 TP 相加或逐列取最大值。
其他 TP 及全部 TP 数据量保留在下方分层证据和摘要中。
“—”表示缺失或无法关联，不表示 0。无任务但 enqueue 非零时仍展示实际接口开销。</p>
{hierarchy_html}
<details><summary>数据来源与统计规则</summary><ul>
<li>总计行、TTFT、原始 residual、update={fmt(number(bd.get("update")))} s
{transfer_summary}
取自 BREAKDOWN 第 {bd["_line"]} 行，不用 step 行拼造请求总量。</li>
<li>{reconciliation}</li>
<li>execution 只取源日志的 execution_id 与 request_ids。默认 prefill 在首 token 的
BREAKDOWN 前选取真实请求执行，不以空批次/decode 顺序补配。</li>
<li>保存仅通过 execution_id/task_id/PID 关联；join 行只证明该执行等待状态，
born_step 与 join 序号不当作执行身份。</li>
<li>put 原生日志用 TTFT_THREAD 的 PID/原生 TID 映射，并由同线程 store_backend_put 的
wall_ns 起止包围，缺失 wall timestamp 时才用显式起止来源行；
keys/bytes 必须同时对账。prefer_same_node 仅在同原生线程内配对，微秒除以 1000000。</li>
<li>Tokenizer 来源行 <span id="tokenizer-source">{tokenizer_source}</span>，
QUEUE 来源行 {queue_source}。
原始 residual={fmt(residual)} s；扣除已关联 render、tokenize、首次排队后为
{fmt(remaining)} s。缺少任一扣减项时不声称已完成请求分解。</li>
<li>QUEUE 仅有 initial_queue_valid=true、零 preemption、一次 QUEUED/SCHEDULED 时
用于首次排队扣减；这不是请求累计排队量，旧日志无有效性证据则不扣减。</li>
</ul>{queue_evidence}{tokenizer_evidence}</details></body></html>"""
    out.write_text(document, encoding="utf-8")
    print(f"OK -> {out} | req={request} steps={len(steps)} worker={worker_text}")
    print(
        f"TTFT={fmt(ttft)}s save_kv={fmt(number(bd.get('save_kv')))}s | {data_summary}"
    )
    for message in dict.fromkeys(warnings):
        print(f"WARNING: {message}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="一次请求的原始日志")
    parser.add_argument("-o", "--out", type=Path, default=Path("ttft_report.html"))
    parser.add_argument("--req", help="核对目标请求 ID；单请求日志可省略")
    parser.add_argument("--mode", choices=("prefill", "all"), default="prefill")
    parser.add_argument(
        "--tp-rank", type=int, default=0, help="保存明细的 TP；默认 TP0"
    )
    parser.add_argument(
        "--tokenizer-line",
        type=int,
        help="兼容旧日志的手工证据行；无业务身份不扣 residual",
    )
    args = parser.parse_args()
    try:
        render(
            parse(args.log),
            args.out,
            args.req,
            args.mode,
            args.tp_rank,
            args.tokenizer_line,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
