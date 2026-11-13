# Compact TTFT diagnostics (schema 2)

This is a breaking, isolated cutover on `feature/pd-ttft-compact-v032`.
The ongoing `feature/pd-metrics-v0.32.0` integration and its working directories
must remain untouched. No schema-1 compatibility emitter or dual-output mode is
introduced. Model/cache/scheduling semantics and official statistics stay intact.

## Display tree

- API Server: req_resolve, render, tokenize, add_request, detokenize.
- EngineCore: schedule; associated Queue time, hash, lookup, alloc, other.
- Worker: load_kv (submit, wait_for_transfer), forward, wait_for_save
  (build_key, lookup, prepare, sync, put), sampling.
- PUT: Mooncake start_batch_put, submit_transfers, wait_transfers,
  finalize_batch_put. Native total is an outer range, not a fifth phase.

`lookup` is one value (local + external non-overlapping cumulative ranges from
one completed schedule). Queue/hash are associated observations, not mutually
exclusive subintervals of schedule. Saving task detail can outlive its entry
call and response. detokenize is outside the official first-token endpoint.

## Wire records

The Python marker remains `[PD_DIAG] `; JSON records have `version: 2`.
Every record carries event, host, pid, native_tid, run_id, timestamp_ns,
record_id, plus explicit request/execution/rank/task identity when applicable.
No monotonic comparisons across process/run domains are permitted.

- `api_summary`: API request-local stages and links; original official summary
  fields under `official`, detokenize cumulative at terminal. If HTTP and engine
  request lifecycle differ, keep explicit scope/identity rather than waiting.
- `core_summary`: one real execution, or `scope: request` terminal queue/hash
  evidence when those observations cannot belong to one execution. The latter
  does not duplicate lifetime cumulative values into every step.
- `worker_summary`: one original Worker entry (`phase: execute|sample`) with
  only retained timing stages. Execute/sample share execution identity, but the
  host does not wait for another phase merely to merge logs.
- `pool_task`: one existing task completion, retained stages and counters,
  explicit `members`, and bounded per-call `native_calls` provenance inside the
  same record. `save_submit`, background `save`, `load`, and final layerwise
  `save_wait` preserve their actual scope; IDs need not match between them.
- `pd_transfer_summary`: one receive task completion containing
  `wait_for_transfer` stage/call evidence and frozen request-generation/remote
  identity; the generation closure is a separate `scope: request_generation`
  record, not a sum or an artificial pure-wait timer.
  Metadata requests and address preparation remain `associated_stages`; neither
  creates a second `submit`. The displayed submit measures Worker start_load_kv.

A stage is `{calls, duration_ns, start_ns, end_ns, status, exceptions, ...}`.
Repeated stage duration is the sum of actual calls, not the first/last envelope.
Real zero is retained. Skipped/fused/missing/invalid are explicit and have no
invented measured duration. Status/complete/completion semantics survive all
aggregation. Retained native call details and identity links are bounded; loss
or truncation is marked incomplete with dropped counts, never silently complete.

Common helper interface (implemented in `vllm.v1.metrics.pd_diagnostics`):

- `Batch(event, **identity)`; `.update(**fields)`;
  `.add_stage(name, *, start_ns=None, end_ns=None, duration_ns=None, calls=1,
  status='ok', **attributes)`; `.add_link(kind, **identity)`;
  `.capture(event, **fields)`; `.finish(status='ok', **fields)` (idempotent).
- `collect(batch)` installs an existing batch in the local ContextVar; never
  finishes it. `current_batch()` returns it.
- `batch(event, **identity)` creates, installs, and finishes a Batch with original
  exception behavior preserved. No extra business wait or device synchronization.
- Existing `scope/current/new_id` retain explicit execution context.
- `span(event, **fields)` measures retained spans and merges them into an active
  Batch. Scheduler external lookup remains a necessary measured internal input.
  Obsolete diagnostic spans are removed at callers, not exposed as output nodes.
- `emit(event, **fields)` merges into an active batch; completed summary/task
  records may be emitted directly. It does not print old leaf-event records.

API output handlers must use the request's saved Batch explicitly, never the
ambient context of the request that first created the shared handler. Core and
Worker also restore identity from the actual scheduled/queued object. No global
unbounded registry or cross-process collector service is added.

EngineCoreRequest transports the frontend's explicit `pd_identity` dictionary.
Request construction copies it before initial hashing; scheduler admission adds
the actual engine identity. Execution members retain observation/request/engine
identity so Worker executions can join the originating API observation without
matching bare request IDs across processes. Terminal queue/hash evidence clears
unrelated ambient identity; an unfinished queue episode has no execution owner.

Scheduler `other` preserves the residual formula: schedule duration minus all
non-overlapping outer manager ranges (including internal free/remove ranges)
minus external lookup calls. The one displayed lookup stage sums local and
external calls; nested manager ranges are counted only by their outer call.

## 每请求日志量

完整计数规则与带条件的容量例子见
[metrics.md 的每请求诊断日志量](metrics.md#每请求诊断日志量按实际调用计数)。
诊断 stdout 按实际 scope/native call 计数：
`H+U+Q+T+Nc+sum_r(Er+Sr+Tr+Br+Dr+Gr)+L+W`。
单请求、9 次实际 execution、TP8/PP1、每步独立 execute+sample、无 Pool/hybrid/
原生写时是 `4+9+8*(9+9)=157` 行；单 Worker rank 文件只有 18 条 Worker summary。
DP2 的一个请求只路由一个 replica，不自动乘 2。Pool 准备/提交/后台任务/最终等待、
独立 bookkeeping、D link/task/generation 与原生 Put/Upsert 按实际次数另加。
共享 batch/task 不能按成员重复相加，后续 decode execution 继续增加日志；异常、取消、
skipped 和后台未收尾按实际到达的 scope 解释。条目数有界不代表字节数恒定，容量按
完整采集的 UTF-8 行（含真实前缀/换行）求和；synthetic helper 对比不是完整请求实测。

## Reader and UI

Only schema 2 is accepted by the new standalone conversion tools. They do not
reconstruct/fake schema-1 records. Standard-library-only Python, no torch/vLLM
runtime imports. Request, execution, rank, task and native call samples remain
distinct; P and D are linked only through explicit producer/consumer identities.

`pd_metrics_report.py` outputs canonical `dashboard.nodes`; `pd_grafana_json.py`
exports `nodes`, `requests`, `observations`, `tasks`, `pd_transfers`, `quality`.
A node has node_id, parent_id, request_key, side (P/D/null), level, node_key,
label, path, scope, repository, kind, state, duration_ns, duration_ms,
execution_id/task_id/rank, source_observations, status/completion/timing_scope,
child_count. Groups/events/association links have no duration.

The HTML report selects one request at a time and uses four merged header rows:
L1 API/EngineCore/Worker; L2 stages; L3 Pool preparation and put; L4 native put
phases. TP/PP ranks appear only on Worker observations, not API/Core rows.
Steps are local execution ordinals, with original IDs retained in evidence.
Multiple tasks remain separate; missing/unmeasured values never become zero.
HTML headers and Grafana use the same canonical `hierarchy` and `l1`–`l4`
fields. Core lookup/alloc/other are schedule children; submit/wait_for_transfer
are load_kv children. Native phases are direct put children at L4; call_id,
native_call and native_total remain provenance attributes, not wrapper nodes.
The matrix includes detokenize and preserves separate load/save task samples.
Both surfaces also expose all captured stages, associated_stages, counters,
official fields, native_calls, completions and associated native fields in a
separate field-evidence table, retaining raw values, units, status and source.
These observations do not introduce business stages or additive TTFT terms.
The provisioned Grafana template includes the global hierarchy, full execution
and rank identities, and complete field evidence; runtime dashboards use the
same panels with only deployment identity/datasource selection changed.

Worker stages are read from their actual keys regardless of execute/sample
summary phase. V2 can record connector submit/save timing during sample; it
must not be discarded or replaced by a synthetic missing execute-stage entry.
Absent Worker keys produce no fabricated stage nodes. An unconfigured connector
is not inferred merely from an absent stage or missing role field. Blank header
cells are layout only; there are no invented L3/L4 "call" stages.

Queue episodes are displayed with the execution that ends their wait, not as
time spent inside that execution. Hash and detokenize call ranges retain their
explicit execution identity; ranges without one stay request-associated.
Request cumulative totals remain source evidence and are not copied into steps.
Bounded range loss retains dropped counts and incomplete state. Older captured
summaries without ranges remain request-level observations, never reconstructed.
Pool stages with zero calls stay in field evidence but do not create business
stage nodes. Measured zero-duration calls remain visible.

`add_request` starts at the actual tokenizer call's return in its executing
thread and ends when API Server `await engine_core.add_request_async` returns
(or raises/cancels). Its `timing_scope=tokenizer_end_to_ipc_await_return`
includes token construction, post-tokenization transforms, renderer processing,
request validation/registration, fanout and IPC submission after that boundary.
The per-prompt `pd_tokenization` carrier follows rendered `EngineInput` into
`EngineCoreRequest`; it is independent of ambient batch/thread state. Async
executor result-delivery delay is included. Encoder/decoder inputs use the last
actual tokenizer return. Saved host/pid/run identity prevents foreign-process
monotonic comparisons. Completions/chat and direct raw AsyncLLM prompts use the
same carrier; each streaming input has its own boundary, excluding waits before
that input's tokenizer. The final empty stream signal has a fresh alternative
local-registration boundary so earlier stream waits cannot leak into its timer.

Pretokenized/embeddings inputs explicitly use `boundary_status=skipped`, fused
Harmony uses `fused`, and unavailable tokenizer boundaries use `missing` with
an explicit reason. These measured alternative ranges have
`timing_scope=alternative_boundary_to_ipc_await_return`; they never claim a true
tokenizer endpoint. Missing or foreign-domain carriers fall back to an explicitly
identified local boundary. No global last-tokenizer timestamp is used.

Each `add_request_send` link retains request/boundary identity, the full original
start/end/duration, status, alternative reason and accounted range. For n>1,
`shared_preparation=true` and `aggregation=union_per_input_boundary` mean the
submission stage charges the union of overlapping child ranges exactly once,
while each child send retains its complete tokenizer-to-return evidence. Full
child ranges must not be summed. Distinct prompts keep distinct boundaries and
submission owners. Links remain bounded with the Batch dropped count. Send
exceptions and cancellation preserve original propagation and timing endpoints;
failure reporting/cleanup after the await is excluded. HTTP/request summaries
do not emit fabricated missing copies.
Detokenize call ranges receive execution/output-batch identity from the actual
EngineCoreOutput, including the popped scheduler output on the async path.
Unscheduled error outputs have no execution identity. Both fields are appended
nullable wire defaults; old captured outputs are not assigned guessed steps.

The Grafana snapshot adapter offers request/side/node drilldown. Three L1 groups
only; no duplicate worker/runner wrapper nodes or per-field/event business tree.
HTML uses the same canonical nodes and is auxiliary. No online response/query
completion service is part of this change.
The canonical table exposes repository and expandable JSON source observations.
Explicit producer/consumer evidence retains canonical endpoint keys after request
merging, so request-filtered links preserve the original engine/request identities.


## Verification and release

Prove original exceptions, cancellation, repeated/zero/skipped spans, mixed
requests, execute/sample, task/member closure, native repeated-call and scope
association. Prove actual CLI and Grafana P/D navigation on deterministic
synthetic logs. Compare schema-1 versus schema-2 line count and bytes for the
same exercised workload; do not treat synthetic times as performance results.
NPU serving, model eval and D production verification are separate, unverified
unless actually exercised. Package only matched new branch files with base
commit/branch manifest and checksums; do not overwrite the ongoing integration
bundle or start/stop its services.

### Local final verification (2026-11-12)

Using `uv run --no-project <V>/.venv/Scripts/python.exe`, the final local
standalone runs passed:

- `-m pytest tests/v1/metrics/test_pd_diagnostics.py tests/v1/metrics/test_pd_compact_core.py tests/v1/metrics/test_pd_compact_batch.py --confcutdir=tests/v1/metrics -q`: 53 passed.
- `-m pytest tests/tools/test_pd_compact.py --confcutdir=tests/tools -q`: 10 passed.
- From the Ascend checkout, `tests/ut/distributed/ascend_store/test_compact_diagnostics_standalone.py`: 7 passed. Its logged cancellation traceback is exercised exception-path evidence; the suite exits successfully.
- `-m ruff format` left all 25 changed/new vLLM Python files and all 12
  changed/new Ascend Python files unchanged; `-m ruff check` passed for both sets.
- After the final browser/UI and log-volume documentation updates, the full
  reader suite passed again: 10 passed. Ruff format checks and lint passed for
  the same 25 vLLM and 12 Ascend Python files. Earlier API/Core/helper/Ascend
  results above apply to their unchanged source; they were not rerun needlessly.

These isolated local unit/reader checks include synthetic input. Real A3 serving
was subsequently exercised as recorded below. They do not establish model
evaluation, production D behavior, native Mooncake compilation, or runtime
performance. The metrics and tools suites deliberately use separate
`--confcutdir` values to avoid the unavailable `tblib` root-test dependency.

The final local delivery is a new timestamped directory beneath
`D:/work/ttft-compact-delivery`, containing per-repository changed-file archives,
tracked binary patches, new-file patches, base commit/blob hashes, SHA-256 file
and artifact checksums, and exact command/output evidence. Earlier source
archives are retained unchanged and are not the final delivery snapshot.

### Linux source archive use

The earlier full source archives preserve Windows checkout line endings.
Their `.sh` files can contain CRLF; the A3 build required `dos2unix` before
executing Linux build scripts. After extraction into a new isolated directory,
convert shell scripts before building, for example:

```bash
find /path/to/isolated/source -type f -name '*.sh' -exec dos2unix {} +
```

The final changed-file archives likewise preserve exact checkout bytes, with
matching SHA-256 hashes; no normalized source hashes are claimed. Apply tracked
and new-file patches to the manifest's matching base checkout, or overlay the
changed-file archive on that base (honoring any manifest deletions), then perform
the Linux shell-script conversion. Earlier full source archives are stale and
are not substitutes for the final matched changes. The package's UI evidence
contains synthetic-log browser screenshots and the browser verification JSON.
The separate `a3-evidence` directory contains real single-server NPU serving
logs and acceptance results; neither establishes disaggregated P/D transfer.

### A3 deployment checks (2026-11-12, single-server serving passed)

The allocated A3 was started and connected. DeepSeek-R1-Distill-Qwen-1.5B
weights were downloaded (3,554,214,621 bytes). The isolated Python 3.11.4
environment uses PyTorch 2.10.0, torch-npu 2.10.0.post7 and Triton Ascend
3.6.0.post2. Actual EngineCoreRequest, Request and OutputProcessor imports,
the pd_identity serialization field, and Linux syntax compilation passed.
The host-only Ascend producer harness also passed all seven checks on A3.

The initial CANN 9.0.0 image could execute an NPU tensor but could not compile
the current custom operators. Matching the A3 Dockerfile required a separate
CANN 9.2.1 Toolkit/Ops installation, Transformer Ops 9.3.0 and NNAL 9.2.1.
An NPU tensor operation passed after that installation. The old CMake directory
retained 9.0.0 include paths; it was preserved separately and a clean build
started against 9.2.1. These are environment checks, not model serving proof.

The text-only smoke environment pins OpenCV 4.11.0.86: the repository's OpenCV
requirement needs NumPy 2 while Triton Ascend pins NumPy 1.26.4. Repository
requirements remain unchanged; this deployment therefore has a documented
dependency exception and does not validate video/multimodal paths. Overlapping
community Triton was removed and Triton Ascend reinstalled per the installation
guide. Single-server A3 serving and real request log counts passed below;
Pool/native writes and disaggregated P/D transfer remain unexercised.

The final review found one pooling-only reporting gap: `AsyncLLM.encode()`
cancel/error cleanup did not record the original diagnostic status before
abort. It now follows `generate()`; the API regression suite passed 27 checks
after adding both transitions, with lint and formatting checks passing for
the two changed files. Earlier Core/helper checks remain applicable unchanged.

The clean native Ascend build completed and installed the source extension and
custom operator package. The required FLA dependency was built from the pinned
`v26.9.1-release` tag, commit `5cc699069c7f91a35aabd583032c8720a0b626fa`,
against CANN 9.2.1 and the host glibc 2.31. The build script emits a fixed
`manylinux_2_34_aarch64` wheel tag; after inspecting every generated `.so`
(maximum required GLIBC symbol version 2.29), its local wheel metadata was
corrected to `manylinux_2_31_aarch64` before installation. This correction
applies only to the locally compiled artifact, not the official binary wheel.

`/health` and `/v1/models` returned 200. Real `/v1/completions` and
`/v1/chat/completions`, each streaming and nonstreaming, passed protocol,
nonempty content, finish and explicit API/Core/Worker identity checks. All four
preserved the server's official timing fields. The completion response ID is
associated with its indexed engine input through the HTTP `api_request` link;
the acceptance reader follows that explicit link rather than guessing suffixes.
This is serving/diagnostic proof, not an accuracy evaluation or an exact-OK test.

The complete run emitted 718 schema-2 diagnostic records: 12 `api_summary`,
240 `core_summary`, and 466 `worker_summary`. Physical diagnostic log lines,
including logger prefixes and LF endings, occupied 1,170,404 bytes. With TP=1,
PP=1 and separate execute/sample summaries, per-request counts were:

| Endpoint | Streaming | Executions | Diagnostic lines | Bytes |
| --- | --- | ---: | ---: | ---: |
| completions | no | 64 | 196 | 319,749 |
| completions | yes | 64 | 196 | 319,905 |
| chat/completions | no | 51 | 157 | 257,297 |
| chat/completions | yes | 51 | 157 | 257,521 |

The four request totals account for 706 lines and 1,154,472 bytes. The remaining
12 lines (15,932 bytes) belong to six empty `no_forward=true` executions with
no request members; they are included only in the whole-run total.

These are whole-request counts including decode, not TTFT-only counts or a
schema-1/schema-2 performance comparison. Streaming client first nonempty
content arrived at 105.39 ms and 62.71 ms respectively; nonstream observations
occur after the full JSON body and must not be labeled TTFT. Startup/JIT and
prefix reuse make these smoke measurements unsuitable as benchmark results.
The evidence archive includes the raw server log, HTTP/SSE responses, exact
identity bridges, build log and launch script. Pool, native Mooncake writes,
disaggregated D execution and multimodal inference remain unexercised.

### Actual Mooncake Pool and dashboards (2026-11-12)

The instrumented Mooncake source was compiled with Ascend Direct, GCC 12 and
an isolated rdma-core 54 userspace dependency, then installed under the isolated
A3 workspace. A real master and `AscendStoreConnector` service were started.
A 1201-token request wrote nine complete 128-token blocks: nine keys and
33,030,144 bytes. `pool_task` reported `succeeded`, with nine successful native
results and zero failures. The uniquely associated `BatchPutTiming` contained
all four native phases (190, 461, 1757 and 103 microseconds respectively).

A second independent vLLM instance, with no local cache from the writer,
found 1152 external tokens and attempted actual loading. It did not pass:
explicit RoCE transfer timed out after 60000 ms. NPU 10 to 11 and 11 to 10
device pings each sent three packets and received zero. The alternative HCCS
path failed `aclrtIpcMemImportByKey` with 107000; fabric-memory registration
reported unsupported cross-server communication. These are observed deployment
failures, not successful cache reuse. Independent P/D transfer remains blocked
on a working device transfer path and has not been claimed as verified.

Both HTML and Grafana dashboards were opened with real captured logs. Grafana
displayed the real request's official statistics, Core execution drilldown,
and all four native put phases under the uniquely associated Pool task.
`http://127.0.0.1:3033/d/a3-pool-real` uses the separate real Pool datasource;
the earlier synthetic dashboard remains separate. These are captured snapshots,
not automatic log tailing. Actual logs, failures, rows, report and screenshot
are retained in the delivery's `a3-pool-evidence` directory.
