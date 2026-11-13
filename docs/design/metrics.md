# Metrics

vLLM exposes a rich set of metrics to support observability and capacity planning for the V1 engine.

## Objectives

- Provide comprehensive coverage of engine and request level metrics to aid production monitoring.
- Prioritize Prometheus integrations, as this is what we expect to be used in production environments.
- Offer logging support (i.e. printing metrics to the info log) for ad-hoc testing, debugging, development, and exploratory use cases.

## Compact P/D diagnostic evidence (schema 2)

This isolated cutover on `feature/pd-ttft-compact-v032` emits `[PD_DIAG]`
JSON records with `version: 2`. The original model, scheduling, cache behavior,
exceptions, cancellation and official statistics remain unchanged. Diagnostics
have overhead; they do not establish additive TTFT attribution or NPU performance.
The ongoing schema-1 integration, its working directories, bundles, Grafana on
3032 and rows API on 8765 must remain untouched. The compact readers reject
schema 1, mixed-version input and old leaf-event records, without reconstruction.

### Retained tree and identity

The report's `dashboard.nodes` is the sole tree used by HTML, request export and
Grafana. Each request/side has three L1 groups:

- **API Server:** req_resolve, render, tokenize, add_request, detokenize.
- **EngineCore:** schedule with lookup, alloc, other; request queue and hash are
  associated cumulative observations, retained once rather than copied per step.
- **Worker:** load_kv (submit, wait_for_transfer), forward, wait_for_save
  (build_key, lookup, prepare, sync, put), sampling. Pool tasks preserve their
  actual save_submit, save, load and save_wait scope separately.

Execute/sample entries, execution IDs, ranks, task IDs, request generations and
native calls remain distinct samples. Shared batch/task durations are never
divided among members or added across TP ranks. Stage `duration_ns` is the sum
of actual repeated calls, not the first-to-last envelope. Real zero remains
measured zero. Missing, skipped, fused, invalid and not-started stages have null
display durations. Groups, positions and association links are not timers.
`status`, `complete`/`completion`, exceptions, timing scope and bounded-detail
drops remain visible; truncated evidence is incomplete.

An HTTP `api_summary` can finish without an engine request. Its req_resolve,
render and tokenize stages stay on an observation-only entry, with explicit links
to engine_submission/engine_input and request children. They are not cloned into
each child's request timers. The saved `pd_identity` on EngineCoreRequest carries
the API observation identity into Core membership and terminal records. Workers
join through the unique Core execution_id and its explicit members, never by
global request-ID matching. P and D associate only through explicit producer and
consumer engine/request identities. Filenames, thread ownership, nearby logs and
similar timestamps do not establish request or side identity.

Official API fields stay under `official` in their original units/formulas.
Side-specific display chooses a unique official summary, never sums P/D. When
`first_token_observed` is false or missing, displayed TTFT stays null even if the
original native TTFT field is nonzero. Terminal cumulative detokenize is outside
the official first-token endpoint. An unavailable authoritative add_request total
remains missing. Receive-task submit and wait_for_transfer measure actual calls;
transfer_envelope_ns and request_generation closure are separate evidence, not a
synthetic pure-wait duration or a request sum.

### Native Mooncake evidence

Keep complete `[BatchPutTiming]` and `[BatchUpsertTiming]` lines alongside Python
summaries. Each matched PUT call has exactly four native children:
start_batch_put, submit_transfers, wait_transfers and finalize_batch_put. Native
total is outer-range metadata on that call, never a fifth phase. `normal_full`,
`prefer_same_node_full` and `prefer_same_node_not_started` remain distinct. An
incomplete/not-started record may have no phase/total durations; none are invented.

Matching requires a retained native_call boundary, native TID, source wall-clock
start/end and UTC offset. Glog local time is converted with the producer's offset,
including printed timestamp precision; the reporting host's timezone is unused.
The same original source file is required unless native host/PID/run are explicit
and agree. Available submitted keys/bytes must agree. Overlapping candidates,
multiple native samples per call, missing offsets, wrong threads/windows or count
mismatches remain unassigned quality evidence. Preserve rotated process logs and
do not provide both stdout and file copies of the same native lines.

### 每请求诊断日志量（按实际调用计数）

以下估算只计算 schema 2 的 `[PD_DIAG]`、Mooncake `[BatchPutTiming]` 和
`[BatchUpsertTiming]` 诊断记录。假设 INFO 日志已开启、生命周期正常收尾、每个进程的
原始输出只采集一次；不包含 access log、常规服务 INFO/WARNING、异常 traceback、
启动/模型加载日志。这里的一“行”指一次诊断 logger 输出，既不是一个树节点，也不是一次
stage 调用。`Batch.add_stage()` 的重复调用累加到同一 summary；`Batch.finish()` 幂等，
不会因为同一对象重复 finish 而增加行数。

先按实际 producer 定义变量；所有变量都是采集范围内的记录/调用次数：

| 变量 | 实际 producer 与计数规则 |
| --- | --- |
| `H` | `PDRequestMiddleware.__call__`：每次匹配的 `/v1/chat/completions` 或 `/v1/completions` HTTP 生命周期，1 条 `api_summary(scope=http)`。请求在校验阶段被拒绝也可以只有这一条。 |
| `U` | `_api_submission`：每次提交生命周期 1 条 `api_summary(scope=engine_submission)`；流式输入使用 `scope=engine_input`，按实际进入次数计。 |
| `Q` | `RequestState` / `OutputProcessor._finish_pd_request`：每个已创建且收尾的引擎请求状态 1 条 `api_summary(scope=request)`，其中 detokenize 是累计值。 |
| `T` | `Request.pd_queue_transition("finished")`：每个收尾的 Core 请求 1 条 `core_summary(scope=request)`，保留 queue/hash 等关联证据。 |
| `Nc` | `EngineCore._pd_schedule` / `Scheduler.schedule`：每次真实调度执行 1 条 `core_summary(scope=execution)`；正常 EngineCore 路径复用同一个 Batch，不再多发一条 Scheduler summary。空调度、失败调度也按实际进入并收尾计。 |
| `Er`, `Sr` | rank `r` 上 `pd_execution`、`pd_sample` 包装的实际入口次数：分别产生 `worker_summary(phase=execute)`、`worker_summary(phase=sample)`。 |
| `Tr` | Pool rank `r` 上 `diagnostics.Task.finish` 的实际 task summary 数，包括提交、准备、后台 save/load、最终 `save_wait` 等各自已有 scope。 |
| `Br` | Pool `diagnostics.completion` 在没有 active Task 时发出的 `pool_task(scope=bookkeeping, operation=store_completion)` 数。active Task 内的 completion 放入 `completions` 数组，不额外发一行。 |
| `L` | hybrid connector 的 `pd_transfer_summary(scope=link)` 数：P 的 `request_finished_all_groups` 发 producer offer，D 的 `build_connector_meta` 发 consumer link，均按实际触发计。 |
| `Dr`, `Gr` | D rank `r` 的 `_pd_finish_task`：每个处理并收尾的 receive task 1 条 `scope=task`；一个 request_generation 在 pending 归零且 final task 已入队时另发 1 条 `scope=request_generation`。 |
| `W` | 所有采集进程实际输出的原生 `Client::BatchPut` / `Client::BatchUpsert` timing 行数；按 native batch call 计，不按 key、block、stage 或 Python task 计。 |

跨全部相关 rank 的诊断 stdout 总量为：

```text
Npython = H + U + Q + T + Nc + sum_r(Er + Sr + Tr + Br + Dr + Gr) + L
Ndiag   = Npython + W
```

没有 Pool 时令 `Tr=Br=0`；没有 hybrid receive 时令 `Dr=Gr=L=0`；没有实际原生写
timing 输出时令 `W=0`。Pool 和 hybrid 是按启用路径加上的项，不能假定每个请求都走两者。
原生 `LogBatchWriteTiming` 一次输出包含四个 phase 和 total，一次调用仍是 **1 行**，
不是 5 行；转发重载不重复输出。prefer-same-node 校验未开始的错误分支也输出 1 条
`phase_status=not_started`，不制造 phase 时长。native READ 的调用明细在 D task
的 `native_calls` 数组内，本节这两个原生 timing marker 不为 READ 另发一行。

**9 步、TP8、PP1 的具体例子。** 设一个 HTTP 对应一次提交、一个引擎请求，正常收尾，
所以 `H=U=Q=T=1`；`Nstep=9` 指这个请求实际参与的 9 次 Core execution，
`Nc=9`，8 个 TP rank 各进入 execute 9 次。对于每步都进入独立 sample 的正常路径，
`Er=Sr=9`，不启用 Pool/hybrid、不发生原生写时：

```text
Ndiag = 4 + 9 + 8 * (9 + 9) = 157 行
      = API 3 + Core 请求终态 1 + Core execution 9 + Worker 144
```

这是该假设下完整生命周期的粗估，不能理解为每请求固定 157 行。若这 9 步只有 `s` 次
进入 sample，公式是 `4 + 9 + 8*(9+s)`；例如 `s=0` 是 85 行，`s=1` 是
93 行，`s=9` 是 157 行。`EngineCore.step` 只有在 execute 返回 `None` 时才调用
`sample_tokens`；sample 即便没有可采样 token，只要进入包装器仍有一条 summary。
DeepSeek 的 MTP1、`enforce_eager=true`、block size 128 本身不额外乘行数：一次入口内
的重复 forward/sampling 调用合并到 stage；它们会影响实际步数、task 数和 native call 数。
9 只是这里的输入例子，不能由 block size、模型名或 speculative token 数直接推成固定步数。
DP2 中一个请求路由到其中一个 replica，**不再乘 2**；另一 replica 的独立请求或对齐/
空执行日志若也进入同一 stdout，要另按其实际记录计入，不能归给本请求。

仅一个 Worker rank 的独立文件，在上述 sample 全进入的例子中是 `9+9=18` 条
`worker_summary`；它不会自动含有另外七个 rank 的 126 条。若采集工具恰好将 API、
Core 和该 rank 合并到一个文件，则此例是 `3+1+9+18=31` 条；真正数量取决于文件
包含哪些进程，不能把“一个文件”默认当成“全 TP”。一个 Core execution 或 Worker
summary 可以关联同批多个请求，stdout 中仍只有原来的一条。给每个请求展示同一共享
记录后再相加会重复计数；吞吐容量估算应按唯一原始记录/`record_id` 去重。

**启用 Pool 的加法。** 单请求、每 rank 工作相同的简化情况下，设每 rank 有
`Tload` 个 load/load_submit scope、`Tsubmit` 个 save_submit scope、`Tsave` 个后台
save scope、`Tprep` 个其它准备 scope、`Twait` 个最终 save_wait scope、`B` 个独立
bookkeeping scope，以及实际 `Kput` 次有 timing 输出的原生 Put/Upsert。则在基础量上加：

```text
Npool = 8 * (Tload + Tsubmit + Tsave + Tprep + Twait + B + Kput)
Ndiag = Nbase + Npool
```

例如选择普通非 layerwise 路径，并明确假设 9 步每 rank 各有一个 load scope（允许
skipped）、一个 save_submit、一个后台 save，且每个 save 恰有一次原生写，则
`Tload=Tsubmit=Tsave=Kput=9`、`Tprep=Twait=B=0`：新增
`8*(9+9+9+9)=288` 行，总计 `157+288=445` 行。若每个 save 实际有两次原生写，
其它条件相同，则 `Kput=18`，总计 517 行。两者是带条件的容量例子，不是对当前模型/
部署的实测或上界。命中缓存、无可保存请求、异步 load、分组写入和不同后端都会改变这些项。
`KVPoolWorker.start_load_kv` 的 skip 在已进入的 scope 内仍输出 1 条，
`KVPoolWorker.wait_for_save` 无任务时也保留 save_submit；“没有数据传输”不能直接令
Pool 行数归零。layerwise dispatcher 将同一执行的多个 layer 合到 task，通常在实际
final layer 收尾；不能直接按层数乘 `Tr`。不过它内部每次真实原生写仍计入 `W`，
另有实际准备和最终 `save_wait` scope，不能把一个 task 当成只发生一次写入。

**D 侧与后续 decode。** 不使用 Pool 的 hybrid 例子：P 有上面的 9 步，D 再参与 1 步，
两侧各有独立的 HTTP/提交/引擎请求/Core 终态，TP8、PP1，且每步都 sample；再假设
P offer 和 D consumer link 各 1 条，每个 D rank 只有一个 receive task、一个正常闭合
的 request_generation，实际没有 Put/Upsert：

```text
P/D 基础量 = 2*4 + (9+1)*(1+2*8) = 178
hybrid 增量 = 2 + 8*(1+1) = 18
P/D 合计 = 196 行
```

若两侧不是各自经过 HTTP，必须用实际 `H/U/Q/T` 替换，不强凑 8 条固定项。receive
task 可以因分段、offset、远端划分而多于一个；generation 可以跨执行，不能每步重复加
一次闭合 summary，也不能把一个请求的多个 generation 强行合成一个。后续每增加一个
正常 execute+sample decode execution，TP8/PP1 基础量再加 `1+8+8=17` 行；执行
没有 sample 时加 9 行。Pool/task/native 的增量仍另计，输出 token 数与 execution 数
并非一一对应，尤其启用 MTP 时。保留日志直到后台任务收尾；HTTP 返回或首 token 出现
时，生命周期总量可能尚未全部打印。

取消、错误或 skipped 不改变“一个已进入并收尾的 scope 一条”的规则，但会改变走到的
scope 数：校验拒绝可只有 HTTP；提交失败可能没有请求终态；调度错误可能没有 Worker；
Worker execute 错误可能没有 sample。Pool 中未处理的已排队工作可以在清理时发
`not_processed`/incomplete；D generation 没有 final task 或尚有 pending 时不会闭合。
进程强退、日志过滤或截断也可能缺记录。因此上述成功路径数字不能充当异常路径的最低量
或完整性检查。`request_id` 字符串过滤会漏掉 `request_id=null`、`request_ids`、
`members`、execution 关联的 Worker 和不带 request ID 的原生行，统计应保留完整原始流。

**字节数。** 当前没有覆盖这个完整请求路径的可信实日志/代表性序列化分布，因此不承诺
“每行几 KB”或“每请求固定多少 KB”。实际文件体积按采集后的行文本计算：

```text
Bdiag = sum_i len((formatted_diagnostic_line_i + actual_line_ending_i).encode("utf-8"))
```

其中须包括 Python logger 或 glog 的真实前缀、JSON、换行；若计算整个服务日志文件，
还要加所有非诊断行。仅 Python payload 的编码使用 `_write` 中的
`json.dumps(values, separators=(",", ":"))`；不能据此漏掉日志前缀。`Batch` 的
128 个 stage/link 上限、Pool 的 128 个 member/completion 和 64 个 native-call detail
上限、D 的 64 个 READ detail 上限只限制保留条目数，截断有 dropped/incomplete 标记。
ID/异常文本长度、batch 的 `request_ids`/`num_scheduled_tokens`、members 和关联数组
都会改变长度；这些有界明细不代表每行恒定字节数或所有字段都有统一字节上限。

已有的 **synthetic accumulator demo** 仅对同一 helper 中 20 次 forward + 20 次
sampling 做过对比：schema 1 为 40 行/16,790 bytes，schema 2 为 2 行/1,198 bytes。
它不是完整请求，不能拿 1,198 bytes 乘上此处请求数当成部署容量，也不是 TTFT/吞吐/
NPU 开销的测量。部署容量先用这里的条件行数规划，再从完整原始采集计算上式。

### Standalone conversion

All tools use only the Python standard library, without importing torch or the
vLLM runtime. Supply each original log once. Output parent directories must exist.

```bash
PD_PYTHON=/usr/local/python3.12.16/bin/python
PD_OUTPUT=/vllm-workspace/pd-compact-output
mkdir -p "$PD_OUTPUT"
"$PD_PYTHON" tools/pd_metrics_report.py prefill.log decode.log -o "$PD_OUTPUT/report.json" --html &&
"$PD_PYTHON" tools/pd_request_json.py prefill.log decode.log -o "$PD_OUTPUT/requests.json" &&
"$PD_PYTHON" tools/pd_grafana_json.py "$PD_OUTPUT/report.json" -o "$PD_OUTPUT/rows.json"
```

Alternatively, use `pd_grafana_json.py --logs prefill.log decode.log -o rows.json`.
The export contains `nodes`, `requests`, `observations`, `tasks`, `pd_transfers`
and `quality`, plus explicit transport links. Request JSON is always an array of
schema-2 request/observation objects containing the same canonical nodes. Report
HTML is auxiliary and written beside report JSON. Invalid or absent diagnostic
records exit 2 before output writes; a pre-existing file may still be stale.
Stop on failure. Supplying P alone does not synthesize D.

### Isolated Grafana snapshot

The source delivery includes `tools/grafana_rows_api.py`,
`tools/grafana_pd_compact_dashboard.json` and
`tools/deploy_pd_compact_dashboard.py`. These replace no live integration assets.

```bash
"$PD_PYTHON" tools/grafana_rows_api.py "$PD_OUTPUT/rows.json" --host 127.0.0.1 --port 8766
# In another terminal, prepare independent data/log/provisioning directories:
"$PD_PYTHON" tools/deploy_pd_compact_dashboard.py "$PD_OUTPUT/grafana" --api-url http://127.0.0.1:8766 --plugins /path/to/existing/grafana/plugins
```

Start the existing Grafana binary using the generated `grafana.ini` and its normal
homepath. It binds to loopback port **3033** with independent data and log paths.
The generator starts/stops no services. JSON API plugin
`marcusolsson-json-datasource` must already be available. For remote/container
Grafana choose an API URL reachable by that Grafana server; loopback names its
own container/host. Do not point the original API at the new output.

The adapter loads once at startup; restart only the isolated adapter after new
conversion. `/requests`, `/sides?request_key=KEY`,
`/node_options?request_key=KEY&side=P` and
`/nodes?request_key=KEY&side=P&parent_id=__root` implement Request → Side →
Parent/node drilldown. A node label opens its children; Parent node → L1 groups
returns to the three roots. The links panel navigates HTTP observations to actual
requests. Official statistics select one side. Empty arrays mean no evidence,
not measured zero. The snapshot is not filtered by Grafana's time picker.

### Deterministic synthetic smoke input

```bash
"$PD_PYTHON" tools/pd_compact_fixture.py "$PD_OUTPUT/fixture"
"$PD_PYTHON" tools/pd_metrics_report.py "$PD_OUTPUT/fixture/synthetic-P.log" "$PD_OUTPUT/fixture/synthetic-D.log" -o "$PD_OUTPUT/report.json" --html
"$PD_PYTHON" tools/pd_grafana_json.py "$PD_OUTPUT/report.json" -o "$PD_OUTPUT/rows.json"
```

This deterministic fixture exercises explicit P/D links, HTTP observation-only
bridges, mixed requests, execute/sample ranks, repeated/zero/skipped/error stages,
request-generation/task separation, bounded drops and native matching/mismatch.
It is **SYNTHETIC**, not inference, throughput, TTFT or a real D production run.
`tests/tools/test_pd_compact.py` covers the reader/export/adapter contracts and
actual CLI outputs. Parent release verification must separately exercise the
isolated Grafana/API surfaces. CPU synthetic checks do not validate native C++
execution, NPU serving, model output, or distributed performance.

Capture real logs on each deployment with the existing launch command and
`VLLM_LOGGING_LEVEL=INFO PYTHONUNBUFFERED=1 GLOG_logtostderr=1`. Retain complete
API/Core/Worker and native streams until asynchronous tasks finish, with a
collection manifest containing branch/base/commit, host, source timezone and
wheel identity. Do not filter to `[PD_DIAG]` alone or overwrite P with D logs.
`tools/deploy_pd_metrics_d.sh` belongs to the ongoing integration's pinned
deployment workflow; it is not a compact release deployment script. A compact
source/wheel deployment requires the separately verified matched release bundle
and target-stack validation, with no service restart implied by conversion.

## Background

Metrics in vLLM can be categorized as follows:

1. Server-level metrics: Global metrics that track the state and performance of the LLM engine. These are typically exposed as Gauges or Counters in Prometheus.
2. Request-level metrics: Metrics that track the characteristics (e.g. size and timing) of individual requests. These are typically exposed as Histograms in Prometheus and are often the SLOs that an SRE monitoring vLLM will be tracking.

The mental model is that server-level metrics help explain the values of request-level metrics.

### Metrics Overview

### v1 Metrics

In v1, an extensive set of metrics are exposed via a Prometheus-compatible `/metrics` endpoint using the `vllm:` prefix, for example:

- `vllm:num_requests_running` (Gauge) - Number of requests currently running.
- `vllm:kv_cache_usage_perc` (Gauge) - Fraction of used KV cache blocks (0–1).
- `vllm:prefix_cache_queries` (Counter) - Number of prefix cache queries.
- `vllm:prefix_cache_hits` (Counter) - Number of prefix cache hits.
- `vllm:prompt_tokens_total` (Counter) - Total number of prompt tokens processed.
- `vllm:generation_tokens_total` (Counter) - Total number of generated tokens.
- `vllm:request_success_total` (Counter) - Number of finished requests (by finish reason).
- `vllm:admission_rejections_total` (Counter) - Admission attempts rejected by a queue limit, by reason.
- `vllm:request_prompt_tokens` (Histogram) - Histogram of input prompt token counts.
- `vllm:request_generation_tokens` (Histogram) - Histogram of generation token counts.
- `vllm:time_to_first_token_seconds` (Histogram) - Time to first token (TTFT).
- `vllm:inter_token_latency_seconds` (Histogram) - Inter-token latency (time between consecutive streamed outputs).
- `vllm:request_time_per_output_token_seconds` (Histogram) - Per-request Time Per Output Token (TPOT).
- `vllm:e2e_request_latency_seconds` (Histogram) - End-to-end request latency.
- `vllm:request_prefill_time_seconds` (Histogram) - Request prefill time.
- `vllm:request_decode_time_seconds` (Histogram) - Request decode time.

These are documented under [Inferencing and Serving -> Production Metrics](../usage/metrics.md).

### Grafana Dashboard

vLLM also provides [a reference example](../../examples/observability/prometheus_grafana/README.md) for how to collect and store these metrics using Prometheus and visualize them using a Grafana dashboard.

The subset of metrics exposed in the Grafana dashboard gives us an indication of which metrics are especially important:

- `vllm:e2e_request_latency_seconds_bucket` - End to end request latency measured in seconds.
- `vllm:prompt_tokens` - Prompt tokens.
- `vllm:generation_tokens` - Generation tokens.
- `vllm:inter_token_latency_seconds` - Inter-token latency in seconds, measured between consecutive streamed outputs. It approximates TPOT when each output contains exactly one token, but differs from request-level TPOT when an output contains multiple tokens or when aggregation weights differ.
- `vllm:request_time_per_output_token_seconds` - Per-request Time Per Output Token (TPOT) in seconds, computed as `(end-to-end latency - TTFT) / (number of output tokens - 1)`. It is recorded as zero for requests that generate no more than one token.
- `vllm:time_to_first_token_seconds` - Time to First Token (TTFT) latency in seconds.
- `vllm:num_requests_running` (also, `_swapped` and `_waiting`) - Number of requests in the RUNNING, WAITING, and SWAPPED states.
- `vllm:kv_cache_usage_perc` - Percentage of used cache blocks by vLLM.
- `vllm:request_prompt_tokens` - Request prompt length.
- `vllm:request_generation_tokens` - Request generation length.
- `vllm:request_success` - Number of finished requests by their finish reason: either an EOS token was generated or the max sequence length was reached.
- `vllm:request_queue_time_seconds` - Queue time.
- `vllm:request_prefill_time_seconds` - Requests prefill time.
- `vllm:request_decode_time_seconds` - Requests decode time.
- `vllm:request_max_num_generation_tokens` - Max generation tokens in a sequence group.

See [the PR which added this Dashboard](https://github.com/vllm-project/vllm/pull/2316) for interesting and useful background on the choices made here.

`vllm:inter_token_latency_seconds` records one sample per streamed output event
(the wall-clock gap between successive outputs), whereas
`vllm:request_time_per_output_token_seconds` is recorded once per finished request
as `(end-to-end latency - TTFT) / (number of output tokens - 1)`. These two
metrics differ when an output bundles multiple tokens (e.g. speculative
decoding) or when mean ITL and mean per-request TPOT are aggregated over
differing weights. See [Understanding the latency metrics](../benchmarking/cli.md#understanding-the-latency-metrics)
for the benchmark definitions. Requests that generate no more than one token
are recorded with a TPOT of zero, while `vllm bench serve` excludes them from
its TPOT statistics. Therefore, the Prometheus histogram and benchmark TPOT
statistics can differ. Use `vllm:request_time_per_output_token_seconds` for
request-level TPOT and `vllm:inter_token_latency_seconds` when you specifically
want the inter-output latency distribution.

### Prometheus Client Library

Prometheus support was initially added [using the aioprometheus library](https://github.com/vllm-project/vllm/pull/1890), but a switch was made quickly to [prometheus_client](https://github.com/vllm-project/vllm/pull/2730). The rationale is discussed in both linked PRs.

During those migrations we briefly lost a `MetricsMiddleware` to track HTTP metrics, but this was reinstated [using prometheus_fastapi_instrumentator](https://github.com/vllm-project/vllm/pull/15657):

```bash
$ curl http://0.0.0.0:8000/metrics 2>/dev/null  | grep -P '^http_(?!.*(_bucket|_created|_sum)).*'
http_requests_total{handler="/v1/completions",method="POST",status="2xx"} 201.0
http_request_size_bytes_count{handler="/v1/completions"} 201.0
http_response_size_bytes_count{handler="/v1/completions"} 201.0
http_request_duration_highr_seconds_count 201.0
http_request_duration_seconds_count{handler="/v1/completions",method="POST"} 201.0
```

### Multi-process Mode

Historically, metrics were collected in the engine core process and multiprocess mode was used to make them available in the API server process. See <https://github.com/vllm-project/vllm/pull/7279>.

More recently, metrics are collected in the API server process and multiprocess mode is only used when `--api-server-count > 1`. See <https://github.com/vllm-project/vllm/pull/17546> and details on [API server scale-out](../serving/data_parallel_deployment.md#internal-load-balancing).

### Built in Python/Process Metrics

The following metrics are supported by default by `prometheus_client`, but they are not exposed when multiprocess mode is used:

- `python_gc_objects_collected_total`
- `python_gc_objects_uncollectable_total`
- `python_gc_collections_total`
- `python_info`
- `process_virtual_memory_bytes`
- `process_resident_memory_bytes`
- `process_start_time_seconds`
- `process_cpu_seconds_total`
- `process_open_fds`
- `process_max_fds`

Therefore, these metrics are unavailable when `--api-server-count > 1`. It's questionable how relevant these are since they do not aggregate these stats for all processes that make up a vLLM instance.

## Metrics Design

The ["Even Better Observability"](https://github.com/vllm-project/vllm/issues/3616) feature where was where much of the metrics design was planned. For example, see where [a detailed roadmap was laid out](https://github.com/vllm-project/vllm/issues/3616#issuecomment-2030858781).

### Legacy PRs

To help understand the background to the metrics design, here are some of the relevant PRs which added the original, now legacy, metrics:

- <https://github.com/vllm-project/vllm/pull/1890>
- <https://github.com/vllm-project/vllm/pull/2316>
- <https://github.com/vllm-project/vllm/pull/2730>
- <https://github.com/vllm-project/vllm/pull/4464>
- <https://github.com/vllm-project/vllm/pull/7279>

### Metrics Implementation PRs

For background, here are the relevant PRs relating to the metrics implementation <https://github.com/vllm-project/vllm/issues/10582>:

- <https://github.com/vllm-project/vllm/pull/11962>
- <https://github.com/vllm-project/vllm/pull/11973>
- <https://github.com/vllm-project/vllm/pull/10907>
- <https://github.com/vllm-project/vllm/pull/12416>
- <https://github.com/vllm-project/vllm/pull/12478>
- <https://github.com/vllm-project/vllm/pull/12516>
- <https://github.com/vllm-project/vllm/pull/12530>
- <https://github.com/vllm-project/vllm/pull/12561>
- <https://github.com/vllm-project/vllm/pull/12579>
- <https://github.com/vllm-project/vllm/pull/12592>
- <https://github.com/vllm-project/vllm/pull/12644>

### Metrics Collection

In v1, we wish to move computation and overhead out of the engine core
process to minimize the time between each forward pass.

The overall idea of V1 EngineCore design is:

- EngineCore is the inner loop. Performance is most critical here
- AsyncLLM is the outer loop. This is overlapped with GPU execution
  (ideally), so this is where any "overheads" should be if
  possible. So AsyncLLM.output_handler_loop is the ideal place for the
  metrics bookkeeping if possible.

We will achieve this by collecting metrics in the frontend API server,
and base these metrics on information we can glean from the
`EngineCoreOutputs` returned by the engine core process to the
frontend.

### Interval Calculations

Many of our metrics are the time interval between various events in
the processing of a request. It is best practice to use timestamps
based on "monotonic time" (`time.monotonic()`) rather than "wall-clock
time" (`time.time()`) to calculate intervals as the former is
unaffected by system clock changes (e.g. from NTP).

It's also important to note that monotonic clocks differ between
processes - each process has its own reference point. So it is
meaningless to compare monotonic timestamps from different processes.

Therefore, in order to calculate an interval, we must compare two
monotonic timestamps from the same process.

### Scheduler Stats

The engine core process will collect some key statistics from the
scheduler - e.g. the number of requests that were scheduled or waiting
after the last scheduler pass - and include those statistics in
`EngineCoreOutputs`.

### Engine Core Events

The engine core will also record the timestamp of certain per-request
events so that the frontend can calculate the interval between these
events.

The events are:

- `QUEUED` - when the request was received by the engine core and
  added to the scheduler queue.
- `SCHEDULED` - when the request was first scheduled for execution.
- `PREEMPTED` - the request has been put back in the waiting queue
  in order to make room for other requests to complete. It will be
  re-scheduled in future and re-start its prefill phase.
- `NEW_TOKENS` - when the output included in `EngineCoreOutput` was
  generated. Since this is common to all requests in a given
  iteration, we use a single timestamp on `EngineCoreOutputs` to
  record this event.

And the calculated intervals are:

- Queue interval - between `QUEUED` and most recent `SCHEDULED`.
- Prefill interval - between most recent `SCHEDULED` and the subsequent
  first `NEW_TOKENS`.
- Decode interval - between first (after the most recent `SCHEDULED`) and
  last `NEW_TOKENS`.
- Inference interval - between most recent `SCHEDULED` and last `NEW_TOKENS`.
- Inter-token interval - between successive `NEW_TOKENS`.

Put another way:

![Interval calculations - common case](../assets/design/metrics/intervals-1.png)

We explored the possibility of having the frontend calculate these
intervals using the timing of events visible by the frontend. However,
the frontend does not have visibility into the timing of the `QUEUED`
and `SCHEDULED` events and, since we need to calculate intervals based
on monotonic timestamps from the same process ... we need the engine
core to record timestamps for all of these events.

#### Interval Calculations vs Preemptions

When a preemption occurs during decode, since any already generated
tokens are reused, we consider the preemption as affecting the
inter-token, decode, and inference intervals.

![Interval calculations - preempted decode](../assets/design/metrics/intervals-2.png)

When a preemption occurs during prefill (assuming such an event
is possible), we consider the preemption as affecting the
time-to-first-token and prefill intervals.

![Interval calculations - preempted prefill](../assets/design/metrics/intervals-3.png)

### Frontend Stats Collection

As the frontend processes a single `EngineCoreOutputs` - i.e. the
output from a single engine core iteration - it collects various
statistics relating to that iteration:

- The total number of new tokens generated in this iteration.
- The total number of prompt tokens processed by the prefills that
  completed in this iteration.
- The queue intervals for any requests that were scheduled in this
  iteration.
- The prefill intervals for any requests that completed prefill in
  this iteration.
- The inter-token intervals, for all requests included in this iteration.
- The Time-To-First-Token (TTFT) for any requests that completed
  prefill in this iteration. However, we calculate this interval
  relative to when the request was first received by the frontend
  (`arrival_time`) in order to account for input processing time.
  Currently `arrival_time` starts when tokenization begins.

For any requests that were completed in a given iteration, we also
record:

- The inference and decode intervals - relative to the scheduled and
  first token events, as described above.
- End-to-end latency - the interval between frontend `arrival_time`
  and the frontend receiving the final token.

### KV Cache Residency Metrics

We also emit a set of histograms that describe how long sampled KV cache
blocks stay resident and how often they are reused. Sampling
(`--kv-cache-metrics-sample`) keeps the overhead tiny; when a block is
chosen we record:

- `lifetime` – allocation ⟶ eviction
- `idle before eviction` – last touch ⟶ eviction
- `reuse gaps` – the pauses between touches when the block gets reused

Those map directly to the Prometheus metrics:

- `vllm:kv_block_lifetime_seconds` – how long each sampled block exists.
- `vllm:kv_block_idle_before_evict_seconds` – idle tail after the final access.
- `vllm:kv_block_reuse_gap_seconds` – time between consecutive touches.

The engine core only ships raw eviction events via `SchedulerStats`; the
frontend drains them, turns them into Prometheus observations, and also
exposes the same data through `LLM.get_metrics()` when logging is on.
Looking at lifetime and idle time on one chart makes it easy to spot
stranded cache or workloads that pin prompts for a long decode.

### Metrics Publishing - Logging

The `LoggingStatLogger` metrics publisher outputs a log `INFO` message
every 5 seconds with some key metrics:

- The current number of running/waiting requests
- The current GPU cache usage
- The number of prompt tokens processed per second over the past 5
  seconds
- The number of new tokens generated per second over the past 5
  seconds
- The prefix cache hit rate over the most recent 1k kv-cache block queries

### Metrics Publishing - Prometheus

The `PrometheusStatLogger` metrics publisher makes the metrics
available via a `/metrics` HTTP endpoint in a Prometheus-compatible
format. A Prometheus instance can then be configured to poll this
endpoint (e.g. every second) and record the values in its time-series
database. Prometheus is often used via Grafana, allowing these metrics
to be graphed over time.

Prometheus supports the following metric types:

- Counter: a value that will increase over time, never reducing, and
  generally reset to zero when the vLLM instance restarts. For
  example, the number of tokens generated over the lifetime of the
  instance.
- Gauge: a value that goes up and down, for example the number of
  requests currently scheduled for execution.
- Histogram: a count of metric samples, recorded in buckets. For
  example, the number of requests whose TTFT was <1ms, <5ms, <10ms,
  <20ms, and so on.

Prometheus metrics can also be labelled, allowing metrics to be
combined according to matching labels. In vLLM, we add a `model_name`
label to every metric which includes the name of the model served by
that instance.

Example output:

```bash
$ curl http://0.0.0.0:8000/metrics
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="meta-llama/Llama-3.1-8B-Instruct"} 8.0
...
# HELP vllm:generation_tokens_total Number of generation tokens processed.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{model_name="meta-llama/Llama-3.1-8B-Instruct"} 27453.0
...
# HELP vllm:request_success_total Count of successfully processed requests.
# TYPE vllm:request_success_total counter
vllm:request_success_total{finished_reason="stop",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1.0
vllm:request_success_total{finished_reason="length",model_name="meta-llama/Llama-3.1-8B-Instruct"} 131.0
vllm:request_success_total{finished_reason="abort",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
...
# HELP vllm:time_to_first_token_seconds Histogram of time to first token in seconds.
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{le="0.001",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
vllm:time_to_first_token_seconds_bucket{le="0.005",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
vllm:time_to_first_token_seconds_bucket{le="0.01",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
vllm:time_to_first_token_seconds_bucket{le="0.02",model_name="meta-llama/Llama-3.1-8B-Instruct"} 13.0
vllm:time_to_first_token_seconds_bucket{le="0.04",model_name="meta-llama/Llama-3.1-8B-Instruct"} 97.0
vllm:time_to_first_token_seconds_bucket{le="0.06",model_name="meta-llama/Llama-3.1-8B-Instruct"} 123.0
vllm:time_to_first_token_seconds_bucket{le="0.08",model_name="meta-llama/Llama-3.1-8B-Instruct"} 138.0
vllm:time_to_first_token_seconds_bucket{le="0.1",model_name="meta-llama/Llama-3.1-8B-Instruct"} 140.0
vllm:time_to_first_token_seconds_count{model_name="meta-llama/Llama-3.1-8B-Instruct"} 140.0
```

!!! note
    The choice of histogram buckets to be most useful to users
    across a broad set of use cases is not straightforward and will
    require refinement over time.

### Cache Config Info

`prometheus_client` has support for
[Info metrics](https://prometheus.github.io/client_python/instrumenting/info/)
which are equivalent to a `Gauge` whose value is permanently set to 1,
but exposes interesting key/value pair information via labels. This is
used for information about an instance that does not change - so it
only needs to be observed at startup - and allows comparing across
instances in Prometheus.

We use this concept for the `vllm:cache_config_info` metric:

```text
# HELP vllm:cache_config_info Information of the LLMEngine CacheConfig
# TYPE vllm:cache_config_info gauge
vllm:cache_config_info{block_size="16",cache_dtype="auto",cpu_offload_gb="0",enable_prefix_caching="False",gpu_memory_utilization="0.9",...} 1.0
```

However, `prometheus_client` has
[never supported Info metrics in multiprocessing mode](https://github.com/prometheus/client_python/pull/300) -
for [unclear reasons](gh-pr:7279#discussion_r1710417152). We
simply use a `Gauge` metric set to 1 and
`multiprocess_mode="mostrecent"` instead.

### LoRA Metrics

The `vllm:lora_requests_info` `Gauge` is somewhat similar, except the
value is the current wall-clock time, and is updated every iteration.

The label names used are:

- `running_lora_adapters`: a per-adapter count of the number requests
  running using that adapter, formatted as a comma-separated string.
- `waiting_lora_adapters`: similar, except counting requests that are
  waiting to be scheduled.
- `max_lora` - the static "max number of LoRAs in a single batch."
  configuration.

Encoding a running/waiting counts for multiple adapters in a
comma-separated string seems quite misguided - we could use labels to
distinguish between per-adapter counts. This should be revisited.

Note that `multiprocess_mode="livemostrecent"` is used - the most
recent metric is used, but only from currently running processes.

This was added in <https://github.com/vllm-project/vllm/pull/9477> and there is
[at least one known user](https://github.com/kubernetes-sigs/gateway-api-inference-extension/pull/54).
If we revisit this design and deprecate the old metric, we should
coordinate with downstream users so they can migrate before the removal.

### Prefix Cache metrics

The discussion in <https://github.com/vllm-project/vllm/issues/10582> about adding prefix cache metrics yielded
some interesting points which may be relevant to how we approach
future metrics.

Every time the prefix cache is queried, we record the number of tokens
queried and the number of queried tokens present in the cache
(i.e. hits).

However, the metric of interest is the hit rate - i.e. the number of
hits per query.

In the case of logging, we expect the user is best served by
calculating the hit rate over a fixed number of the most recent
queries (the interval is fixed to 1k most recent queries for now).

In the case of Prometheus though, we should take advantage of the
time-series nature of Prometheus and allow the user to calculate the
hit rate over an interval of their choosing. For example, a PromQL
query to calculate the hit interval of the past 5 minutes:

```text
rate(cache_query_hit[5m]) / rate(cache_query_total[5m])
```

To achieve this, we should record the queries and hits as counters in
Prometheus, rather than recording the hit rate as a gauge.

## Deprecated Metrics

### How To Deprecate

Deprecating metrics shouldn't be taken lightly. Users may not notice a
metric has been deprecated, and may be quite inconvenienced when it is
suddenly (from their perspective) when it is removed, even if there is
an equivalent metric for them to use.

As an example, see how `vllm:avg_prompt_throughput_toks_per_s` was
[deprecated](https://github.com/vllm-project/vllm/pull/2764) (with a comment in the code),
[removed](https://github.com/vllm-project/vllm/pull/12383), and then [noticed by a user](https://github.com/vllm-project/vllm/issues/13218).

In general:

1. We should be cautious about deprecating metrics, especially since
   it can be hard to predict the user impact.
2. We should include a prominent deprecation notice in the help string
   that is included in the `/metrics' output.
3. We should list deprecated metrics in user-facing documentation and
   release notes.
4. We should consider hiding deprecated metrics behind a CLI argument
   in order to give administrators
   [an escape hatch](https://kubernetes.io/docs/concepts/cluster-administration/system-metrics/#show-hidden-metrics)
   for some time before deleting them.

See the [deprecation policy](../contributing/deprecation_policy.md) for
the project-wide deprecation policy.

### Unimplemented - `vllm:tokens_total`

Added by <https://github.com/vllm-project/vllm/pull/4464>, but apparently never implemented. This can just be
removed.

### Duplicated - Queue Time

The `vllm:time_in_queue_requests` Histogram metric was added by
<https://github.com/vllm-project/vllm/pull/9659> and its calculation is:

```python
    self.metrics.first_scheduled_time = now
    self.metrics.time_in_queue = now - self.metrics.arrival_time
```

Two weeks later, <https://github.com/vllm-project/vllm/pull/4464> added `vllm:request_queue_time_seconds` leaving
us with:

```python
if seq_group.is_finished():
    if (seq_group.metrics.first_scheduled_time is not None and
            seq_group.metrics.first_token_time is not None):
        time_queue_requests.append(
            seq_group.metrics.first_scheduled_time -
            seq_group.metrics.arrival_time)
    ...
    if seq_group.metrics.time_in_queue is not None:
        time_in_queue_requests.append(
            seq_group.metrics.time_in_queue)
```

This seems duplicative, and one of them should be removed. The latter
is used by the Grafana dashboard, so we should deprecate or remove the
former.

### Prefix Cache Hit Rate

See above - we now expose 'queries' and 'hits' counters rather than a
'hit rate' gauge.

### KV Cache Offloading

Two legacy metrics relate to a "swapped" preemption mode that is no
longer relevant in v1:

- `vllm:num_requests_swapped`
- `vllm:cpu_cache_usage_perc`

In this mode, when a request was preempted (e.g. to make room in KV
cache to complete other requests), kv cache blocks were swapped out to
CPU memory. The `--swap-space` flag has been removed as this feature
is no longer used in V1.

Historically, [vLLM has long supported beam search](https://github.com/vllm-project/vllm/issues/6226). The
SequenceGroup encapsulated the idea of N Sequences which
all shared the same prompt kv blocks. This enabled KV cache block
sharing between requests, and copy-on-write to do branching. CPU
swapping was intended for these beam search like cases.

Later, the concept of prefix caching was introduced, which allowed KV
cache blocks to be shared implicitly. This proved to be a better
option than CPU swapping since blocks can be evicted slowly on demand
and the part of the prompt that was evicted can be recomputed.

SequenceGroup was removed in V1, although a replacement will be
required for "parallel sampling" (`n>1`).
[Beam search was moved out of the core](https://github.com/vllm-project/vllm/issues/8306). There was a
lot of complex code for a very uncommon feature.

In V1, with prefix caching being better (zero over head) and therefore
on by default, the preemption and recompute strategy should work
better.

## Future Work

### Parallel Sampling

Some legacy metrics are only relevant in the context of "parallel
sampling". This is where the `n` parameter in a request is used to
request multiple completions from the same prompt.

As part of adding parallel sampling support in <https://github.com/vllm-project/vllm/pull/10980>, we should
also add these metrics.

- `vllm:request_params_n` (Histogram)

  Observes the value of the 'n' parameter of every finished request.

- `vllm:request_max_num_generation_tokens` (Histogram)

  Observes the maximum output length of all sequences in every finished
  sequence group. In the absence of parallel sampling, this is
  equivalent to `vllm:request_generation_tokens`.

### Speculative Decoding

Some legacy metrics are specific to "speculative decoding". This is where
we generate candidate tokens using a faster, approximate method or
model and then validate those tokens with the larger model.

- `vllm:spec_decode_draft_acceptance_rate` (Gauge)
- `vllm:spec_decode_efficiency` (Gauge)
- `vllm:spec_decode_num_accepted_tokens` (Counter)
- `vllm:spec_decode_num_draft_tokens` (Counter)
- `vllm:spec_decode_num_emitted_tokens` (Counter)

There is a PR under review (<https://github.com/vllm-project/vllm/pull/12193>) to add "prompt lookup (ngram)"
speculative decoding to v1. Other techniques will follow. We should
revisit these metrics in this context.

!!! note
    We should probably expose acceptance rate as separate accepted
    and draft counters, like we do for prefix caching hit rate. Efficiency
    likely also needs similar treatment.

### Autoscaling and Load-balancing

A common use case for our metrics is to support automated scaling of
vLLM instances.

For related discussion from the
[Kubernetes Serving Working Group](https://github.com/kubernetes/community/tree/master/wg-serving),
see:

- [Standardizing Large Model Server Metrics in Kubernetes](https://docs.google.com/document/d/1SpSp1E6moa4HSrJnS4x3NpLuj88sMXr2tbofKlzTZpk)
- [Benchmarking LLM Workloads for Performance Evaluation and Autoscaling in Kubernetes](https://docs.google.com/document/d/1k4Q4X14hW4vftElIuYGDu5KDe2LtV1XammoG-Xi3bbQ)
- [Inference Perf](https://github.com/kubernetes-sigs/wg-serving/tree/main/proposals/013-inference-perf)
- <https://github.com/vllm-project/vllm/issues/5041> and <https://github.com/vllm-project/vllm/pull/12726>.

This is a non-trivial topic. Consider this comment from Rob:

> I think this metric should focus on trying to estimate what the max
> concurrency that will cause the average request length > queries per
> second ... since this is really what will "saturate" the server.

A clear goal is that we should expose the metrics required to detect
this saturation point, so administrators can implement auto-scaling
rules based on those. However, in order to do so, we need to have a
clear view on how an administrator (and automated monitoring system)
should judge an instance as approaching saturation:

> To identify, what is the saturation point for model server compute
> (the inflection point where we cannot get more throughput with a
> higher request rate, but start to incur additional latency) so we
> can autoscale effectively?

### Metric Naming

Our approach to naming metrics probably deserves to be revisited:

1. The use of colons in metric names seems contrary to
   ["colons are reserved for user defined recording rules"](https://prometheus.io/docs/concepts/data_model/#metric-names-and-labels).
2. Most of our metrics follow the convention of ending with units, but
   not all do.
3. Some of our metric names end with `_total`:

    If there is a suffix of `_total` on the metric name, it will be removed. When
    exposing the time series for counter, a `_total` suffix will be added. This is
    for compatibility between OpenMetrics and the Prometheus text format, as OpenMetrics
    requires the `_total` suffix.

### Adding More Metrics

There is no shortage of ideas for new metrics:

- Examples from other projects like
  [TGI](https://github.com/IBM/text-generation-inference?tab=readme-ov-file#metrics)
- Proposals arising from specific use cases, like the Kubernetes
  auto-scaling topic above
- Proposals that might arise out of standardisation efforts like
  [OpenTelemetry Semantic Conventions for Gen AI](https://github.com/open-telemetry/semantic-conventions/tree/main/docs/gen-ai).

We should be cautious in our approach to adding new metrics. While
metrics are often relatively straightforward to add:

1. They can be difficult to remove - see the section on deprecation
   above.
2. They can have a meaningful performance impact when enabled. And
   metrics are usually of very limited use unless they can be enabled
   by default and in production.
3. They have an impact on development and maintenance of the
   project. Every metric added over time has made this effort more
   time-consuming, and perhaps not all metrics justify this ongoing
   investment in their maintenance.

## Tracing - OpenTelemetry

Metrics provide an aggregated view over time of the system's
performance and health. Tracing, on the other hand, tracks individual
requests as they move through different services and components. Both
fall under the more general heading of "Observability".

vLLM has support for OpenTelemetry tracing:

- Added by <https://github.com/vllm-project/vllm/pull/4687> and reinstated by <https://github.com/vllm-project/vllm/pull/20372>
- Configured with `--otlp-traces-endpoint` and `--collect-detailed-traces`
- [OpenTelemetry blog post](https://opentelemetry.io/blog/2024/llm-observability/)
- [User-facing docs](../../examples/observability/opentelemetry/README.md)
- [Blog post](https://medium.com/@ronen.schaffer/follow-the-trail-supercharging-vllm-with-opentelemetry-distributed-tracing-aa655229b46f)
- [IBM product docs](https://www.ibm.com/docs/en/instana-observability/current?topic=mgaa-monitoring-large-language-models-llms-vllm-public-preview)

OpenTelemetry has a
[Gen AI Working Group](https://github.com/open-telemetry/community/blob/main/projects/gen-ai.md).

Since metrics is a big enough topic on its own, we consider the topic
of tracing to be quite separate from metrics.

### OpenTelemetry Model Forward vs Execute Time

The current implementation exposes the following two metrics:

- `vllm:model_forward_time_milliseconds` (Histogram) - The time spent
  in the model forward pass when this request was in the batch.
- `vllm:model_execute_time_milliseconds` (Histogram) - The time spent
  in the model execute function. This will include model forward,
  block/sync across workers, cpu-gpu sync time and sampling time.

These metrics are only enabled when OpenTelemetry tracing is enabled
and if `--collect-detailed-traces=all/model/worker` is used. The
documentation for this option states:

> collect detailed traces for the specified modules. This involves
> use of possibly costly and or blocking operations and hence might
> have a performance impact.

The metrics were added by <https://github.com/vllm-project/vllm/pull/7089> and show up in an OpenTelemetry trace
as:

```text
-> gen_ai.latency.time_in_scheduler: Double(0.017550230026245117)
-> gen_ai.latency.time_in_model_forward: Double(3.151565277099609)
-> gen_ai.latency.time_in_model_execute: Double(3.6468167304992676)
```

We already have `inference_time` and `decode_time` metrics, so the
question is whether there are sufficiently common use cases for the
higher-resolution timings to justify the overhead.

Since we are going to treat the question of OpenTelemetry support
separately, we will include these particular metrics under that topic.
