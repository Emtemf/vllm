# Metrics

vLLM exposes a rich set of metrics to support observability and capacity planning for the V1 engine.

## Objectives

- Provide comprehensive coverage of engine and request level metrics to aid production monitoring.
- Prioritize Prometheus integrations, as this is what we expect to be used in production environments.
- Offer logging support (i.e. printing metrics to the info log) for ad-hoc testing, debugging, development, and exploratory use cases.

## P/D diagnostic evidence

The paired P/D diagnostic implementation emits `[PD_DIAG]` JSON observations
through the INFO logger. Original model, scheduler, and cache behavior and
native metric formulas are preserved. Additional IDs, counters, spans, and
logging are diagnostic observations with overhead, not zero-cost instrumentation
or complete TTFT accounting. This is separate from Prometheus aggregation and
does not enable device profiling or add device synchronization.

Each observation includes process/clock identity and a unique record ID.
Execution IDs link scheduler batches to workers; request and task links retain
shared work rather than attributing a whole batch to every member. Shared batch
and parallel TP ranges are not per-request additive latency. Dense DP replicas
use their preserved `data_parallel_index`, including after the local parallel
rank has been reset for an independent engine.

API observation supports root-path mounts, token-ID-only generated responses,
and delayed closure of resumable input. Request summaries preserve the native
timing formulas. `first_token_observed=false` marks a terminal output with no
generated token: its native first-output timestamp is not proof of a first
token. Request/report/Grafana display TTFT must be null in that case, even when
the original native TTFT field is nonzero; retain the original field as evidence.

To inspect collected observations with the container's existing Python (no uv
or vLLM import is required by these standalone converters):

```bash
cd /vllm-workspace/vllm
PD_PYTHON=/usr/local/python3.12.16/bin/python
mkdir -p /vllm-workspace/pd-metrics-output
"$PD_PYTHON" tools/pd_metrics_report.py prefill.log decode.log -o /vllm-workspace/pd-metrics-output/report.json &&
"$PD_PYTHON" tools/pd_metrics_report.py prefill.log decode.log -o /vllm-workspace/pd-metrics-output/report.html --html &&
"$PD_PYTHON" tools/pd_request_json.py prefill.log decode.log -o /vllm-workspace/pd-metrics-output/requests.json &&
"$PD_PYTHON" tools/pd_grafana_json.py /vllm-workspace/pd-metrics-output/report.json -o /vllm-workspace/pd-metrics-output/rows.json
```

For Grafana rows only, convert raw logs in one command without intermediate files:

```bash
"$PD_PYTHON" tools/pd_grafana_json.py --logs prefill.log decode.log -o rows.json
```

Malformed diagnostic records or logs without diagnostic records cause exit code 2
before writing output. The existing report-JSON input remains supported.

The HTML report starts with five non-additive mainline groups:

1. API input: req_resolve, render, tokenize, input assembly, registration, send.
2. EngineCore receipt, initialization/block hash, input queue residence,
   scheduler queue/schedule, and execute submission.
3. Worker input_prepare, forward, logits, sample, draft, output_build, and KV
   load/save entry points.
4. EngineCore result collection, scheduler update, output queue, and send.
5. API output receipt and the official endpoint.

These are process-owned navigation groups, not five latency totals or dozens
of additive TTFT components. Core return traffic is not placed inside API
output processing. A single step × TP table shows shared schedule details,
Worker stages, and horizontal `wait_for_save` columns for Pool queue/key/lookup/
prepare/sync work and Mooncake put start/submit/wait/finalize. Save tasks must
have one explicit execution and a unique matching Worker lifecycle/rank before
their values enter that row. Neither tasks nor native phases are added to the
connector boundary.
Missing stage timing is distinct from a task with no work.
Request-level detokenizer is supplemental work outside the official endpoint,
not a sixth stage or part of stage five; it is never copied into TP rows.

`engine_input_received` marks successful input decoding;
`engine_request_init` spans the original preprocessing call. Input queue
positions pair by complete host/PID/run identity and `input_message_id`.
Older evidence without a message ID can pair only a unique request-ID pair;
repeated streaming/resume handoffs remain ambiguous rather than being zipped
or matched by proximity. Output queue positions pair by the same clock domain
and `output_batch_id`. Queue residence uses explicit `queue_enter_ns` and
`queue_exit_ns`, not log-emission timestamps or time blocked on an empty get.
Missing, inverted, incomplete, or ambiguous pairs remain unknown, not zero.

`engine_output_send` measures the shared batch's encoding and host send call;
it does not establish transport completion. The optional `output_batch_id`
links Core output positions/send to `api_output_received` and
`api_output_endpoint` as identity evidence only. Old outputs lacking the ID
remain usable without an invented batch or latency. `api_output_received`
preserves the handler's actual local acquisition boundary in `received_ns`,
separate from emission time. API endpoint
`iteration_timestamp` reuses the existing `IterationStats` wall timestamp;
it is not subtracted from monotonic clocks or used to redefine native TTFT.
The emitter's `timestamp_ns` is process-local monotonic evidence, not a
cross-host timeline.

`engine_batch_enqueued` and `engine_batch_dequeued` retain execution-position
evidence but are not timers. Even historical records with tiny pass-span
durations are displayed as instantaneous positions, not latency. Business
aliases retain their real source events: req_build is the single `input_process`
assembly timer, req_send is `engine_submit`, and req_resolve is `api_resolve`.
Registration and ID-link positions use `frontend_request_registered` and
`request_id_link`, not invented `req_register` or `events` timers. req_resolve
measures ASGI ingress to the typed handler, usually outside the official request
start; it is not labeled model/business validation. No renamed-timer
compatibility layer is introduced.

Each request has one request-level row, followed by its compute step rows.
The TP column starts inside the Worker header. Request/step identity and non-TP
cells merge across each step's Worker rows. Explicit `no_forward` rounds without
a measured forward remain in collapsed maintenance evidence; no duration cutoff
is used, and short measured forward calls remain visible. Assembly and send are
timers; instantaneous ID matching/registration stay in raw evidence, not columns.

The primary table omits preprocessing initialization, input-queue residence,
output send timing, and Mamba-state copy. These observations remain available
in raw evidence; hiding a column does not remove instrumentation. Initialization
and output send are host-call timers, and input-queue residence is an interval,
not an instantaneous event. The per-step `block hash` column measures actual
hash computation with explicit execution identity, not queue time or a request
cumulative value. It is separate from the schedule header: hashes may be computed
during output update. Initial prompt hashes normally precede the first schedule
and have no execution identity. Missing per-step ranges display as no step record,
not measured zero or proof that hash computation did not run.

The second-level `schedule` header spans `local lookup / external lookup /
alloc / other` alongside its batch total. In multi-request batches, local lookup and
allocation cells list request-specific `scheduler_kv_range` values, with the
batch aggregate in the tooltip. Only hash ranges with an original execution ID
enter the step column; request cumulative hash stays in raw evidence.
These detail columns are not promised to form an additive decomposition.

`wait_for_save` is shown on the Worker TP row as the connector boundary. It is
not an additive breakdown of asynchronous Pool work: after save submission, a
background task continues in its own process/thread lifecycle. The same TP row
shows enqueue, queue wait, build_key, lookup, prepare, sync, Python put, native
phase1–4, keys/MiB, status, and task ID under nested headers. Submission tasks
fill enqueue only, without missing placeholders in backend columns. Native
split scopes merge by their unique `pool_native_call` boundary. Within each
task, phases accumulate over distinct calls, with call count and per-call values
in tooltips; repeated same-phase evidence stays unknown. Totals with different
scope are not added. `no_task` means no save work, not failed request association.
Unbound or cross-step tasks remain in collapsed evidence. Load entry timing does
not prove transfer completion, and missing independent transfer-wait timings
are not inferred. The ordinary-step control flow is displayed above the table:
EngineCore schedules and dispatches execute, conditionally dispatches sampling,
then updates from the result. Batch-queue execution may overlap steps; column
order is not a universal timeline. Worker execute/sample interval coverage
remains separate from request TTFT and is not a request residual.

The request and step columns stay fixed during horizontal scrolling. The request
selector filters rows; detailed provenance and raw JSON are collapsed last.
Prefill-only logs are supported: omit `decode.log` to inspect P without D.
Explicit lifecycle and transport links establish request association. Ambiguous
reused IDs remain visible candidates and do not contribute durations to a request.
Scheduler and mixed-task work stays shared; TP rows are never summed.
Missing, zero, skipped, and fused observations are displayed distinctly.
Derived dashboard fields do not change the source schema-v1 observations.

`scheduler_external_lookup` measures the existing connector call, including
attempts that cannot yet be scheduled. Each schedule summary retains its
`external_lookup_ns` and excludes it from `other_ns`. Request hash ranges remain
request-level unless an explicit execution identity is present; truncated ranges
are marked incomplete. Engine `update_from_output`, released-request save fences,
and P/D metadata/preparation/reformat boundaries are displayed separately.

The report also reads `[BatchPutTiming]` and `[BatchUpsertTiming]` text records.
For each whole-key native put, the paired Ascend code emits `pool_native_call`
with its task identity, native TID, wall-clock start/end, source UTC offset, and
submitted keys/bytes. The report requires the same input source and thread, one
matching wall-clock call interval, and agreement with any native keys/bytes.
Glog timestamp precision is respected; a timestamp bucket overlapping multiple
calls remains ambiguous. The source offset is used, not the reporting machine's
timezone. Matching phases appear in the original save task's put cells.

Thread ownership alone cannot select a put or request. Logs lacking a usable
call boundary remain in collapsed pending evidence, not a detached main-table
native row or a guessed task. The `phase1_only`, `prefer_same_node_remaining`,
and `normal_full` scopes stay distinct; the remaining-path total excludes phase 1.

For raw-log inspection:

```bash
grep -E '"event":"(scheduler_external_lookup|engine_scheduler_update|request_hash_ranges|pool_save_fence)"' prefill.log
grep -E '"event":"pd_transfer_(meta_fetch|prepare|reformat|batch_finished|task_finished)"' decode.log
grep -E 'Batch(Put|Upsert)Timing|"native_tid":' prefill.log
```

The report does not add parallel TP intervals, subtract clocks from different
processes, or treat submitted bytes as measured wire traffic. Missing evidence
remains unknown. CPU regression checks do not establish NPU performance or
distributed serving correctness; validate the paired build on the target stack.

Request exports are available through `tools/pd_request_json.py` (raw logs to
request JSON) and `tools/pd_grafana_json.py` (raw logs or report JSON to dashboard rows).
Terminal `scheduler_queue_summary.decode_steps` counts decode scheduling
iterations, including MTP iterations; observed execution counts are separate.
`request_cache_lookup` reports effective HBM/connector hit boundaries, not
successful KV loading. Request block references measure referenced capacity,
not exclusive physical HBM allocation. P/D native batch timings and the
first-to-last-batch request-generation envelope have separate scopes;
submitted/succeeded/failed byte classifications are not measured wire bytes.
Missing or conflicting evidence remains null with an explicit state.

The report and request converters accept one or more original log paths, not
report JSON or shell glob strings. Expand globs in Bash and use each original
file once; do not also input a concatenated copy. The Grafana converter accepts
exactly one **report JSON**, not raw logs or request JSON. Output directories
must already exist. Report conversion writes parse errors to its JSON and exits
with status 2 if any record is invalid; request conversion rejects invalid
records before writing output. Stop the pipeline on either failure and inspect
the original file/line instead of serving a stale rows file. Request JSON is
one object for one request and an array for zero or multiple requests.

P and D official summaries stay separate, including when explicit transport
links associate them with one request. The report retains `native_summaries`;
request JSON retains `request_summary_by_side.p/d`; Grafana rows retain
`native_summary_by_side.P/D` and `p_ttft_ms`, `d_ttft_ms`, `p_e2e_ms`,
`d_e2e_ms`, and side-specific token fields. Generic single-summary fields remain
null when several summaries are present. Do not pick the first summary or add
P and D TTFT/E2E into client latency. Missing role evidence remains unknown;
the CLI does not accept a side manifest or infer side from filenames. Keep a
collection manifest for the operator's deployment-side provenance.

### Linux log capture

On **each** container, retain the existing model, network and KV launch options.
Set `PD_SIDE=P` on P and `PD_SIDE=D` on D. Replace only `PD_START_SCRIPT` with
the existing serving startup script; this example runs that script in the
foreground with its child API/Core/Worker output captured together:

```bash
set -o pipefail
PD_SIDE=P
PD_START_SCRIPT=/absolute/path/to/your/existing-start.sh
PD_CAPTURE=/vllm-workspace/pd-metrics-logs/$PD_SIDE/$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$PD_CAPTURE"
{
  printf 'side=%s\nhost=%s\nstart_utc=%s\n' "$PD_SIDE" "$(hostname)" "$(date -u +%FT%TZ)"
  printf 'source_timezone=%s\n' "$(date +%z)"
  git -C /vllm-workspace/vllm rev-parse HEAD
  git -C /vllm-workspace/vllm-ascend rev-parse HEAD
  /usr/local/python3.12.16/bin/python -m pip show mooncake-transfer-engine
} > "$PD_CAPTURE/manifest.txt"
export VLLM_LOGGING_LEVEL=INFO PYTHONUNBUFFERED=1 GLOG_logtostderr=1
bash "$PD_START_SCRIPT" 2>&1 | tee "$PD_CAPTURE/serving.log"
```

Start capture before startup and send the real request through the deployment's
existing P/D entrypoint from another terminal. Keep capture through request
termination and asynchronous save completion. The emitter uses INFO logging;
there is no extra `PD_DIAG` enable flag. A custom logging configuration must
retain that logger. `GLOG_logtostderr=1` requests native glog on stderr; if the
launch script overrides logging, collect its actual native files too. A script
that daemonizes or starts workers in other containers requires those workers'
own complete log streams; this pipeline alone cannot capture detached output.

For an already-running Docker service, capture from its Linux host using its
actual container name, separately for P and D:

```bash
PD_CONTAINER=your-existing-p-container
PD_CAPTURE=./pd-metrics-logs/P
mkdir -p "$PD_CAPTURE"
docker logs --timestamps --since 30m --follow "$PD_CONTAINER" > "$PD_CAPTURE/serving.log" 2>&1
```

Stop following only after the request and save work finish. Docker logs cover
container stdout/stderr, not arbitrary files. Ascend's default file-handler
directory is `~/ascend/log/vllm_ascend`, configurable by `ascend_log_path`;
preserve the original and rotated files if that handler is active. Do not feed
both file and stdout copies of the same native timing records. Preserve full
lines, original source files, timezone and host/PID/TID/run identities. In
particular, `pool_native_call` and matching `[BatchPutTiming]` /
`[BatchUpsertTiming]` must remain in the same input source; filtering down to
`[PD_DIAG]` alone loses native details. Copy P and D logs to one analysis host
without overwriting their separate directories, then run the conversions above
with the actual collected paths. Shared batch durations are never divided by
member count; parallel TP and overlapping stages are never summed or subtracted
from another process/host clock.

### Grafana snapshot hand-off

`rows.json` contains `requests`, `observations`, `tasks`, `hbm`, `pd_transfers`
and `quality` arrays. It is a diagnostic snapshot, not Prometheus time-series
data, and the Grafana time picker does not filter it. The companion delivery
assets are outside this source repository: `ttft-patches/grafana_rows_api.py`,
`ttft-patches/grafana_pd_request_dashboard.json`, and
`ttft-patches/grafana-demo/README.md`. Stage the first two on Linux, for example
under `/vllm-workspace/pd-grafana`, before starting the read-only adapter:

```bash
/usr/local/python3.12.16/bin/python /vllm-workspace/pd-grafana/grafana_rows_api.py /vllm-workspace/pd-metrics-output/rows.json --host 127.0.0.1 --port 8765
```

The adapter loads once at startup; restart it after regenerating rows. In a
second terminal, `curl -fsS http://127.0.0.1:8765/requests` retrieves the request
keys. Use `curl -fsS --get --data-urlencode 'request_key=KEY_FROM_REQUESTS'
http://127.0.0.1:8765/observations` for one request; the other five endpoint
names match the arrays above. An empty array means no rows for that scope,
not measured zero. Install Grafana JSON API plugin
`marcusolsson-json-datasource` (demo: 1.4.4 on Grafana OSS 12.0.2), create that
datasource with the adapter base URL, import the delivered dashboard JSON, and
select it in **Rows datasource**, then choose **Request**. For container/remote
Grafana, configure an address reachable from the Grafana server; its loopback
is not the analysis host's loopback. See the companion delivery guide for Linux
Docker startup and exact import/query steps.

The supplied sample log, sample JSON and portable Grafana screenshots are
**synthetic demo data**, not Linux/NPU inference measurements. Keep the demo's
synthetic labels; use a separately named dashboard for collected real logs and
record their manifest. Null/missing evidence must remain null throughout
Grafana queries and transformations. A rendered dashboard does not establish
NPU execution or distributed inference correctness.

### Paired P/D container deployment

Use `tools/deploy_pd_metrics_d.sh` on both P and D. Copy it outside the two
source roots, for example `/vllm-workspace/deploy_pd_metrics_d.sh`, before
deployment so source checkout cannot remove the rollback entrypoint. Stop
serving and transfer-client workers through the normal supervisor first;
leave the independent shared `mooncake_master` running.

Use the final verified and published 40-character core commit supplied with the
release, for both the downloaded script and the explicit source override:

```bash
PD_RELEASE_CORE=FINAL_VERIFIED_PUBLISHED_40_CHARACTER_CORE_SHA
curl -fL "https://raw.githubusercontent.com/Emtemf/vllm/$PD_RELEASE_CORE/tools/deploy_pd_metrics_d.sh" -o /vllm-workspace/deploy_pd_metrics_d.sh &&
PD_DEPLOY_CORE_PIN="$PD_RELEASE_CORE" \
PD_DEPLOY_ASCEND_PIN=afdbf1e3e3057b32cfe22075244e552a25113330 \
bash /vllm-workspace/deploy_pd_metrics_d.sh --apply
```

The default interpreter is `/usr/local/python3.12.16/bin/python`; Git and the
interpreter's existing pip/packaging are required, not uv. The existing editable
roots must be `/vllm-workspace/vllm` and `/vllm-workspace/vllm-ascend`, or supplied
through `--vllm-dir` and `--ascend-dir`. Existing compiled extensions and editable
registration are preserved; vLLM and Ascend are not rebuilt or reinstalled.

The script's committed default core pin is
`f8b9de31a9e3a1b8ad76dee78917534121be39d1`, and its imagebase Ascend pin is
`afdbf1e3e3057b32cfe22075244e552a25113330`. These identities alone do not prove
publication or final release validation. The final release command must use
`PD_DEPLOY_CORE_PIN` for the newer verified commit; do not pull a newer branch
and then apply an older default. It fetches missing history before checking
both approved baselines, refuses unrelated or dirty trees, saves source identity
and wheel copies, then switches to detached commits. The original branch tips
remain available for rollback.

Compatible `mooncake-transfer-engine==0.3.13` diagnostic wheels are discovered
under known Mooncake build directories and `/vllm-workspace` staging. Distinct
valid builds are ambiguous; identical copies are deduplicated. The candidate
must contain Ascend transport, store, master and `BatchPutTiming`. If it already
matches the installed payload, it also provides exact rollback. Otherwise the
script locates the exact original wheel or downloads the image's official
Huawei wheel over verified HTTPS and compares its payload with the installed
files before accepting it. Optional `--wheel` and `--backup-wheel` select explicit
files. If P's build directory is not mounted on D, copy the built `.whl` from
`/data/y00624827/Mooncake-wheelbase/mooncake-wheel/dist/` into
`/vllm-workspace/wheels/` on D. No compatible artifact means no cutover.

`--check` performs no Git fetch, checkout, installation or data-file writes;
it may validate an absent original wheel through an in-memory HTTPS download.
`--apply` installs only the saved Mooncake wheel through the selected
interpreter's `-m pip install --no-index --force-reinstall --no-deps`.
Keep the printed state directory and rollback command. Rollback restores the
recorded source branches/commits and exact prior wheel payload; changed source
or saved wheel state is refused. No mode stops or restarts services. Restart
each service with its existing launch command after successful deployment,
complete a real request, and collect full API/core/worker logs from both sides.
CPU checks do not establish Linux/NPU serving or model correctness.


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
