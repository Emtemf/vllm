# TTFT 离线诊断汇总工具用法(prelaunch)

## 开启诊断
- 环境变量 `VLLM_TTFT_DIAGNOSTICS=1`,默认关闭,关闭时零开销(装饰器直接透传原函数)。
- 日志输出到 `vllm.ttft_diagnostics` logger,格式 `[TTFT_DIAG] {json}`,JSON 含 `version, event, run_id, host, pid, mono_ns, wall_ns`。

## 运行
```bash
python tools/ttft_diagnostics.py report serve.log          # 汇总单个日志
python tools/ttft_diagnostics.py report a.log b.log        # 合并多个日志(按 mono_ns 排序)
python tools/ttft_diagnostics.py report serve.log --request req-123   # 单请求时间线
python tools/ttft_diagnostics.py preflight        # 预检环境,不初始化模型/Store
```

## 事件清单
| 阶段 | 事件 | 关键字段 |
|---|---|---|
| 前端 | frontend_render(_start) | frontend_id, route, boundary |
| 前端 | frontend_input | frontend_id, route, execution, reason |
| 调度 | step_plan | step_id, num_scheduled_tokens, num_new_reqs |
| 调度 | request_step | step_id, request_id, scheduled_tokens, frontend_id |
| 调度 | request_wait | request_id, frontend_id, state=queued |
| 调度 | request_core_end | request_id, frontend_id, status |
| EngineCore | core_execute / core_update | step_id, duration_ns |
| Worker | worker_execute / worker_sample / worker_output | step_id, runner, rank, request_ids |
| Store | store_save_check | request_ids, checked_keys, submitted_keys/bytes, status |
| Store | store_save_task / store_load_task | task_id, request_ids, rank |
| Store | store_put / store_get | trace_id, keys, bytes, backend, status |
| Store | request_match | request_id, local_tokens, external_tokens, result |

## 关联链
`frontend_id → EngineCoreRequest.ttft_frontend_id → Request.ttft_frontend_id → request_step/request_wait/request_core_end 事件`
`step_id → SchedulerOutput.ttft_step_id → ModelRunnerOutput.ttft_step_id → worker_* 事件`

## 解析规则
- 只解析 `[TTFT_DIAG] ` 前缀行;跨进程 `mono_ns` 不可比较,按 `run_id+pid` 分组展示。
- `duration_ns` 嵌套 span 不求和;missing 显示为 `-`,不补 0。
