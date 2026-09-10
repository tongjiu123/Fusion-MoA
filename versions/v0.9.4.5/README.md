# Fusion-MOA v0.9.4.5：共享协作证据候选

本目录是 **完整 v0.9.4.4 基线＋共享协作证据增量** 的独立源码包，不导入其他版本目录。当前为本地整理后的公开候选；未部署、未重启服务、未启动新版测评，尚无 v0.9.4.5 的提分或生产可靠性证明。

## 本版增量

- Analyst、Integrator 和 D12 Deliberation 使用同一份有界 Evidence Snapshot。原始／最新 user 更新保留在证据中，引用沿用原消息编号；被截断的条目标记 `truncated=true`。
- 完整 Analyst 意见继续传递：hypothesis、abstain、claim、reason_code、支持／反例引用、probe 与预期／证伪观察，不只传一个状态摘要。
- Integrator 区分事实、假设、分歧与弃权，不把单个假设当作多数共识，也不自行变成第四位 Analyst；输出仍为内部 `{"brief":"..."}`。
- D12 可以部分采纳、拒绝或继续原路径，直接正常返回 text／tool_calls，不要求单独 ACCEPT／REJECT 声明。共享协作证据不替代 D12 的完整用户上下文：原 system／user／compact 上下文、最新 user 角色和外部请求的 tools 保留。
- Soft Stall 是周期协作时点，不等于已经失败、必须恢复或必须换方向；Hard Stall 和显式 Panel 保留各自语境。当前更新不会自动撤销所有旧约束。
- 临时证据与意见沿私有调用返回值传递，在公开 Panel 事件和 Trace 前移除，不增加持久存储、指标、请求或额外生成。

全部或部分 abstain 仍走原协作流程，没有增加共识门或硬拦截。16 轮周期、Hard／Result Cycle、Episode、Recovery、Compact、fail-open、D12 唯一执行与输出权均沿用基线；预算、采样与模型槽不因版本整理而改变。

## 独立包与版本兼容

```text
versions/v0.9.4.5/
├── fusion_terminal_server_v0945.py
├── v0945_next/                   # 本版运行代码与既有 tests
│   └── schemas/hypothesis_v3.json
├── requirements.txt              # 继承的固定依赖
├── run_gateway.sh
├── deploy_metax_gateway.sh
├── amd_mt2/                      # MT2 模板与共享模型资源说明
├── amd_mt7/                      # 继承的容量资格工具，不是新版实测
└── start_metax_d12_*.sh           # 沐曦专用共享模型启动器
```

入口与项目内导入使用 `v0945_next`；health、版本 header、boundaries 和公共身份为 `v0.9.4.5`。Trace 的结构没有变，`schema_version` 仍是 `fusion-moa-v0944-trace-v1`：这是格式兼容标签，不代表加载旧代码。

支持以下模型别名，兼容旧客户端配置：

| 版本前缀 | 自动 MoA | D12 solo |
|---|---|---|
| v0945 | `fusion-moa-v0945-general` | `fusion-moa-v0945-solo-d12` |
| v0944 | `fusion-moa-v0944-general` | `fusion-moa-v0944-solo-d12` |
| v0943 | `fusion-moa-v0943-general` | `fusion-moa-v0943-solo-d12` |
| v0942 | `fusion-moa-v0942-general` | `fusion-moa-v0942-solo-d12` |

版本环境变量由入口按 `V0945_*` → `V0944_*` → 既有更早版本 fallback 解析。`FUSION_*`、`D12_BASE_URL` 等非版本化部署变量沿用原名。

## 启动模板及不变默认值

这些是将来获准部署时使用的模板，本轮没有执行。

| 项目 | 通用入口／run 脚本 | MT2 service 显式值 |
|---|---:|---:|
| Gateway 端口 | 19025 | 19018 |
| Executor timeout（秒） | 3300 | 6900 |
| Executor max tokens | 8192 | 16384 |
| Output recovery max tokens | 512 | 512 |
| Analyst deadline（秒） | 120 | 120 |
| Analyst／Integrator max tokens | 768 | 768 |

两者均默认 D16、D9、D13 三个 Analyst，D13 兼 Integrator。`run_gateway.sh` 从本目录启动新入口，预算与版本变量交入口解析，避免 shell 默认值掩盖旧版本环境设置。

`deploy_metax_gateway.sh` 只用于对应的沐曦环境，保留 **19026** 资格端口；代码挂载和日志／容器名使用 v0945。它会创建 Gateway 容器，但不会启动第二套模型。不能把该脚本或 `/dev/mxcd` 配方用于 AMD；任何启动前都须确认端口与服务所有权。

本仓库只提供源码与部署模板，**不附带模型权重、定制 vLLM 镜像或服务器 Python 环境，也不保证作者使用的镜像是公开可拉取的镜像**。部署者须提供适配 GPU、驱动、模型架构与这些启动参数的兼容镜像：AMD 模型脚本使用必填 `AMD_VLLM_IMAGE`，沐曦模型脚本使用必填 `METAX_VLLM_IMAGE`，沐曦 Gateway 脚本使用必填 `METAX_GATEWAY_IMAGE`（须含 `/opt/conda/bin/python` 及本包依赖）。模型脚本还要求对应的 vLLM 命令入口；镜像不兼容时不能只改标签后假定可运行。

## MT2 共享资源与未来原位切换

详见 [amd_mt2/README.md](amd_mt2/README.md)。公开模板使用以下示例布局，部署前应按目标机器调整：

- `/opt/fusion-moa/versions/v0.9.4.5/fusion_terminal_server_v0945.py`
- 日志目录 `/var/log/fusion-moa/v0.9.4.5/gateway`
- 新 run_id `amd-mt2-v0945-shared-evidence-candidate`

以下是可跨 Gateway 版本共享的部署资源，**不是旧代码依赖**：

- 部署者准备的 Python 环境，示例 `/opt/fusion-moa/venv/`；
- 模型权重根目录，示例 `/opt/fusion-moa/models/`，模型脚本可通过 `FUSION_MODEL_ROOT` 覆盖；
- 继承模板中的模型容器名 `fusion-moa-v0944-mt2-d12-qual`、`fusion-moa-v0944-mt2-d9-qual`；这些名字不表示容器或权重随本仓库提供。

AMD 脚本还支持 `D12_MODEL_PATH`、`D9_MODEL_PATH` 指定完整权重路径；沐曦日志根目录可用 `FUSION_GATEWAY_RUN_ROOT` 覆盖，默认 `/var/log/fusion-moa/v0.9.4.5/metax-qual`。systemd 模板使用显式示例路径，须修改模板或使用部署者自己的 drop-in；它不会自动展开 shell 的路径变量。

版本整理不运行 `start_d12_d9.sh`，不重新下载权重、不加载第二套模型，也不轮换现有模型容器。

旧 Harbor 服务的 Requires 仍指向稳定的 `fusion-moa-v0944-mt2-gateway.service`。未来经授权原位切换时，可以把新版 service 模板内容安装到该稳定 unit 文件中，保留 unit 名及依赖关系；**不能另启一个同占 19018 的 v0945 unit**。切换前需保留旧 unit 配置并确认活动任务已妥善处理，再核对新代码、共享 venv、日志权限和回滚目标；今天不做这些服务器操作。

## 基线、证据与待验证项

完整旧版 v0.9.4.4 基线保存在作者本地，**不在本仓库的相邻目录中**；此处不提供指向它的链接，也不以它作为运行依赖。本版本不从 history 加载代码；部署者如需回滚，应自行保留经验证的旧发布包与服务配置，而不能只依赖四个文件的局部改前副本。

本地静态与已有真实数据的函数级检查，只能支持源码／数据传递层面的结论。此前沐曦、MT2 v0.9.4.4 的生产记录、模型 readiness 与 Benchmark 分数均属于历史基线，不能归作 v0.9.4.5 实测。新版的目标机运行、模型生成质量、端到端回退和测评收益仍待未来单独授权验证。
