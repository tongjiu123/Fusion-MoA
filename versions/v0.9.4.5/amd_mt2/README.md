# AMD MT2：v0.9.4.5 候选模板与共享模型资源

本目录是公开候选的部署示例，**没有部署、重启或启动新版测评，也没有新版提分证据**。service 是未来切换模板；下面的模型参数与 readiness 文件继承自 2026-09-08 的 v0.9.4.4 恢复记录，不是 v0.9.4.5 资格结果。私有机器目录已替换成公开示例，部署者须提供模型、兼容镜像及 Python 环境。

## Gateway 模板

`fusion-moa-v0945-mt2-gateway.service` 使用：

| 项目 | 值 |
|---|---|
| 代码 WorkingDirectory | `/opt/fusion-moa/versions/v0.9.4.5` |
| 入口 | `fusion_terminal_server_v0945.py` |
| 共享 Python 环境 | `/opt/fusion-moa/venv/bin/python` |
| Gateway 端口 | 19018，不改变 |
| run_id | `amd-mt2-v0945-shared-evidence-candidate` |
| 日志根目录 | `/var/log/fusion-moa/v0.9.4.5/gateway` |
| D12 timeout / max tokens | 6900 秒 / 16384 |
| Output recovery max tokens | 512 |
| Analyst deadline / max tokens | 120 秒 / 768 |

模板显式使用 `V0945_*`，入口仍兼容 `V0944_*` 及既有更早变量。D12／D16／D9／D13 的本机上游仍分别是 8001／8002／8004／8006；D13 兼任 Integrator。Trace schema 标签保持 `fusion-moa-v0944-trace-v1`，只是未变的数据格式名。

## 不随代码版本轮换的模型资源

| 槽 | served model | GPU / TP | 上下文 / max_num_seqs | Prefix / MTP |
|---|---|---|---|---|
| D12 | `Qwen/Qwen3.8-27B`，BF16 | 0–3 / 4 | 262144 / 2 | 开 / 5 |
| D9 | `TeichAI/Qwen3.6-27B-Fable-5-v1`，现有 W4A16 权重 | 5 / 1 | 131072 / 2 | 开 / 未启用 |

`start_d12_d9.sh` 保留原参数与容器名，权重使用可配置公开路径：

- D12 容器：`fusion-moa-v0944-mt2-d12-qual`；默认权重 `/opt/fusion-moa/models/Qwen3.8-27B`，可用 `D12_MODEL_PATH` 指定完整路径。
- D9 容器：`fusion-moa-v0944-mt2-d9-qual`；默认权重 `/opt/fusion-moa/models/D9-Fable5-W4A16/modelscope-master-20260902`，可用 `D9_MODEL_PATH` 指定完整路径。
- 两个默认权重路径的公共根目录可用 `FUSION_MODEL_ROOT` 覆盖，默认 `/opt/fusion-moa/models`。
- 镜像由必填 `AMD_VLLM_IMAGE` 指定。部署者必须提供适配 AMD GPU、驱动、模型架构、解析器及 MTP 参数的定制 vLLM 镜像，并提供 `vllm` 入口；这里没有公开镜像可用性的保证，镜像和权重均不随包附带。
- 原脚本的 host network／IPC、64g shm、`/dev/kfd`、`/dev/dri`、video group、`HIP_VISIBLE_DEVICES`、只读模型挂载均未改。
- 原启动参数仍为 BF16 计算、auto KV、显存比例 0.95、batched tokens 4096、Prefix／chunked prefill、eager、text-only；D12 另有 long-prefill threshold 2048、qwen3_xml 工具解析、qwen3 推理解析和 MTP5。

容器名中的 v0944 是继承的共享模型资源标识，不是新 Gateway 导入旧运行代码，也不表示仓库已创建这些容器。新版整理不执行模型启动脚本、不下载或复制权重、不启第二套模型。

`start_d12_d9.sh` 仅供未来确认容器确实缺失且 GPU 空闲后的单槽恢复使用；它不删除／停止其他容器。旧原容器完整 inspect 未保存，继承脚本来自当时已核实主要参数的重建记录，不能称为误删前容器所有环境变量的逐字节复原。`readiness-d12.json`、`readiness-d9.json` 仅是继承的短检查请求；本候选未重跑它们，未随包包含旧响应，也未做新版 256K 双并发资格验证。

## 未来原位切换：保留稳定 unit 名

旧 Harbor 服务依赖 `fusion-moa-v0944-mt2-gateway.service`。将来获准切换时，可将本目录新版模板的内容替换到服务器现有稳定 unit 文件：

`/etc/systemd/system/fusion-moa-v0944-mt2-gateway.service`

这样保留旧 unit 名与 Harbor 的 Requires 关系，而代码、日志及 run_id 指向 v0945。**不要直接另启动 `fusion-moa-v0945-mt2-gateway.service` 与旧服务争用 19018。**

未来切换前需要备份稳定 unit、核实活动测评状态及回滚目标；部署新代码后核对示例路径、共享 venv、日志目录权限和 health／header 版本，再按授权处理服务重载与重启。systemd 不会展开 shell 路径变量，应修改模板或使用自己的 drop-in。作者的完整旧 v0.9.4.4 基线保存在作者本地、未随此仓库发布；部署者须自行保存旧发布包及原 unit 配置作为回滚目标，不需要轮换模型权重或同时保留两个同端口 Gateway。本轮没有执行任何这些服务器操作。
