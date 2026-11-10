# ThinkPRM-14B 4-bit 实际试跑记录（2026-11-10）

已完成官方权重下载、vLLM 实际 NF4 4-bit 加载、两条独立合成短题生成，以及 57 条原输入的长度重算。`deepseek_r1 + guidance` 能分开核查正文和最终 JSON，最终 JSON 均合法并以 EOS 结束。但是，明确错误的短题被判为 PASS，原模型卡提示的对照也漏检。最长真实前缀和 17 条小测均未执行；Git/LFS 远端仍返回 503，按仓库要求在源样本采集前停止。本轮不扩到 57 条。

本目录属于技术诊断，不是正式 Stage 3 配对实验。合成题不计入原 17 条，也不替代验证器核验。此前的源输入、复核标签和结果未被改写。

官方模型为 [launch/ThinkPRM-14B](https://huggingface.co/launch/ThinkPRM-14B/tree/1d1b415037cccf826fab6cc1b02c1dac199c0c0f)，固定 revision 为 `1d1b415037cccf826fab6cc1b02c1dac199c0c0f`。六片权重共 `29,540,133,960` bytes（27.51 GiB），579 个 tensor 全部为 BF16，每片 SHA-256 均与官方 LFS 元数据一致。数据盘路径为 `/root/autodl-tmp/thinkprm-14b/1d1b415037cccf826fab6cc1b02c1dac199c0c0f`。Xet 传输停滞后尝试官方 HTTP，再用官方 revision 的镜像 Range 传输并逐片核对官方 SHA；末段慢连接以小段续传和官方 HTTP 补齐。下载残留已清理，数据盘剩余 13.730 GiB。详见 [下载证据](download_complete.json)和[网络尝试](download_network_attempts.json)。

环境为 vLLM 0.20.0、BitsAndBytes 0.50.2、PyTorch 2.11.0、Transformers 5.19.0、llguidance 1.3.0。xgrammar 仍为 0.2.3，本次不使用它，也未修改其安装包。实际启动使用 vLLM Python API 的 `quantization='bitsandbytes', load_format='bitsandbytes'`，对应 CLI 的 `--quantization bitsandbytes --load-format bitsandbytes`；BF16 为原始磁盘权重及计算 dtype，不表示 GPU 以全 BF16 权重运行。

[启动日志](launch.log)和[GPU 模型实检](engine.json)确认：192 个线性层使用 `BitsAndBytesLinearMethod`，权重为 `torch.uint8`，量化状态全部为 `nf4`。其余 147 个参数张量为 BF16。vLLM 报告模型加载占用 9.92 GiB，KV cache 为 18.06 GiB、98,640 token。引擎初始化实测 44.76 秒，全设备显存峰值为 29.082 GiB。KV 分配成功不等于最长前缀完整生成通过。

本轮只有验证器驻留 GPU，无生成器和返工请求；每次仅一个请求。配置为 BF16 计算、上下文 63,891、输出预算 8,192、prefill 分块 1,024、显存比例 0.90、eager 模式、关闭 prefix caching。采样为 greedy（temperature=0、top_p=1、top_k=0），种子 20261110，无额外停止字符串。完整配置见 [manifest.json](manifest.json)。约束配置为 `backend='guidance', reasoning_parser='deepseek_r1', enable_in_reasoning=False`；[Schema](final_schema.json)只含 `verdict`、`diagnosis`、`hint`，verdict 仅允许 PASS/FAIL。

两条合成题均为 `2x+3=7`，完整前缀第一步为 `2x=4`，第二步分别写 `x=2` 和 `x=1`。使用模型卡的原题/解答分区、chat template 和 assistant 后缀，并添加最终 JSON 指令。输入均为 238 token。

| 合成题 | 应判 | 实判 | 输出 token | 完整请求耗时 | 全设备峰值 | 终止 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 正确除法 | PASS | PASS | 278 | 21.68 秒 | 29.435 GiB | EOS |
| 错误除法 | FAIL | PASS | 278 | 17.19 秒 | 29.435 GiB | EOS |

两条均有 228 个核查正文 token，随后是 `</think>`；最终部分含 EOS 共 49 token。引擎返回的输入 token 与请求一致。严格 JSON 解析、Schema 校验、EOS 和正文结束标记检查均通过。正文、最终输出、原始 token、终止原因和显存采样分别保存在 [synthetic/](synthetic/)。显存为 NVML 每 20 ms 采样的全设备峰值，包含 vLLM 预分配 KV 池；耗时包含本次 prefill、生成与结果处理，不含下载和模型加载。

错误题的原始正文明确声称 `4/2=1` 正确，不能把它解释为仅最终 verdict 选错。为定位原因，另起相同配置的引擎做了三个合成题对照，未调用任何源样本：

| 对照 | 输出 token | 耗时 | 结果 |
| --- | ---: | ---: | --- |
| 错误题作为新引擎首请求，保留 guidance | 278 | 20.54 秒 | 仍判 PASS，最终 JSON 合法 |
| 相同提示，移除 Schema 约束 | 237 | 14.88 秒 | 核查仍称除法正确，最终为 `Is the solution correct? Yes` |
| 原始模型卡提示，无新增 JSON 指令、无 Schema | 118 | 7.39 秒 | 仍称除法正确，最终为 Yes |

三个对照均完整 EOS。带 Schema 与不带 Schema 的 token 从首 token 到 `</think>`（索引 228）完全相同，首次差异在索引 229。这直接支持本次 Schema 只约束最终响应，没有造成前面的算术漏检。解析器如实拆分了输出，也没有输出预算耗尽。因此，观察到的是该模型/量化/运行配置的检错失败，而不是本次解析器、约束后端或截断故障。没有 BF16 或其他运行实现的对照，不能进一步认定是量化导致。详见 [controls/](controls/)及[汇总](summary.json)。

两个原始 PASS 输出的 hint 非空，未遵从提示中的“PASS 用空 hint”，但满足用户要求的三字段 Schema。移除 Schema 的对照把 JSON 写在 `</think>` 前，关闭 reasoning 后仍输出原生 Yes；这也说明只在提示里要求 JSON 不足以保证最终响应格式。CPU 回放确认 guidance 能拒绝非法原始 backspace，并允许反斜杠、引号、换行转义；另有一种合法 Unicode 转义被拒绝，而其字面 Unicode 字符可接受。该限制单独保留在 [CPU 记录](cpu_backend_check.json)，不据此宣称后端支持所有合法 JSON 表达形式。

57 条输入使用原始完整 `quoted_prefix`，未引入标准答案、标准解、复核标签或未来文本，也未对原始反斜杠进行二次解码。使用 ThinkPRM tokenizer 和实际模板重算后，最短 305、中位 12,906、最长 55,699 token。最长为第 34 条，原前缀 147,596 字符；计划上下文为 `55,699 + 8,192 = 63,891`。CPU 编解码回环保留全文，详见 [逐条长度](input_inventory.json)。tokenizer 元数据的 4,096 提示产生过长度警告，但本轮没有截断；模型配置上限为 131,072，vLLM 已按 63,891 初始化。**最长源输入未做 GPU 生成，输出 token、耗时、实际生成峰值及终止原因均为未测。**不能将其归类为权重 OOM、KV 不足或输出截断。

17 条计划保留关键漏检 `18,19,23,24,25,33,52`，原 9B FAIL `32,34,50,53,54`；暂定 PASS 为 `4,22,43,46,49`。仓库没有找到预先冻结的五条 PASS 清单，这五条是本轮按既有复核口径选择，覆盖简单正确、已纠正错误和废弃支路；没有据 ThinkPRM 输出选样。原复核为非盲模型暂定对照，并非独立专家真值。见 [选样](selection.json)和[待执行逐条表](audit17_pending.csv)。实际完成 **0/17**，检错、定位、反馈可用性、误报和无有效判决的源样本统计均为 NA，而非零。

远端复查中，Git 读取/推送退出 128，LFS 推送退出 2，均明确返回 HTTP 503。未关闭 LFS 检查、未修改传输配置。错误原文见 [backup_status.json](backup_status.json)。[Stage 3 README](../../../README.md)要求验证器 smoke、开发和核验启动前远端备份成功，失败保留本地并停止；因此未开始真实最长前缀和 17 条采集。

当前不支持扩大到 57 条。后续应先恢复 Git/LFS 远端备份，并查清简单算术漏检在模型、量化或运行实现之间的来源。只有短题检错可靠后，才在新的诊断记录中执行最长完整生成及既定 17 条；本目录保留首轮失败，不能改写为后续修复成功。
