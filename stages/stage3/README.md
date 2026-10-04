# 第三阶段：探针风险与即时检查收益

固定第二阶段 7B greedy 生成器与 B/C 探针，在原 `unused_ids` 顺序划分的
20 道 pilot、40 道 dev、130 道 test 上执行 Step 2 公共快照的 NOW / DELAY_2 配对。
B 表示未经干预完整轨迹最终答错风险；每个合格快照均执行两个动作。
PRM 沿用第一阶段 NF4 / BF16 compute / double quant 与确定性通用反馈，阈值固定 0.35。

在仓库根目录使用已有 Python：

```bash
/root/miniconda3/bin/python -m stages.stage3.run_timing --config stages/stage3/config.json --phase prepare --resume
/root/miniconda3/bin/python -m stages.stage3.run_timing --config stages/stage3/config.json --phase self-check
/root/miniconda3/bin/python -m stages.stage3.run_timing --config stages/stage3/config.json --phase backup --resume
/root/miniconda3/bin/python -m stages.stage3.run_timing --config stages/stage3/config.json --phase pilot --resume --stop-after 1
/root/miniconda3/bin/python -m stages.stage3.run_timing --config stages/stage3/config.json --phase pilot --resume
/root/miniconda3/bin/python -m stages.stage3.run_timing --config stages/stage3/config.json --phase dev --resume
/root/miniconda3/bin/python -m stages.stage3.run_timing --config stages/stage3/config.json --phase test --resume
/root/miniconda3/bin/python -m stages.stage3.run_timing --config stages/stage3/config.json --phase analyze --resume
/root/miniconda3/bin/python -m stages.stage3.run_timing --config stages/stage3/config.json --phase backup --resume
```

`prepare` 在模型下载前固定 PRM SHA，再确定 `runs/<seed>-<config_hash>/`。
`manifest.json` 保留原题完整字段、原顺序 ID、配置、探针和导入源文件哈希与环境。
`snapshots/<key>.json/.npz` 保存原 token 前缀、pending 和选定层特征；
`arms/<key>.NOW.json` / `.DELAY_2.json` 保存独立路径、真实检查、反馈、预算与时间。
不保存 GPU KV，不改动 stage1/stage2。

每 10 个完整题目进行普通提交和推送，只纳入 stage3 与必要 LFS 元数据；
推送失败会停止后续批次，恢复先重试备份。
第一批有配对结果后，从独立临时 checkout 取回快照、两臂和 NPZ 并验证 SHA256。
损坏 JSON 保留原字节，并优先从已成功提交备份恢复；无法恢复则明确停止。

dev 的 B/C 三分位边界只使用合格快照的风险分数，test 开始前冻结并成功备份。
恢复时会核对成功备份所包含的冻结文件、分组和 source 哈希；
只有本地冻结文件或旧的成功备份不能放行 test，必须先保存待备份状态并补推。
主时间指标 `T_postfork_wall` 是“从公共快照到结束的后续处理时间”；
公共生成、离线特征/探针和实验分叉恢复单独记录，FAIL 后真实恢复不扣除。
臂内解析、控制和检查时的原子状态/事件持久化开销计入后续时间，
模型下载/加载、外部 Git 备份和事后 Math-Verify 评分不计入。
基础设施中断后，原配对臂文件移入 `interrupted_pairs/` 留存，
使用保存的同一原始 token 快照重新完成两个臂，并以新的完整同卡配对作为正式计时。
长期快照只保存 token 与 CPU 特征，不保存 GPU KV。
`analyze` 仅 CPU 读取已保存结果，pilot/dev/test 分开报告；未真实执行的阶段标为未执行 / NA。
