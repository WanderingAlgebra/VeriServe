# 结构化输出非法JSON的根因诊断

结论：当时确实使用了vLLM原生结构化采样。第21、22条非法JSON的可复现原因是所安装xgrammar 0.2.3对`minLength: 1`字符串字段生成了错误语法，既漏掉部分原始控制字符禁令，也遗漏合法JSON转义分支。不是只在提示词里要求JSON，也没有证据表明模型绕过了正确的采样掩码。

## 原始配置及出错机制

执行代码`stages/stage3/quantized_prefix_audit_bounded.py`向LLM传入`structured_outputs_config={'backend':'xgrammar'}`，每次采样传入`StructuredOutputsParams(json=req['schema'])`。诊断与FAIL提示字段的schema包含`type: string, minLength: 1`，本意是确保非空。

本机xgrammar将这些字段编译成：

```text
root_case_0_prop_0 ::= "\"" [^"\\\r\n]{1,} "\""
```

字符类仅排除了双引号、反斜杠、CR和LF，漏排U+0000—U+001F中的其他原始控制字符；同时没有JSON转义分支。因此字面的U+0008会被当成合法字符串内容，而合法的反斜杠转义反而不可输出。普通无长度约束字符串使用的`basic_string_sub`包含控制字符禁令及转义分支，未出现这项特定问题。

## CPU复现证据

没有重新调用模型。使用当时同一tokenizer、相同248320词表大小、原schema和已保存的原始生成token逐个重放，并读取出错位置的下一个token掩码：

|样本|出错token位置，0起算|token ID|原语法掩码允许|原语法能接受整条至EOS|修正语法原型允许|
|---|---:|---:|---|---|---|
|21|107|196|是|是|否|
|22|99|196|是|是|否|

token196在xgrammar的词表解码与实际tokenizer解码均为U+0008，排除了本例中二者对这个token的字节解释不一致。原语法确实允许它进入候选集合，严格JSON解析器随后正确拒绝了文本。

最小schema对照也确认：

|测试内容|普通string|带minLength=1的原语法|修正语法原型|
|---|---|---|---|
|非空普通文本|允许|允许|允许|
|空字符串|允许|禁止|禁止|
|字面backspace或TAB|禁止|错误地允许|禁止|
|合法转义的反斜杠、引号或换行|允许|错误地禁止|允许|

修正原型仍要求非空，使用“合法原始字符或合法JSON转义”的重复来表达非空字符串。这只是对编译语法的CPU验证，未修改安装包、旧运行脚本、schema文件、原始输出或判定标签，也未完成GPU生成复测。相关EBNF、[evidence.json](evidence.json)和[reproduce.py](reproduce.py)保留在本目录。

上游同类问题：[xgrammar issue #800](https://github.com/mlc-ai/xgrammar/issues/800)。这里的结论以本机复现为主要依据，并非仅凭网上相似报告推断。

## 对既有结果解释的补充

- 第21、22条保留原始“格式错误、无有效判决”状态；不能事后修补文本后计入PASS。
- 这两条格式失败应归于当前结构化后端缺陷，不能用来证明27B的数学判断能力差。
- 9B结构化首轮也使用同一后端和相同类型的非空字符串约束，因此这不是27B量化版本专属问题。9B当次没有触发非法控制字符，不代表其生成内容完全未受错误转义限制影响。
- 54条有效输出全为PASS仍是该次运行的事实；由于错误语法也限制了正常文本的候选选择，修复后表现需要另行测量，不能把本轮结果当作不受后端影响的27B能力结论。
- 第27条的重复生成及输出截断是已观察到的另一种现象；与本语法缺陷是否有因果关系尚未验证。

复现命令，在仓库根目录执行，仅使用CPU：

```bash
/root/veriserve-vllm-env/bin/python stages/stage3/diagnostics/qwen35-27b-quantized-prefix-audit/20261110-json-constraint-diagnosis/reproduce.py
```
