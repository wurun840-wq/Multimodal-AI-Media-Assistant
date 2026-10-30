# 录音文案 — Final Project 组会（Option C 多语言模式）

自录素材，单人朗读，中英交替。目标时长约 5 分钟。
内容是这个 Final Project 本身的进度讨论（自指式素材，做 demo 时很好讲）。
队友姓名为虚构；README 中需注明这是作者本人朗读的**脚本化场景**，不是真实会议录音。

---

## 第 1 段 · 中文 · 开场与议程（约 0:00–0:30）

大家好，今天是二零二六年八月十一号，星期二。这是我们多模态媒体助手期末项目的第三次碰头，我是吴东润，负责这次的记录。今天要过四件事：第一是转录管线的进展，第二是结构化分析和检索，第三是工具 agent 的设计，第四是多语言扩展和提交时间安排。

> 停顿一秒

## 第 2 段 · English · Part 1, transcription pipeline（约 0:30–1:10）

Let's start with Part One, the transcription pipeline. Chen Zhiyuan got Whisper large v3 turbo running on Colab last Thursday. On our five minute forty second test recording, it produced forty-seven timestamped segments, and the whole transcription took about fifty seconds on a T4. We checked the segment boundaries by hand, and only three of them were split at an awkward place, which is good enough for the chunking we want to do later.

> 停顿一秒

## 第 3 段 · 中文 · Part 2 与 Part 3（约 1:10–1:55）

第二部分和第三部分是我做的。结构化分析这块，我让模型输出 JSON，八个必需字段都要校验，列表字段必须真的是列表。第一版大概有五分之一的概率会在 JSON 外面多输出一段解释文字，我加了重试，最多重试两次，现在基本稳定了。检索这边，我按三十秒左右一块来切，整段录音一共切出十九个 chunk。我们用 Qwen3-Embedding-0.6B 测了十二个跨语言查询，有十个在 top-3 里命中了正确的片段，这个结果我觉得可以接受。

> 停顿一秒

## 第 4 段 · English · Part 4, the agent loop（约 1:55–2:45）

Now Part Four, the agent. Priya Nair is building the tool loop. We have three tools so far: search transcript, count mentions, and get media metadata. The loop is capped at five steps, and an unknown tool name returns an error observation instead of crashing the whole run. One thing worth recording: the T4 only has sixteen gigabytes of memory, so we have to delete the Whisper model and empty the CUDA cache before we load the language model. Don WU


 hit an out of memory error twice before we figured that out.

> 停顿一秒

## 第 5 段 · 中文 · Part 5 多语言扩展（约 2:45–3:20）

第五部分我们选的是 Option C，多语言模式，不做人脸识别。我们的测试录音本身就是中英交替的，所以要保留原文转录，再单独生成一份英文翻译，两份都要存下来。Don提了一个我觉得很好的点：不要用同一段文字既做检索又做翻译展示，容易混淆，应该在数据结构里分成两个字段。这一条我们采纳了。

> 停顿一秒

## 第 6 段 · English · Schedule and deadline（约 3:20–3:55）

About the schedule. The submission deadline is August fifteenth and it will not move. We also need to record the three to five minute demo video, and we agreed to do that on August thirteenth, after everything else is frozen. That leaves us two days of buffer, which is not much, so nothing should slip after the thirteenth.

> 停顿一秒

## 第 7 段 · 中文 · 遗留问题（约 3:55–4:25）

还有两个没解决的问题。第一，本地跑的时候没有英伟达显卡，只能换更小的模型，README 里必须写清楚我们实际测的硬件和模型，不能照抄推荐配置。第二，Whisper 输出中文数字的时候，有时候是阿拉伯数字，有时候是汉字，这会影响 count mentions 这个工具的统计，陈志远说他加一层归一化。

> 停顿一秒

## 第 8 段 · 中文 · 决定与行动项汇总（约 4:25–5:00）

最后过一遍今天的决定和行动项。决定有三个：第一，多模态扩展确定选 Option C，不做人脸；第二，结构化输出加重试，最多两次；第三，原文转录和英文翻译分成两个字段存。行动项有四个：Priya 在八月九号之前把 agent 的多工具任务演示做完；陈志远加数字归一化；Don负责录 demo 视频；我在这周五之前把 README 的初稿写出来。下次碰头是八月八号星期二下午三点。今天就到这里，谢谢大家。

---

## 录制注意事项

- 语速正常偏慢，数字念清楚；段与段之间停顿约 1 秒，方便 Whisper 切出干净的 segment 边界。
- 安静环境，无背景音乐。手机或电脑内置麦克风即可。
- 录完统一转成 16 kHz 单声道 WAV（Whisper 内部采样率）：
  ```bash
  $(.venv/bin/python -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())") \
    -i raw.m4a -ac 1 -ar 16000 sample_media/meeting.wav
  ```
  本机没有系统 ffmpeg，用 `.venv` 里 `imageio_ffmpeg.get_ffmpeg_exe()` 返回的二进制。
- 文件名用 ASCII，避免中文路径在子进程调用中出问题。
- README 中注明：self-recorded by the author, single speaker, scripted scenario, 队友姓名为虚构，无第三方声音，无需额外授权。

## 本文案对评分点的覆盖

| 评分要求 | 文案中的对应内容 |
|---|---|
| 两种语言真实出现 | 中文 5 段 / 英文 3 段，交替出现，非逐句翻译 |
| 具体数字 | 5:40 录音、47 个 segment、50 秒、8 个必需字段、1/5 概率、重试 2 次、30 秒一块、19 个 chunk、12 个查询 10 个命中、3 个工具、5 步上限、16 GB、8/9、8/13、8/15、2 天缓冲 |
| 人名 | 林晓雯、陈志远、Priya Nair、Marcus Weber |
| 机构 / 产品名 | Colab、T4、Whisper large v3 turbo、Qwen3-Embedding-0.6B、英伟达 |
| 明确决定 | 3 条（选 Option C / 加重试 / 原文与译文分字段） |
| 行动项 | 4 条，均带责任人，1 条带截止日期 |
| 可区分话题 | 转录 / 分析与检索 / agent / 多语言与排期 = 4 个 |

## 配套的五个 RAG 测试问题

| # | 类型 | 问题 | 证据位置 |
|---|---|---|---|
| 1 | 直接事实 | 整段录音一共切出了多少个 chunk？ | 第 3 段（中文） |
| 2 | 直接事实 | How many timestamped segments did Whisper produce? | 第 2 段（英文） |
| 3 | 换词表述 + 跨语言 | agent 最多能跑几轮？（原文说的是 "capped at five steps"） | 第 4 段（英文） |
| 4 | 跨多个 chunk + 跨语言 | agent 的演示能在提交截止之前做完吗？ | 第 8 段（8/19）+ 第 6 段（8/25） |
| 5 | 无法回答 | 这个项目的预算是多少？ | 全文未提及任何金额、成本、预算 |

**录制时不要临场加入任何金额、成本、报价相关的内容**，否则第 5 题就不再是 unanswerable，Part 3 的 2 分和 Part 6 的第 3 项测试都会失效。

## Option C 要求的跨语言检索对比

刻意设计了两个"只在单一语言中出现"的事实，用另一种语言去查，验证 embedding 的跨语言能力：

| 事实 | 只出现在 | 用对侧语言查询 |
|---|---|---|
| 12 个跨语言查询中 10 个 top-3 命中 | 中文（第 3 段） | "How well did the embedding model do on cross-lingual retrieval?" |
| T4 只有 16 GB，须先删 Whisper 再加载 LLM | English（第 4 段） | "为什么要在加载语言模型之前删掉 Whisper？" |

对每个查询记录 top-k chunk、相似度分数，以及同一问题用两种语言提问时的检索结果差异，填进 `report.md` 的多语言对比表。

**注意**：第 8 段的汇总只复述决定和行动项，不复述上面这两个事实——不要在录制时即兴补充，否则单语言独占性被破坏，跨语言对比就失去意义。
