# Multimodal AI Media Assistant

An end-to-end system that turns a short recording into a timestamped
transcript, a schema-validated structured analysis, a semantic index,
grounded question answering with timestamp citations, a bounded
tool-using agent, and a multilingual analysis.

Every stage runs locally on open-source models. Nothing is sent to an
external API.

**Author:** Wu Dongrun

---

## Quick start

```bash
uv sync                                    # create .venv and install
python -m tests.test_smoke                 # 16 checks, ~1 s, no models loaded
python -m media_assistant.cli --media sample_media/meeting.wav
```

That one command writes every deliverable: `transcript.json`,
`analysis.json`, `report.md`, `transcript_bilingual.json` and
`visual_output.png`.

To re-run the analysis stages on a transcript that already exists,
without paying for transcription again:

```bash
python -m media_assistant.cli --transcript transcript.json
```

`project.ipynb` walks through the same stages one at a time with output
shown at each step. Run it from the project root so `import
media_assistant` resolves.

Useful flags: `--small` (smaller fallback models), `--skip-translation`
(runs Part 5 without the per-segment translation, which is the slowest
stage), `--top-k`, `--max-agent-steps`, `--out`.

---

## Models

| Role | Model | Size | Notes |
|---|---|---|---|
| Speech recognition | `openai/whisper-large-v3-turbo` | 809 M | fp16, sequential long-form decoding |
| Language model | `Qwen/Qwen3-4B-Instruct-2507` | 4 B | bf16; a non-thinking instruct variant, so no `<think>` blocks to strip out of JSON |
| Text embeddings | `Qwen/Qwen3-Embedding-0.6B` | 0.6 B | 1024-dim, fp32, asymmetric query/document prompts |

`Config.small()` swaps in `whisper-base` and `Qwen3-1.7B` for machines
that cannot hold the stack above. Chinese transcription is noticeably
worse and the 1.7B model is a hybrid reasoning model whose `<think>`
blocks the JSON extractor has to strip, so it is a fallback rather than
the configuration the submitted results came from.

Every tunable lives in [`media_assistant/config.py`](media_assistant/config.py);
the full configuration is written into `transcript.json` with each run.

## Hardware tested

| | |
|---|---|
| Machine | Apple M1, 16 GB unified memory, macOS 26.1 |
| Accelerator | Metal (MPS). **No CUDA** — the recommended Colab T4 path was not used |
| Python | 3.12.13 · torch 2.13.0 · transformers 5.15.0 · sentence-transformers 5.7.0 |

Measured on this machine:

- transcription runs at a realtime factor of **0.43** (a 5-minute
  recording takes about 2 minutes);
- the language model generates about **5 tokens/second**, which is what
  a 4 B bf16 model costs on the M1's memory bandwidth;
- a full CLI run, including the per-segment translations, takes roughly
  **25–35 minutes**. `--skip-translation` cuts that by more than half.

### The 16 GB constraint shapes the architecture

Whisper-turbo, the 4 B language model and the embedding model do not fit
in 16 GB at once. Each model wrapper therefore has an explicit
`load()` / `release()` pair, and `MediaAssistant.process` runs stages in
the order that keeps peak memory down:

```
prepare audio → transcribe → (Whisper translate pass) → RELEASE ASR
             → build index → analyse with the LLM → answer / agent
```

Both Whisper passes happen before the ASR model is released, because
reloading it later would cost more than the second pass itself. The
embedding model stays resident, since every later query has to be
encoded.

---

## What each part does

### Part 1 — media ingestion and transcription
[`media.py`](media_assistant/media.py) · [`speech.py`](media_assistant/speech.py)

Accepts audio or video and converts the audio track to mono 16 kHz WAV.
There is no system `ffmpeg` on the test machine, so every call goes
through the binary bundled with `imageio-ffmpeg` — including audio
decoding, because the transformers ASR pipeline shells out to a system
`ffmpeg` when handed a file path.

`probe_metadata` reports the container's real contents before anything is
decoded, so the cases that actually occur are caught up front: a missing
path (`FileNotFoundError`), a zero-byte file, a text file renamed
`.wav`, and a video with **no audio track** — all `MediaError` with a
message naming the problem.

Timestamps stay numeric (`start`, `end` in seconds) everywhere inside the
system and are formatted only for display.

**On confidence:** no per-segment confidence score is exposed. Where the
decoder reports `avg_logprob` it is passed through unchanged, as a mean
token log probability. It is not calibrated, and it is never rescaled
into a percentage.

### Part 2 — LLM analysis with structured output
[`language.py`](media_assistant/language.py)

The model returns JSON; Python decides whether to accept it.
`check_analysis` verifies required keys, field types, that every
list-valued field really is a list, and the cardinality rules the brief
states in prose (3–6 key points, at least one topic). On failure the
specific problems are fed back and the model is asked to fix them, up to
`analysis_max_retries` times — the first attempt greedy so a good run is
reproducible, later attempts sampled so a bad answer is not simply
repeated. Still invalid after every retry raises
`AnalysisValidationError`; a partial analysis is never returned.

`entities` and `action_items` accept either a plain string or an object,
because a 4 B model is markedly more reliable when not forced into one
nested shape. Both are normalised to objects, so nothing downstream has
to branch.

### Part 3 — semantic search and RAG
[`chunking.py`](media_assistant/chunking.py) · [`retrieval.py`](media_assistant/retrieval.py)

Segments are grouped into chunks of about 30 seconds under two rules:

1. **A chunk never spans a change of language.** In a bilingual recording
   a switch is also a topic boundary, and averaging both languages into
   one vector blurs exactly what the cross-lingual test in Part 5
   measures.
2. **Neighbouring chunks overlap by one segment**, so a fact stated
   across a boundary is still whole in at least one chunk. The overlap is
   applied only within a language run, for the same reason.

Queries and documents are embedded asymmetrically: the query side gets
Qwen3-Embedding's instruction prompt, the document side does not. This
matters most for cross-lingual retrieval. Vectors are L2-normalised, so
cosine similarity is a dot product; with a few dozen chunks a dense
matmul beats any index structure, and there is nothing to build.

`ask()` returns the retrieved evidence *and* the answer, so retrieval and
generation can be marked separately. Unanswerable questions are caught
twice:

- **before generation** — if the best similarity is below
  `min_similarity`, the question is refused without calling the model;
- **during generation** — the model is told to emit an
  `INSUFFICIENT_EVIDENCE` sentinel rather than improvise.

Either path returns `answered: False` with a machine-readable
`refusal_reason`.

Citations are parsed back out of the answer and checked against the
chunks that were actually retrieved, so a plausible-looking but invented
timestamp is reported as ungrounded rather than counted.

**Retrieval threshold.** `min_similarity = 0.42` was calibrated, not
guessed: on the development transcript, questions the recording cannot
answer topped out at **0.352**, while the weakest genuinely answerable
question scored **0.520**. 0.42 sits between them, leaning low because a
false refusal costs more than a false accept — the model-side sentinel is
the second line of defence.

### Part 4 — tool-using agent
[`tools.py`](media_assistant/tools.py) · [`agent.py`](media_assistant/agent.py)

Four deterministic tools: `search_transcript`, `count_mentions`,
`get_transcript_range`, `get_media_metadata`.

The model only ever *chooses* a tool. The application parses its JSON,
validates the tool name and every argument against a declarative
`ParamSpec`, calls the Python function itself, and returns the result as
an observation. The step limit is enforced by the loop, not requested of
the model; when it is reached the agent stops calling tools and answers
from what it already observed.

Unknown tool names, malformed JSON, missing arguments, wrong types,
undeclared parameters and out-of-range values all become observations the
model can recover from — a wrong guess costs a step, not the run. Every
step is recorded, so the agent's behaviour appears in `report.md` rather
than being taken on trust.

`count_mentions` normalises before counting: case folding, full-width to
half-width, and Chinese numerals 0–99 mapped to digits, because Whisper
writes the same spoken number as `19` in one segment and `十九` in the
next. Numerals above 99 are left alone rather than converted
half-correctly.

### Part 5 — multilingual media mode (Option C)
[`multilingual.py`](media_assistant/multilingual.py)

Four things rather than one call to a translate function:

1. **Language detection per segment** (Whisper's `return_language`),
   aggregated into a distribution and a timeline of spans.
2. **The original transcript preserved.** Translations live in separate
   fields (`translation_llm`, `translation_whisper`) so no later stage
   can accidentally embed a translation and retrieve against it.
3. **Two independent translation routes compared** — Whisper decoding the
   audio directly into English (`task="translate"`), and the LLM
   translating the transcript text. They disagree in interesting places:
   Whisper hears the audio but has no document context, the LLM has
   context but only sees text that may already contain recognition
   errors. The routes are aligned by time overlap, since the translate
   pass re-segments the audio.
4. **Cross-lingual retrieval compared.** The recording deliberately
   states some facts in only one language; asking in the other language
   tests whether retrieval is semantic rather than lexical. The
   comparison reports both languages' top-k chunks, their scores, whether
   the top hit agrees, and the overlap.

`visual_output.png` shows the language timeline and the cross-lingual
comparison table.

### Part 6 — reliability and tests
[`reliability.py`](media_assistant/reliability.py) · [`tests/test_smoke.py`](tests/test_smoke.py)

```bash
python -m tests.test_smoke                 # 16 checks, no models loaded
python -m tests.test_smoke --with-models   # adds the LLM-backed checks
pytest tests/test_smoke.py                 # pytest collects the same functions
```

The four required conditions:

| Condition | Where |
|---|---|
| normal input | transcript and index invariants; a factual question answered with grounded citations |
| invalid / missing file | missing path, zero-byte file, text file renamed `.wav`, video with no audio track |
| RAG question with weak evidence | the budget question, which the recording never mentions |
| tool / agent error | unknown tool, missing argument, wrong type, undeclared parameter, out-of-range value, malformed JSON, step-limit exhaustion |

Deterministic assertions include `start <= end` on every segment, text
present on every segment, unique segment and chunk ids, monotonic
ordering, one embedding per chunk, and unit-norm embeddings.

The agent tests use a stub LLM that returns scripted replies, so the loop
mechanics — step limit, error recovery, tool dispatch — are tested
without a model and without randomness.

`run_reliability_checks` is also called by the CLI, so the results appear
in section 7 of `report.md`: the same checks the tests run, recorded
against the actual submitted run.

---

## Layout

```
media_assistant/
├── config.py         every tunable, in one place
├── exceptions.py     one named type per intentional failure mode
├── utils.py          timestamp formatting, seeding, memory release
├── media.py          Part 1 — ingestion, probing, audio extraction
├── speech.py         Part 1 — Whisper, with an explicit model lifecycle
├── llm.py            shared LLM wrapper + tolerant JSON extraction
├── language.py       Part 2 — analysis, schema, validation, retries
├── chunking.py       Part 3 — segments to chunks
├── embeddings.py     Part 3 — asymmetric query/document embedding
├── retrieval.py      Part 3 — index, search, grounded answering
├── tools.py          Part 4 — deterministic tools + argument validation
├── agent.py          Part 4 — the bounded loop
├── multilingual.py   Part 5 — Option C + the visual output
├── reliability.py    Part 6 — the recorded checks
├── report.py         report.md and artifact writing
├── demo.py           the five required questions and the agent tasks
├── assistant.py      facade; owns model lifetimes and run state
└── cli.py            one command, every deliverable

tests/
├── fixtures/sample_transcript.json
└── test_smoke.py

sample_media/
├── recording_script.md    the script the recording was read from
└── meeting.wav            the recording itself
```

`project.ipynb` sits at the project root so `import media_assistant`
works without installing the package — `pyproject.toml` keeps
`package = false` deliberately.

`Final_Project_Multimodal_AI_Media_Assistant.ipynb` is the original
project brief, kept for reference. It is not the submitted notebook.

---

## Test media

A self-recorded bilingual (Mandarin/English) project meeting, read from
[`sample_media/recording_script.md`](sample_media/recording_script.md).
The recording is 4 minutes 31 seconds and was converted to mono 16 kHz
WAV with the command in that script. Whisper segments it into 55
segments: 43 Mandarin (167.7 s, 66%) and 12 English (86.6 s, 34%), across
6 language switches.

**This is a scripted scenario read aloud by the author, not a recording
of a real meeting.** The teammates named in it (Chen Zhiyuan, Priya Nair,
Marcus Weber, 林晓雯) are fictional, as are its dates, which are not
internally consistent — the meeting is dated 11 August while an action
item falls due on 9 August. One speaker, no third-party voices, so no
permission from anyone else was required and no biometric data is
included.

The script deliberately plants material the evaluation needs: specific
numbers and dates, named people and organisations, explicit decisions and
owned action items, four distinguishable topics, and two facts stated in
only one language each so that cross-lingual retrieval can be tested
honestly. It contains no monetary amounts at all, which is what makes the
budget question genuinely unanswerable.

`tests/fixtures/sample_transcript.json` is a **hand-written** transcript
of the same script, used to develop and test Parts 2–6 before the
recording existed. It is a test fixture, not media, and not a model
output.

---

## Mapping to the brief's starter API

The brief sketches a few function names. They all exist, though the
package passes state explicitly rather than through module globals.

| Brief | Here |
|---|---|
| `extract_audio(input_path, output_wav)` | `media.extract_audio` — same signature |
| `transcribe_audio(audio_path)` | `speech.transcribe_audio` (alias for `speech.transcribe`), or `SpeechRecognizer.transcribe` to reuse a loaded model |
| `REQUIRED_ANALYSIS_KEYS` | `language.REQUIRED_ANALYSIS_KEYS` — the same eight keys |
| `analyze_transcript(transcript_text)` | `language.analyze_transcript` — also accepts segments |
| `build_transcript_index(transcript_segments)` | `retrieval.build_transcript_index` |
| `search_transcript(query, top_k=3)` | `retrieval.search_transcript(query, top_k, index=...)`, or `MediaAssistant.search` |
| `ask_media(question, top_k=3)` | `retrieval.ask_media(question, top_k, index=...)`, or `MediaAssistant.ask` |
| `TOOL_FUNCTIONS` | `Toolbox.TOOL_FUNCTIONS` — per instance, so the tools are bound to one recording's data rather than to import-time globals |
| `run_agent(user_request, max_steps=5)` | `agent.run_agent(request, toolbox, max_steps=...)`, or `MediaAssistant.agent` |
| `MediaAssistant.process / search / ask / agent` | `assistant.MediaAssistant` — same four methods |

---

## Building the submission

```bash
python -m media_assistant.cli --media sample_media/meeting.wav   # produce the artifacts
python -m tests.test_smoke                                       # confirm the checks pass

cd ..
cp -R Final_Project final_project_wu_dongrun
cd final_project_wu_dongrun
rm -rf .venv .cache_media __pycache__ media_assistant/__pycache__ tests/__pycache__ .remember
cd ..
zip -r final_project_wu_dongrun.zip final_project_wu_dongrun
```

Removing `.venv` matters: it holds the model weights and would make the
archive several gigabytes. The models re-download on `uv sync` from the
ids listed above.

---

## Known limitations

- **Language detection is per segment, not per sentence.** A sentence
  that switches language mid-way is labelled with whichever language
  Whisper predicts for the segment containing it. Note that this needs a
  dedicated pass (`SpeechRecognizer.detect_segment_languages`): Whisper's
  sequential long-form decoding detects one language from the first 30
  seconds and stamps it on every segment, which makes any recording look
  monolingual. A reliability check now fails the run if a segment's
  writing system contradicts its label.
- **Whisper's translate task needs a different checkpoint.**
  `large-v3-turbo` is a pruned large-v3 re-finetuned on transcription
  data only, so `task="translate"` returns the source language instead of
  English — verified on this recording. The translate route therefore
  loads `large-v3` separately, which is why a full run takes ~43 minutes
  rather than ~15.
- **The Whisper translate route sees each segment in isolation.** It is
  run per segment so that its output aligns exactly with the transcript,
  but that costs it the surrounding context the LLM route has. The
  comparison in the report reflects that asymmetry, and it is the main
  reason the LLM route reads better on this recording.
- **Q4's answer is under-evidenced.** The multi-chunk question is
  answered correctly, but the retrieved chunks contain only the 15 August
  submission deadline, not the 9 August action-item deadline that the
  question also depends on. The conclusion happens to be right; the
  evidence shown for it is incomplete. `top_k` was left at 3 rather than
  raised to make this one row look better.
- **Mandarin recognition contains homophone errors.** `多模态` is
  transcribed as 多摩台, `期末项目` as 寂寞项目, `字段` as 自断, `归一化`
  as 规议化, and `Priya Nair` as Ryanair. These are left uncorrected: the
  transcript is model output, and hand-editing it would misrepresent what
  the pipeline produces.
- **A face being visible would not prove someone is speaking** — noted
  because this project chose Option C precisely to avoid making
  visibility claims about people. There is no speaker diarization here:
  the system does not know who is talking, only what language is being
  spoken.
- **`count_mentions` normalises Chinese numerals only up to 99.** Larger
  numbers are compared as written, so `一千二百` and `1200` count
  separately.
- **The similarity threshold is calibrated on one recording.** A
  different recording, or a noisier one, may need a different value; it
  is one line in `config.py`, and the calibration procedure is described
  above.
- **Timestamp citations are checked for existence, not for relevance.** A
  citation that overlaps a retrieved chunk is counted as grounded even if
  that chunk does not actually support the sentence it is attached to.
- **Generation is slow on this hardware** (~5 tokens/second). The
  pipeline is correct but not interactive; a CUDA machine or a quantised
  model would change that, and neither was used for the submitted
  results.
- **The two translation routes are compared with a word-overlap ratio**,
  which flags rows worth reading and nothing more. It is not a
  translation quality metric.
- **`analysis_language` forces the report into English.** A transcript in
  a third language would still be summarised in English unless that
  setting is changed.
