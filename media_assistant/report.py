"""Report generation and artifact writing.

Produces the four deliverables the brief asks for -- ``transcript.json``,
``analysis.json``, ``report.md`` and the visual output -- from the state
one run left behind. Nothing is recomputed here: the report renders what
actually happened, so it cannot disagree with the run it describes.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .agent import AgentRun
from .config import Config
from .utils import format_segment_line, format_span, truncate


def _table(header: list[str], rows: list[list[str]]) -> str:
    """Render a markdown table, or a placeholder when there is no data."""
    if not rows:
        return "_(none)_\n"
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        cells = [str(cell).replace("|", "\\|").replace("\n", " ") for cell in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _bullets(items: list[str]) -> str:
    if not items:
        return "_(none)_\n"
    return "\n".join(f"- {item}" for item in items) + "\n"


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------

def _section_overview(media: dict, transcript_metadata: dict, config: Config) -> str:
    duration = transcript_metadata.get("audio_duration_seconds") or media.get("duration_seconds", 0)
    rows = [
        ["Source file", media.get("filename", "—")],
        ["Duration", f"{format_span(0, duration)} ({duration:.1f} s)"],
        ["Container / streams", f"{media.get('container', '—')}, "
                               f"audio={media.get('has_audio', '—')}, video={media.get('has_video', '—')}"],
        ["Speech model", transcript_metadata.get("asr_model", config.asr_model)],
        ["Language model", config.llm_model],
        ["Embedding model", config.embedding_model],
        ["Device / dtype", f"{transcript_metadata.get('device', config.device)} · "
                           f"ASR {transcript_metadata.get('asr_dtype', config.asr_dtype)} · "
                           f"LLM {config.llm_dtype}"],
        ["Segments", transcript_metadata.get("segment_count", "—")],
        ["Transcription time", f"{transcript_metadata.get('transcription_seconds', 0):.1f} s "
                               f"(realtime factor {transcript_metadata.get('realtime_factor', '—')})"],
    ]
    note = transcript_metadata.get("confidence_note")
    body = "## 1. Recording and configuration\n\n" + _table(["Property", "Value"], rows)
    if note:
        body += f"\n> **On confidence:** {note}\n"
    return body


def _section_analysis(analysis: dict) -> str:
    if not analysis:
        return "## 2. Structured analysis\n\n_(analysis was not run)_\n"

    validation = analysis.get("_validation", {})
    action_rows = [
        [item.get("description", ""), item.get("owner") or "—", item.get("due") or "—"]
        for item in analysis.get("action_items", [])
    ]
    entity_rows = [
        [item.get("name", ""), item.get("type", "")] for item in analysis.get("entities", [])
    ]

    return (
        "## 2. Structured analysis (LLM, schema-validated)\n\n"
        f"**{analysis.get('title', '—')}**\n\n"
        f"{analysis.get('summary', '—')}\n\n"
        "### Key points\n\n" + _bullets(analysis.get("key_points", [])) +
        "\n### Topics\n\n" + _bullets(analysis.get("topics", [])) +
        "\n### Action items and decisions\n\n" + _table(["Item", "Owner", "Due"], action_rows) +
        "\n### People and organisations mentioned\n\n" + _table(["Name", "Type"], entity_rows) +
        "\n### Languages observed\n\n" + _bullets(analysis.get("languages", [])) +
        "\n### Suggested follow-up questions\n\n" + _bullets(analysis.get("follow_up_questions", [])) +
        "\n### Validation\n\n"
        f"Accepted on attempt {validation.get('attempts', '—')} of "
        f"{validation.get('max_attempts', '—')}. Checked keys: "
        f"`{'`, `'.join(validation.get('checked', []))}`. Every list-valued field was "
        "confirmed to be a list before the analysis was accepted.\n"
    )


def _section_index(index_summary: dict, config: Config) -> str:
    rows = [[key.replace("_", " "), json.dumps(value, ensure_ascii=False)
             if isinstance(value, dict) else value]
            for key, value in index_summary.items()]
    return (
        "## 3. Transcript index\n\n"
        "Segments are grouped into chunks of roughly "
        f"{config.chunk_target_seconds:.0f} s, with {config.chunk_overlap_segments} "
        "segment of overlap between neighbours. Chunks never span a change of "
        "language, so a bilingual passage is never averaged into one vector.\n\n"
        + _table(["Property", "Value"], rows)
    )


def _section_rag(rag_results: list[dict], config: Config) -> str:
    body = (
        "## 4. Retrieval-augmented question answering\n\n"
        "For every question the retrieved evidence is shown before the answer, so "
        "retrieval and generation can be judged separately. A question whose best "
        f"match scores below {config.min_similarity:.2f} is refused without calling "
        "the language model; the model is also instructed to decline when the "
        "evidence it receives is not enough.\n"
    )

    for number, entry in enumerate(rag_results, start=1):
        result = entry["result"]
        body += (
            f"\n### Q{number} — {entry.get('category', 'question')}\n\n"
            f"**{result['question']}**\n\n"
            f"_{entry.get('expectation', '')}_\n\n"
            "Retrieved evidence:\n\n"
            + _table(
                ["rank", "chunk", "timespan", "language", "score", "text"],
                [
                    [r["rank"], r["chunk_id"], r["timespan"], r["language"], f"{r['score']:.4f}",
                     truncate(r["text"], 110)]
                    for r in result["retrieved"]
                ],
            )
            + f"\n**Answer** (answered: `{result['answered']}`"
        )
        if result.get("refusal_reason"):
            body += f", reason: `{result['refusal_reason']}`"
        body += f")\n\n> {result['answer']}\n\n"

        citations = result.get("citations") or []
        if citations:
            body += "Timestamp citations: " + ", ".join(
                f"`{c['text']}`" + ("" if c["matches_retrieved_chunks"] else " ⚠ not in retrieved evidence")
                for c in citations
            ) + "\n"
        elif result["answered"]:
            body += "_No timestamp citation was produced for this answer._\n"
    return body


def _section_agent(agent_run: AgentRun | None) -> str:
    if agent_run is None:
        return "## 5. Tool-using agent\n\n_(agent demonstration was not run)_\n"

    rows = []
    for step in agent_run.steps:
        if step.tool is None:
            # The step where the model stopped calling tools and answered.
            rows.append([
                step.step,
                "_(no tool)_",
                "—",
                "error" if step.error else "ok",
                step.error or "produced the final answer",
            ])
            continue
        rows.append(
            [
                step.step,
                step.tool,
                truncate(json.dumps(step.args, ensure_ascii=False), 60) if step.args else "—",
                "error" if step.error else "ok",
                truncate(step.error or step.observation or "", 90),
            ]
        )
    return (
        "## 5. Tool-using agent\n\n"
        f"**Task:** {agent_run.request}\n\n"
        f"The loop is bounded at {agent_run.max_steps} steps by the application, not "
        "by the model. Tool names and every argument are validated before execution; "
        "a rejected call becomes an observation the model can recover from.\n\n"
        + _table(["step", "tool", "arguments", "status", "observation / error"], rows)
        + f"\nTools used: `{'`, `'.join(agent_run.distinct_tools_used) or '—'}` "
        f"({agent_run.successful_tool_calls} successful of {agent_run.tool_calls} calls). "
        f"Stopped because: `{agent_run.stopped_because}`.\n\n"
        f"**Final answer**\n\n> {agent_run.final_answer}\n"
    )


def _section_multilingual(multilingual: dict | None) -> str:
    if not multilingual:
        return "## 6. Multimodal extension\n\n_(extension was not run)_\n"

    distribution = multilingual["language_distribution"]
    language_rows = [
        [language, stats["segments"], f"{stats['seconds']:.1f}", f"{stats['share'] * 100:.1f}%"]
        for language, stats in distribution["per_language"].items()
    ]
    timeline_rows = [
        [span["timespan"], span["language"], f"{span['duration']:.1f}"]
        for span in multilingual["language_timeline"]
    ]
    cross_rows = [
        [
            row["label"],
            f"`{row['evidence_language']}`",
            f"{', '.join(row['chinese_top_chunks'])} ({row['chinese_top_score']:.3f})",
            f"{', '.join(row['english_top_chunks'])} ({row['english_top_score']:.3f})",
            "yes" if row["same_top_hit"] else "no",
            row["overlap_at_k"],
        ]
        for row in multilingual["cross_lingual_comparison"]
    ]
    comparison = multilingual.get("translation_comparison", [])
    translation_rows = [
        [
            row["timespan"],
            truncate(row["original"], 70),
            truncate(row["whisper_translate"], 70) or "_(not run)_",
            truncate(row["llm_translate"], 70),
            f"{row['routes_agree']:.2f}",
        ]
        for row in comparison
    ]
    # Route A needs the audio. A run started from an existing transcript has
    # no audio to decode, so say that rather than leaving an empty column
    # that reads like a failure.
    route_a_note = ""
    if comparison and not any(row["whisper_translate"].strip() for row in comparison):
        route_a_note = (
            "\n> Route A is empty here: this run started from an existing "
            "transcript, so there was no audio for Whisper to decode a second "
            "time. Run from `--media` to populate it.\n"
        )

    return (
        "## 6. Multimodal extension — Option C, multilingual media mode\n\n"
        "### 6.1 Language detection\n\n"
        f"{distribution['language_count']} languages detected across "
        f"{distribution['switch_count']} switches.\n\n"
        + _table(["language", "segments", "seconds", "share"], language_rows)
        + "\n### 6.2 Language timeline\n\n"
        + _table(["timespan", "language", "seconds"], timeline_rows)
        + "\n### 6.3 Cross-lingual retrieval\n\n"
        "Each fact below is stated in only one language in the recording. Asking "
        "in the other language tests whether retrieval is semantic rather than "
        "lexical.\n\n"
        + _table(
            ["question", "evidence is in", "asked in Chinese", "asked in English",
             "same top hit", "overlap@k"],
            cross_rows,
        )
        + "\n### 6.4 Two translation routes compared\n\n"
        "The original transcript is preserved unchanged; translations are held in "
        "separate fields. Route A is Whisper decoding the audio directly into "
        "English; route B is the LLM translating the transcript text. The overlap "
        "column is a blunt word-overlap ratio, useful only for spotting rows worth "
        "reading.\n\n"
        + _table(
            ["timespan", "original", "Whisper translate", "LLM translate", "overlap"],
            translation_rows,
        )
        + route_a_note
        + f"\n### 6.5 Visual output\n\n![Multilingual analysis]({multilingual['figure_path']})\n"
    )


def _section_reliability(reliability: dict | None) -> str:
    if not reliability:
        return ""
    rows = [
        [case["name"], case["condition"], "pass" if case["passed"] else "FAIL", case["detail"]]
        for case in reliability.get("cases", [])
    ]
    return (
        "## 7. Reliability checks\n\n"
        + _table(["check", "condition", "result", "detail"], rows)
        + f"\n{reliability.get('passed', 0)} of {reliability.get('total', 0)} checks passed.\n"
    )


def _section_transcript(segments: list[dict]) -> str:
    lines = "\n".join(format_segment_line(segment) for segment in segments)
    return (
        "## Appendix — full timestamped transcript\n\n"
        f"```text\n{lines}\n```\n"
    )


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------

def build_report(
    *,
    segments: list[dict],
    media_metadata: dict,
    transcript_metadata: dict,
    index_summary: dict,
    analysis: dict,
    rag_results: list[dict],
    agent_run: AgentRun | None,
    multilingual: dict | None,
    config: Config,
    reliability: dict | None = None,
    generated_at: str | None = None,
) -> str:
    """Render the human-readable report for one run."""
    stamp = generated_at or datetime.now().strftime("%Y-%m-%d %H:%M")
    title = analysis.get("title") or media_metadata.get("filename") or "Recording"

    return "\n".join(
        [
            f"# Media analysis report — {title}",
            "",
            f"Generated {stamp} by the multimodal AI media assistant.",
            "",
            _section_overview(media_metadata, transcript_metadata, config),
            _section_analysis(analysis),
            _section_index(index_summary, config),
            _section_rag(rag_results, config),
            _section_agent(agent_run),
            _section_multilingual(multilingual),
            _section_reliability(reliability),
            _section_transcript(segments),
        ]
    )


def write_artifacts(
    assistant,
    *,
    output_dir: str = ".",
    rag_results: list[dict] | None = None,
    agent_run: AgentRun | None = None,
    multilingual: dict | None = None,
    reliability: dict | None = None,
) -> dict[str, str]:
    """Write every deliverable and return the paths written."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    transcript_path = directory / "transcript.json"
    transcript_path.write_text(
        json.dumps(assistant.transcript_payload(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    written["transcript"] = str(transcript_path)

    if assistant.analysis:
        analysis_path = directory / "analysis.json"
        analysis_path.write_text(
            json.dumps(assistant.analysis, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        written["analysis"] = str(analysis_path)

    # The report links to the figure, so the link has to be relative to the
    # report itself — an absolute path breaks as soon as the folder is zipped
    # and opened somewhere else.
    for_report = multilingual
    if multilingual and multilingual.get("figure_path"):
        figure = Path(multilingual["figure_path"])
        try:
            relative = figure.resolve().relative_to(directory.resolve())
        except ValueError:
            relative = Path(figure.name)
        for_report = {**multilingual, "figure_path": relative.as_posix()}

    report_path = directory / "report.md"
    report_path.write_text(
        build_report(
            segments=assistant.segments,
            media_metadata=assistant.media_metadata,
            transcript_metadata=assistant.transcript_metadata,
            index_summary=assistant.index.summary if assistant.index else {},
            analysis=assistant.analysis,
            rag_results=rag_results or [],
            agent_run=agent_run,
            multilingual=for_report,
            config=assistant.config,
            reliability=reliability,
        ),
        encoding="utf-8",
    )
    written["report"] = str(report_path)

    if multilingual and multilingual.get("figure_path"):
        written["visual"] = multilingual["figure_path"]
        bilingual = multilingual.get("bilingual_segments")
        if bilingual:
            bilingual_path = directory / "transcript_bilingual.json"
            bilingual_path.write_text(
                json.dumps(bilingual, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            written["bilingual_transcript"] = str(bilingual_path)

    return written
