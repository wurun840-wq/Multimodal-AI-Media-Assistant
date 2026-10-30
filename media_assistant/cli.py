"""Command-line entry point: one command produces every deliverable.

    python -m media_assistant.cli --media sample_media/meeting.wav

Or, to re-run the analysis stages on a transcript that already exists
(useful while developing, and to avoid paying for transcription twice):

    python -m media_assistant.cli --transcript transcript.json

Stages run in the order that keeps peak memory down: all Whisper passes,
then the index, then everything that needs the language model.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .config import Config
from .demo import DEFAULT_AGENT_TASK, DEFAULT_RAG_QUESTIONS, FAILING_AGENT_TASK
from .exceptions import MediaAssistantError
from .reliability import run_reliability_checks
from .report import write_artifacts
from .utils import LOGGER, configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="media_assistant",
        description="Analyse a recording end to end and write every deliverable.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--media", help="audio or video file to analyse")
    source.add_argument(
        "--transcript",
        help="an existing transcript.json to reuse instead of transcribing again",
    )
    parser.add_argument("--out", default=".", help="directory for the artifacts (default: .)")
    parser.add_argument(
        "--workdir", default=".cache_media", help="where converted audio is cached"
    )
    parser.add_argument(
        "--small", action="store_true",
        help="use the smaller fallback models (see Config.small)",
    )
    parser.add_argument("--top-k", type=int, default=None, help="chunks to retrieve per question")
    parser.add_argument(
        "--max-agent-steps", type=int, default=None, help="hard cap on agent tool calls"
    )
    parser.add_argument("--skip-analysis", action="store_true", help="skip Part 2")
    parser.add_argument("--skip-rag", action="store_true", help="skip the Part 3 question set")
    parser.add_argument("--skip-agent", action="store_true", help="skip the Part 4 demonstration")
    parser.add_argument("--skip-multilingual", action="store_true", help="skip the Part 5 extension")
    parser.add_argument(
        "--skip-translation", action="store_true",
        help="run Part 5 but without the per-segment translations (much faster)",
    )
    parser.add_argument("--skip-checks", action="store_true", help="skip the Part 6 checks")
    parser.add_argument("--quiet", action="store_true", help="only log warnings and errors")
    return parser


def _config_from_args(args: argparse.Namespace) -> Config:
    overrides = {}
    if args.top_k is not None:
        overrides["top_k"] = args.top_k
    if args.max_agent_steps is not None:
        overrides["max_agent_steps"] = args.max_agent_steps
    return Config.small(**overrides) if args.small else Config(**overrides)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(level=30 if args.quiet else 20)

    config = _config_from_args(args)
    from .assistant import MediaAssistant  # imported here to keep --help fast

    started = time.perf_counter()
    assistant = MediaAssistant(config)

    try:
        # -- transcript ------------------------------------------------
        if args.media:
            LOGGER.info("Processing %s", args.media)
            assistant.process(
                args.media,
                workdir=args.workdir,
                analyse=not args.skip_analysis,
            )
        else:
            LOGGER.info("Loading transcript from %s", args.transcript)
            assistant.load_transcript(args.transcript)
            if not args.skip_analysis:
                assistant.analyse()

        # -- Part 3: the five required questions ------------------------
        rag_results = []
        if not args.skip_rag:
            for question in DEFAULT_RAG_QUESTIONS:
                LOGGER.info("Asking (%s): %s", question.category, question.question)
                rag_results.append(
                    {
                        "category": question.category,
                        "expectation": question.expectation,
                        "result": assistant.ask(question.question),
                    }
                )

        # -- Part 4: the multi-tool demonstration -----------------------
        agent_run = None
        if not args.skip_agent:
            LOGGER.info("Running the agent demonstration")
            agent_run = assistant.agent(DEFAULT_AGENT_TASK)

        # -- Part 5: Option C ------------------------------------------
        multilingual = None
        if not args.skip_multilingual:
            LOGGER.info("Running the multilingual extension")
            multilingual = assistant.multilingual_analysis(
                figure_path=Path(args.out) / "visual_output.png",
                translate=not args.skip_translation,
            )

        # -- Part 6: reliability ---------------------------------------
        reliability = None
        if not args.skip_checks:
            LOGGER.info("Running reliability checks")
            unanswerable = next(
                q.question for q in DEFAULT_RAG_QUESTIONS if q.category == "unanswerable"
            )
            normal = next(q.question for q in DEFAULT_RAG_QUESTIONS if q.category == "factual")
            reliability = run_reliability_checks(
                assistant,
                unanswerable_question=unanswerable,
                normal_question=normal,
                failing_agent_request=FAILING_AGENT_TASK,
                include_model_checks=not args.skip_rag,
            )

        written = write_artifacts(
            assistant,
            output_dir=args.out,
            rag_results=rag_results,
            agent_run=agent_run,
            multilingual=multilingual,
            reliability=reliability,
        )
    except MediaAssistantError as exc:
        LOGGER.error("%s: %s", type(exc).__name__, exc)
        return 2
    except FileNotFoundError as exc:
        LOGGER.error("%s", exc)
        return 2
    finally:
        assistant.release_all()

    elapsed = time.perf_counter() - started
    print(f"\nDone in {elapsed / 60:.1f} min. Wrote:")
    for label, path in written.items():
        print(f"  {label:22s} {path}")
    if reliability:
        print(
            f"\nReliability: {reliability['passed']}/{reliability['total']} checks passed"
            + ("" if reliability["all_passed"] else "  ← see report.md section 7")
        )
    return 0 if (reliability is None or reliability["all_passed"]) else 1


if __name__ == "__main__":
    sys.exit(main())
