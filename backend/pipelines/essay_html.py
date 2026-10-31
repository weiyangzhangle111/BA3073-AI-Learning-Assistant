from __future__ import annotations

import json
from typing import Any, Callable

from backend.artifacts.filesystem import ArtifactRun
from backend.pipelines.authorship import add_author_credit
from backend.pipelines.common import (
    PipelineError,
    PipelineResult,
    extract_fenced_or_raw,
    format_citations,
    format_log,
)
from backend.pipelines.html_to_pdf import (
    A4_PAGE,
    ConvertError,
    HtmlToPdfConverter,
)
from backend.providers.base import (
    ModelProviderError,
    TextGenerationProvider,
    TextGenerationRequest,
)
from backend.timing import RunTimingRecorder, measure_stage


def run_essay_html_pipeline(
    *,
    artifact_run: ArtifactRun,
    model_profile: dict[str, Any],
    model_provider: TextGenerationProvider,
    pdf_converter: HtmlToPdfConverter,
    task_text: str,
    context_bundle: str,
    output_preference: str | None,
    options: dict[str, Any] | None,
    search: dict[str, Any],
    max_output_tokens: int,
    emit_event: Callable[[str, str], None] | None = None,
    timing: RunTimingRecorder | None = None,
) -> PipelineResult:
    log_lines = [
        "Pipeline: essay_html",
        f"Output preference: {output_preference or 'pdf'}",
        "Stage: generate_source",
    ]
    system_prompt, user_prompt = build_essay_html_prompt(
        task_text=task_text,
        context_bundle=context_bundle,
        options=options or {},
        search=search,
    )
    if emit_event:
        emit_event("generate_source", "Generating essay HTML source")

    try:
        with measure_stage(timing, "provider_generation"):
            raw_output = model_provider.generate_text(
                TextGenerationRequest(
                    profile=model_profile,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_output_tokens=max_output_tokens,
                    temperature=0.2,
                )
            )
    except ModelProviderError as exc:
        raise PipelineError(
            exc.code,
            exc.message,
            stage="generate_source",
            log_lines=log_lines,
        ) from exc
    except Exception as exc:
        raise PipelineError(
            "provider_unavailable",
            "The model provider request failed.",
            stage="generate_source",
            log_lines=log_lines,
        ) from exc

    source = extract_fenced_or_raw(raw_output, accepted_languages={"html", "htm"})
    source = add_author_credit(source.strip()) + "\n"

    # Apply consistent print styling after AI HTML generation.
    # Keep existing author attribution and document content.
    fixed_style = """<style id="ba3073-fixed-print-style">
    @page {
        size: A4;
        margin: 25.4mm;
        @bottom-center {
            content: counter(page);
            font-size: 10pt;
        }
    }

    body {
        font-family: "Times New Roman", "Liberation Serif", serif !important;
        font-size: 12pt !important;
        line-height: 1.5 !important;
        color: #111 !important;
        background: white !important;
    }

    h1 {
        font-family: inherit !important;
        font-size: 16pt !important;
        font-weight: bold !important;
    }

    h2 {
        font-family: inherit !important;
        font-size: 14pt !important;
        font-weight: bold !important;
    }

    h3 {
        font-family: inherit !important;
        font-size: 12pt !important;
        font-weight: bold !important;
    }

    h1, h2, h3 {
        break-after: avoid !important;
    }

    p, li {
        font-family: inherit !important;
        font-size: 12pt !important;
        line-height: 1.5 !important;
    }

    table {
        width: 100% !important;
        border-collapse: collapse !important;
    }

    th, td {
        font-family: inherit !important;
        font-size: 11pt !important;
        text-align: left !important;
        vertical-align: top !important;
        overflow-wrap: anywhere !important;
    }

    code, pre {
        font-family: "Liberation Mono", monospace !important;
        font-size: 10pt !important;
        white-space: pre-wrap !important;
        overflow-wrap: anywhere !important;
    }

    .caption, figcaption {
        font-family: inherit !important;
        font-size: 10pt !important;
        text-align: center !important;
    }

    .reference-entry, .reference-pass {
        font-family: inherit !important;
        font-size: 11pt !important;
        text-align: left !important;
        text-indent: -18pt !important;
        margin-left: 18pt !important;
        overflow-wrap: anywhere !important;
    }

    tr, figure, pre, .figure-placeholder {
        break-inside: avoid;
    }
    </style>"""

    import re
    source, replacements = re.subn(
        r"</head\s*>",
        lambda m: fixed_style + "\\n" + m.group(0),
        source,
        count=1,
        flags=re.IGNORECASE,
    )
    if replacements != 1:
        raise PipelineError(
            "invalid_html",
            "Generated HTML is missing a head closing tag.",
            stage="generate_source",
            log_lines=log_lines,
        )

    html_path = artifact_run.write_output(
        "main.html",
        source,
        kind="source",
        media_type="text/html",
    )
    log_lines.extend(["Output: output/main.html", "Stage: convert_pdf"])
    if emit_event:
        emit_event("convert_pdf", "Converting essay HTML to PDF")

    pdf_path = artifact_run.run_dir / "output" / "main.pdf"
    try:
        with measure_stage(timing, "convert_pdf"):
            convert_result = pdf_converter.convert(
                html_path=html_path,
                pdf_path=pdf_path,
                page_config=A4_PAGE,
            )
    except ConvertError as exc:
        artifact_run.write_log("convert.log", exc.log_text)
        raise PipelineError(
            "convert_failed",
            exc.message,
            stage="convert_pdf",
            log_lines=log_lines,
        ) from exc
    except Exception as exc:
        artifact_run.write_log(
            "convert.log",
            "HTML-to-PDF converter raised an unexpected error.\n",
        )
        raise PipelineError(
            "convert_failed",
            "HTML-to-PDF conversion failed.",
            stage="convert_pdf",
            log_lines=log_lines,
        ) from exc

    artifact_run.write_log("convert.log", convert_result.log_text)
    artifact_run.write_output(
        "main.pdf",
        convert_result.pdf_path.read_bytes(),
        kind="pdf",
        media_type="application/pdf",
    )
    log_lines.extend(["Convert: pdf_ok", "Output: output/main.pdf"])
    return PipelineResult(log_text=format_log(log_lines))


def build_essay_html_prompt(
    *,
    task_text: str,
    context_bundle: str,
    options: dict[str, Any],
    search: dict[str, Any],
) -> tuple[str, str]:
    system_prompt = (
        "You are an academic technical-report assistant specializing in "
        "Mobile and Cloud Computing, particularly BA3073: From AI Agent "
        "to Cloud Service. Write accurate, evidence-based reports using "
        "formal, clear, and readable academic English. "

        "CONTENT AND STRUCTURE: "
        "Follow the user's requested scope. For a short test report, "
        "cover only the requested topics and avoid unnecessary chapters. "
        "For the full BA3073 assignment report, organize the discussion "
        "around Introduction, System Architecture, Local Docker "
        "Deployment, Prompt Engineering Improvements, Alibaba Cloud "
        "ACR and ECS Deployment, Domain and HTTPS Configuration, "
        "Access Control, Testing and Evidence, Security and Cost "
        "Considerations, Limitations, Future Improvements, and "
        "Conclusion. Combine related sections where appropriate. "
        "Explain what was done, why it was necessary, and how it was "
        "verified. Avoid repetitive introductions and generic discussion. "

        "TECHNICAL KNOWLEDGE: "
        "Connect actual implementation decisions to relevant concepts "
        "from mobile and cloud computing, including IaaS, virtualization, "
        "containers, resource sharing, networking, DNS, firewalls, "
        "persistent storage, security responsibilities, and cloud costs. "
        "Use technical terminology accurately. Explain concepts in "
        "relation to the deployed system rather than adding unrelated "
        "advanced technologies. "

        "EVIDENCE AND ACCURACY: "
        "Use only facts explicitly supported by the user's task, "
        "provided context, source files, and verification results. "
        "Do not invent or assume completed procedures, commands, "
        "configuration values, test results, screenshots, IP addresses, "
        "domains, resource limits, security mechanisms, or costs. "
        "Do not claim that model parameters, authentication, deployment "
        "settings, or application code were modified without evidence. "
        "Distinguish VERIFIED, IMPLEMENTED BUT UNVERIFIED, PLANNED, "
        "and NOT AVAILABLE where relevant. "
        "When evidence is missing, identify the gap concisely. "
        "Do not turn missing evidence into fictional implementation. "

        "REPORT QUALITY: "
        "Prioritize substantive analysis over excessive length. "
        "Use concise paragraphs, numbered headings, relevant tables, "
        "and meaningful figure captions. "
        "For a complete assignment report, remain within 20 A4 pages "
        "including references and screenshots. "
        "For short reports, follow the requested length instead of "
        "generating the full assignment structure. "
        "Use screenshots only when actual image evidence is available; "
        "otherwise identify where the user should insert them. "
        "Never fabricate citations or bibliographic information. "
        "Include references only when they are relevant and verifiable. "

        "TECHNICAL DEPTH AND SPECIFICITY: "
        "Explain the actual application architecture when relevant: "
        "browser interface, FastAPI backend, SQLite persistence, "
        "external Qwen model API, Docker image, and container runtime. "
        "Distinguish services running inside the container from "
        "external cloud API services. "
        "For each implementation stage, describe the objective, "
        "configuration, executed procedure, observed result, "
        "and technical significance. "
        "Include actual commands, file names, ports, software "
        "versions, and verification outputs when supported "
        "by the supplied evidence. "
        "Do not omit verified technical details merely to "
        "make the report shorter. "
        "Do not treat unverified details as completed work. "

        "EVIDENCE PRESENTATION: "
        "Connect each significant technical claim to its "
        "corresponding evidence, such as terminal output, "
        "container status, HTTP response, configuration file, "
        "or browser screenshot. "
        "Summarize verification results in concise tables "
        "when useful. "
        "Distinguish successful HTTP response checks from "
        "browser rendering and end-to-end feature testing. "
        "Avoid generic statements such as tests passed "
        "without describing what was actually tested. "
        "Do not replace concrete evidence with vague "
        "assurances about academic integrity. "

        "COURSE-BASED ANALYSIS: "
        "Apply relevant BA3073 concepts to explain decisions. "
        "For example, distinguish virtual machines from "
        "OS-level containers, discuss IaaS responsibilities "
        "for ECS, explain port mapping and firewall boundaries, "
        "and analyze storage persistence and cloud costs. "
        "Use only concepts relevant to the actual task. "
        "Prioritize accurate application of course knowledge "
        "over listing advanced technologies. "

        "DEPTH AND LENGTH CONTROL: "
        "A short deployment test report should remain focused "
        "but include enough substantive detail to explain "
        "the architecture, commands, verification, and limitations. "
        "Follow a requested approximate page count while "
        "prioritizing completeness of essential evidence. "
        "For the final assignment report, provide sufficient "
        "technical analysis and real deployment evidence "
        "across all three tasks. "
        "A reasonable planning range is 14 to 18 A4 pages "
        "including screenshots and references, subject to "
        "the mandatory maximum of 20 pages. "
        "Never add filler merely to reach a page target. "

        "SECURITY AND ACADEMIC INTEGRITY: "
        "Never expose passwords, API keys, tokens, or private secrets. "
        "Clearly distinguish AI-assisted writing from operations "
        "performed and verified by the student. "
        "Return only complete self-contained HTML source without "
        "Markdown fences or surrounding explanations."
    )
    output_instruction = (
        "Return one complete, valid, self-contained HTML5 document "
        "starting with <!doctype html> and containing html, head, "
        "and body elements. Include all CSS inside a style element. "
        "Do not use external stylesheets, fonts, scripts, CDNs, "
        "or remote images. "

        "ACADEMIC DOCUMENT FORMATTING: "
        "Use A4 portrait pages with CSS @page { size: A4; "
        "margin: 25.4mm; }. "
        "Use Times New Roman as the preferred body font, with "
        "Times and Liberation Serif as fallback fonts. "
        "Set body font-size to 12pt, line-height to 1.5, "
        "and text color to #111111 on a white background. "
        "Use consistent paragraph spacing of approximately "
        "6pt after each paragraph. "
        "Maintain readable, professional academic formatting. "

        "HEADINGS AND HIERARCHY: "
        "Use numbered main sections where appropriate. "
        "Set h1 section headings to 14pt bold and h2 "
        "subheadings to 12pt bold. "
        "Use consistent heading spacing. "
        "Avoid excessive bold text and decorative colors. "
        "Do not insert unnecessary blank pages. "

        "TITLE PAGE AND PAGE NUMBERS: "
        "For a complete assignment report, provide a simple "
        "academic cover page with the report title, course "
        "code BA3073, course name Mobile and Cloud Computing, "
        "and placeholders for student details not provided. "
        "For short tests, omit the cover page unless requested. "
        "Do not create HTML page-number elements, fixed footers, "
        "or manually positioned page numbers. "
        "Use CSS paged-media margin boxes for automatic numbering: "
        "@page { size: A4; margin: 25.4mm; "
        "@bottom-center { content: counter(page); "
        "font-family: Times New Roman, Times, serif; "
        "font-size: 10pt; } }. "
        "Do not add separate Page labels in the document body. "
        "Do not invent student identifiers. "

        "FIGURES, TABLES, AND CODE: "
        "Use real evidence screenshots only when supplied. "
        "If necessary, add clearly labeled figure placeholders "
        "that the student can replace later. "
        "Use figure captions in 10pt or 11pt font. "
        "Use tables only when they improve clarity. "
        "Set table headers in bold and maintain readable cell "
        "padding and borders. "
        "Use a monospace font for commands and terminal output. "
        "Allow code lines to wrap rather than overflow the page. "
        "Keep figures, captions, and tables together where possible. "

        "PRINT AND PAGINATION: "
        "Use CSS break-inside: avoid for important figures, "
        "tables, and code blocks when practical. "
        "Avoid orphaned headings and isolated captions. "
        "Do not force a new page for every heading. "
        "Use compact but readable spacing to reduce empty pages. "
        "Ensure content remains within printable margins. "

        "REFERENCES: "
        "Use APA 7 style consistently when references are included. "
        "Apply hanging indentation to reference entries. "
        "Do not fabricate bibliographic details, authors, "
        "publication dates, DOIs, or URLs. "
        "Use citations only for sources actually supplied "
        "or reliably identified in the provided context. "

        "OUTPUT CONSTRAINTS: "
        "Use inline SVG or CSS only for diagrams when needed. "
        "Do not depend on external resources at render time. "
        "Keep a complete assignment report within 20 pages, "
        "including the cover, figures, and references. "
        "The final PDF must be reviewed for actual page count "
        "and rendering quality. "
        "Return HTML source only, without Markdown fences."
    )
    user_prompt = "\n\n".join(
        [
            "[Assignment Task]\n" + task_text.strip(),
            "[Prepared Context]\n" + context_bundle.strip(),
            "[Search Citations]\n" + format_citations(search),
            "[Options]\n" + json.dumps(options, sort_keys=True),
            "[Output Contract]\n" + output_instruction,
        ]
    )
    return system_prompt, user_prompt
