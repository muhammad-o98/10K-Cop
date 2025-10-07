"""
Q&A with citations, derived insights, and visualization suggestions.

- Uses the hybrid retriever to fetch 10-K sections + analytics snippets.
- Prompts the LLM to produce STRICT JSON:
    {
      "answer": "concise CFO-grade answer...",
      "citations": [
        {"fy": "2023", "section": "7", "source": "2023.json", "snippet": "..." }
      ],
      "derived_metrics": [
        {"name": "operating_margin_delta", "value": -0.023, "formula": "FY23 op_margin - FY22 op_margin", "source": "ANALYTICS"}
      ],
      "viz_suggestions": [
        {"type": "waterfall", "title": "YoY Operating Income drivers", "data_refs": ["Revenues","COGS","OperatingExpenses"]}
      ]
    }

- Safe-parses the JSON and returns a Python dict with the above fields.
"""

from __future__ import annotations
import json
from typing import Dict, Any, List

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.output_parsers import OutputFixingParser
from langchain.schema import StrOutputParser

from .retriever import build_hybrid_retriever
from .llm_client import get_llm


SYSTEM_INSTRUCTIONS = """You are a CFO-grade financial analyst assistant.
You must:
- Answer concisely and only using the provided CONTEXT.
- Include citations that identify fiscal year (fy), section, and source filename.
- Interpret relevant ratios and analytics if present to justify conclusions.
- Propose at least one visualization that would help a board understand the answer.
- Output STRICT JSON with keys: answer, citations, derived_metrics, viz_suggestions. No extra text.

If the question cannot be answered from context, set answer to "Insufficient context." and return an empty citations list.
"""

JSON_SCHEMA_HINT = """Return a JSON object with this structure and no markdown:
{
  "answer": string,
  "citations": [ { "fy": string, "section": string, "source": string, "snippet": string } ],
  "derived_metrics": [ { "name": string, "value": number, "formula": string, "source": string } ],
  "viz_suggestions": [ { "type": string, "title": string, "data_refs": [string] } ]
}
"""

PROMPT = PromptTemplate.from_template(
    """{system}

Question: {question}

CONTEXT:
{context}

{schema}
"""
)


def _format_context(docs) -> str:
    """
    Compact context: prefix each chunk with metadata for useful citation.
    """
    lines: List[str] = []
    for d in docs:
        md = d.metadata or {}
        fy = md.get("fy", "")
        section = md.get("section", "")
        source = md.get("source", "")
        head = f"[fy={fy} section={section} source={source}]"
        text = d.page_content.strip().replace("\n", " ")
        if len(text) > 1200:
            text = text[:1200] + " ..."
        lines.append(f"{head} {text}")
    return "\n\n".join(lines)


def answer_with_citations(ticker: str, question: str, k: int = 6) -> Dict[str, Any]:
    retriever = build_hybrid_retriever(ticker=ticker, k=k)
    llm = get_llm()

    # Manually build retrieval + prompt so we can enforce strict JSON output
    # Retrieve
    docs = retriever.get_relevant_documents(question)

    # Prepare prompt
    ctx = _format_context(docs)
    prompt = PROMPT.format(
        system=SYSTEM_INSTRUCTIONS,
        question=question,
        context=ctx,
        schema=JSON_SCHEMA_HINT,
    )

    # Try strict output; if the model drifts, attempt to fix to JSON
    raw = llm.invoke(prompt)
    text = raw if isinstance(raw, str) else str(raw)

    # Fast path: direct JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # Attempt to extract the first JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        fragment = text[start : end + 1]
        try:
            return json.loads(fragment)
        except Exception:
            pass

    # Final fallback to a minimal structured response
    return {
        "answer": text.strip(),
        "citations": [
            {
                "fy": d.metadata.get("fy", ""),
                "section": d.metadata.get("section", ""),
                "source": d.metadata.get("source", ""),
                "snippet": d.page_content[:200],
            }
            for d in docs[:3]
        ],
        "derived_metrics": [],
        "viz_suggestions": [],
    }