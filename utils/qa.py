import json
import networkx as nx
from utils.models import run_llm


def graph_to_triplets_text(G):
    lines = []
    for u, v, data in G.edges(data=True):
        rel = data.get("description", "related_to")
        lines.append(f"{u} --{rel}--> {v}")
    return "\n".join(lines)


def run_qa(question: str, subgraph: nx.Graph, model_key: str):
    context_str = graph_to_triplets_text(subgraph)

    system_prompt = """
You are a specialized Graph Reasoning Assistant.
You answer questions EXCLUSIVELY based on the provided knowledge graph.
You must strictly follow these rules:

1. Only use information explicitly contained in the graph triplets.
2. Do NOT use prior knowledge, assumptions, or external facts.
3. If the graph does not contain enough information to answer, reply:
   "No information available in the knowledge graph."
4. If the graph partially answers but lacks details, say exactly what is missing.
5. When useful, reason explicitly over nodes and relationships.
6. Prefer short, precise answers unless the question requires elaboration.
7. NEVER hallucinate entities or relationships not present in the graph.
8. Answer **in English**.
"""

    user_prompt = f"""
Question:
{question}

You are given a subgraph represented as semantic triplets (subject --relation--> object).

<GRAPH>
{context_str}
</GRAPH>

Instructions:
- Use only these triplets to answer.
- If reasoning is needed, walk through connections step-by-step.
- If multiple triplets are relevant, synthesize them coherently.
- If there is conflicting or ambiguous information, state it explicitly.
- If the answer cannot be inferred from the graph, say:
  "No information available in the knowledge graph."

Now produce your final answer.
"""

    return run_llm(model_key, system_prompt, user_prompt)


def verify_answer(answer: str, subgraph: nx.Graph, model_key: str) -> dict:
    """
    Verifies whether the claims in `answer` are supported by the subgraph triplets.

    Returns a dict with:
      - claims: list of {claim, status, evidence}
      - supported: int
      - partial: int
      - unsupported: int
      - total: int
      - gf_strict: float   (supported / total)
      - gf_weighted: float ((supported + 0.5 * partial) / total)
      - raw_json: str      (raw LLM output, for debugging)
    """
    triplets_str = graph_to_triplets_text(subgraph)

    system_prompt = """
You are a strict knowledge graph auditor.
Your only task is to verify whether claims in a given answer are supported
by a provided set of graph triplets.

Rules:
1. Do NOT use any external knowledge — only the triplets.
2. Be conservative: classify as SUPPORTED only when there is clear,
   direct evidence in the triplets.
3. Decompose the answer into individual, atomic factual claims.
4. For each claim, assign exactly one status:
   - SUPPORTED:   directly traceable to one or more triplets (cite them).
   - PARTIAL:     indirectly suggested by the triplets but not explicit.
   - UNSUPPORTED: no triplet evidence exists for this claim.
5. Respond ONLY with valid JSON — no preamble, no explanation outside the JSON.
"""

    user_prompt = f"""
Given these knowledge graph triplets:
<TRIPLETS>
{triplets_str}
</TRIPLETS>

And this answer to verify:
<ANSWER>
{answer}
</ANSWER>

Decompose the answer into individual factual claims and classify each one.

Respond ONLY in this exact JSON format:
{{
  "claims": [
    {{
      "claim": "text of the claim",
      "status": "SUPPORTED",
      "evidence": ["Subject --relation--> Object"]
    }},
    {{
      "claim": "text of the claim",
      "status": "PARTIAL",
      "evidence": ["Subject --relation--> Object"]
    }},
    {{
      "claim": "text of the claim",
      "status": "UNSUPPORTED",
      "evidence": []
    }}
  ]
}}

Important: grounding_fidelity will be computed externally — do NOT include it in your response.
"""

    raw = run_llm(model_key, system_prompt, user_prompt)

    return _parse_verification(raw)


def _parse_verification(raw: str) -> dict:
    """
    Parses the LLM JSON output and computes grounding fidelity metrics.
    Returns a safe dict even if parsing fails.
    """
    # Strip markdown code fences if present (some models wrap JSON in ```json ... ```)
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()

    try:
        data = json.loads(cleaned)
        claims = data.get("claims", [])

        supported   = sum(1 for c in claims if c.get("status") == "SUPPORTED")
        partial     = sum(1 for c in claims if c.get("status") == "PARTIAL")
        unsupported = sum(1 for c in claims if c.get("status") == "UNSUPPORTED")
        total       = len(claims)

        gf_strict   = supported / total if total > 0 else 0.0
        gf_weighted = (supported + 0.5 * partial) / total if total > 0 else 0.0

        return {
            "claims":      claims,
            "supported":   supported,
            "partial":     partial,
            "unsupported": unsupported,
            "total":       total,
            "gf_strict":   round(gf_strict, 4),
            "gf_weighted": round(gf_weighted, 4),
            "raw_json":    raw,
            "parse_error": None,
        }

    except (json.JSONDecodeError, KeyError) as e:
        return {
            "claims":      [],
            "supported":   0,
            "partial":     0,
            "unsupported": 0,
            "total":       0,
            "gf_strict":   0.0,
            "gf_weighted": 0.0,
            "raw_json":    raw,
            "parse_error": str(e),
        }