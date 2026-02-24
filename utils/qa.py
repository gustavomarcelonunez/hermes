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