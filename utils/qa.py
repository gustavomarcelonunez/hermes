from openai import OpenAI
import networkx as nx
import os
from dotenv import load_dotenv


load_dotenv()

def get_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("OPENAI_API_KEY no está definida")
    return OpenAI(api_key=api_key)

MODEL = "gpt-4o-mini"


def subgraph_to_text(G: nx.Graph):
    triples = []
    for u, v, data in G.edges(data=True):
        rel = data.get("label", "related_to")
        u_label = G.nodes[u].get("label", u)
        v_label = G.nodes[v].get("label", v)
        triples.append(f"{u_label} --[{rel}]--> {v_label}")
    return "\n".join(triples[:100])  # limitar por seguridad


def run_qa(question: str, subgraph: nx.Graph):
    client = get_client()
    context_str = subgraph_to_text(subgraph)

    system_prompt = """
You are an expert assistant answering questions using ONLY the knowledge 
contained in the provided graph context. Do not hallucinate.
If the answer is not in the graph, say 'No information available in the knowledge graph.'
Answer always in english.
"""

    user_prompt = f"""
Question: {question}

Here is the relevant part of the knowledge graph (node relations):

{context_str}

Provide a concise answer using ONLY this information.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1
    )

    return response.choices[0].message.content