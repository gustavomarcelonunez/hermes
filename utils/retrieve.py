import numpy as np
from openai import OpenAI
import networkx as nx
from dotenv import load_dotenv
import os

load_dotenv()

def get_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("OPENAI_API_KEY no está definida")
    return OpenAI(api_key=api_key)

EMBED_MODEL = "text-embedding-3-small"

def embed_query(query: str):
    client = get_client()
    response = client.embeddings.create(
        input=query,
        model=EMBED_MODEL
    )
    return np.array(response.data[0].embedding)

def choose_hops(question: str):
    if len(question) < 80:
        return 1
    elif len(question) < 200:
        return 2
    else:
        return 3

def get_top_k_nodes(query, embeddings, node_list, min_k=10, sim_threshold=0.55):
    query_emb = embed_query(query)

    sims = embeddings @ query_emb / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-8
    )

    # ordenamos
    idx_sorted = np.argsort(sims)[::-1]

    selected = []
    for i in idx_sorted:
        if sims[i] < sim_threshold and len(selected) >= min_k:
            break
        selected.append((node_list[i], float(sims[i])))

    return selected


def extract_subgraph(G: nx.Graph, nodes, hops=1):
    """
    Extrae un subgrafo que incluye nodos relevantes + sus vecinos.
    """
    expanded = set(nodes)

    for _ in range(hops):
        neighbors = set()
        for n in expanded:
            neighbors.update(G.neighbors(n))
        expanded.update(neighbors)

    return G.subgraph(expanded).copy()