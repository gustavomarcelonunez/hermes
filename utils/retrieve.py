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


def get_top_k_nodes(query: str, embeddings, node_list, k=10):
    query_emb = embed_query(query)

    sims = embeddings @ query_emb / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-8
    )

    idx = sims.argsort()[-k:][::-1]  # top K

    return [(node_list[i], float(sims[i])) for i in idx]


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