import numpy as np
import networkx as nx
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Cargar variables del .env
load_dotenv()

def get_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("OPENAI_API_KEY no está definida en el entorno ni en el .env")
    return OpenAI(api_key=api_key)

EMBED_MODEL = "text-embedding-3-small"


def embed_text(text: str):
    client = get_client()
    response = client.embeddings.create(
        input=text,
        model=EMBED_MODEL
    )
    return response.data[0].embedding


def build_graph_embeddings(G: nx.Graph, graph_name: str, save_dir="embeddings"):
    os.makedirs(save_dir, exist_ok=True)

    node_list = list(G.nodes())
    embeddings = []

    for node in node_list:
        label = G.nodes[node].get("label", str(node))
        emb = embed_text(label)
        embeddings.append(emb)

    embeddings = np.array(embeddings)

    np.save(f"{save_dir}/{graph_name}_embeddings.npy", embeddings)

    with open(f"{save_dir}/{graph_name}_index.json", "w") as f:
        json.dump(node_list, f)

    return embeddings, node_list


def load_graph_embeddings(graph_name: str, save_dir="embeddings"):
    emb_path = f"{save_dir}/{graph_name}_embeddings.npy"
    idx_path = f"{save_dir}/{graph_name}_index.json"

    if not os.path.exists(emb_path) or not os.path.exists(idx_path):
        return None, None

    embeddings = np.load(emb_path)
    with open(idx_path, "r") as f:
        node_list = json.load(f)

    return embeddings, node_list