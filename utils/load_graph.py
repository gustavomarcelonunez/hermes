import networkx as nx
import os

def load_graph(graph_name: str):
    """
    Carga uno de los 6 grafos desde /data y devuelve un grafo NetworkX.
    """
    filename = f"data/{graph_name}.graphml"

    if not os.path.exists(filename):
        raise FileNotFoundError(f"No se encontró el archivo {filename}")

    try:
        G = nx.read_graphml(filename)
    except Exception as e:
        raise RuntimeError(f"Error al cargar {filename}: {e}")

    return G