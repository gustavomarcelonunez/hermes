import networkx as nx

def compute_metrics(G: nx.Graph):
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    
    density = nx.density(G)

    degrees = [d for _, d in G.degree()]
    avg_degree = sum(degrees) / num_nodes if num_nodes > 0 else 0

    components = nx.number_connected_components(G)

    try:
        # Diametro solo tiene sentido en la componente gigante
        largest_cc = max(nx.connected_components(G), key=len)
        subgraph_lcc = G.subgraph(largest_cc)
        diameter = nx.diameter(subgraph_lcc)
    except Exception:
        diameter = None

    return {
        "nodes": num_nodes,
        "edges": num_edges,
        "density": density,
        "avg_degree": avg_degree,
        "components": components,
        "diameter": diameter,
    }