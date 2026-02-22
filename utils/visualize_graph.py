from pyvis.network import Network
import networkx as nx

def visualize_graph(G: nx.Graph, height="600px", width="100%"):
    """
    Devuelve HTML para incrustar la visualización interactiva de un grafo NetworkX.
    """

    net = Network(height=height, width=width, directed=False, notebook=False)
    net.barnes_hut()

    # Convertir NetworkX a PyVis
    for node, data in G.nodes(data=True):
        net.add_node(
            node,
            label=data.get("label", node),
            title=str(data),  # datos completos para popup
        )

    for source, target, data in G.edges(data=True):
        net.add_edge(
            source,
            target,
            title=str(data),
        )

    return net.generate_html()