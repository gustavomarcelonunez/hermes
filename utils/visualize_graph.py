from pyvis.network import Network
import networkx as nx

def visualize_graph(G: nx.Graph, height="600px", width="100%"):
    net = Network(height=height, width=width, directed=False, notebook=False)
    net.barnes_hut()

    for node, data in G.nodes(data=True):
        node_type = data.get("type", "").strip()
        desc = data.get("description", "")

        type_str = "No type" if not node_type or node_type == "NONE" else node_type
        desc_str = "No description" if not desc else desc

        words = desc_str.split()
        lines = []
        current = ""
        for word in words:
            if len(current) + len(word) + 1 > 100:
                lines.append(current)
                current = word
            else:
                current = f"{current} {word}".strip()
        if current:
            lines.append(current)
        desc_wrapped = "\n".join(lines)

        title = f"Type: {type_str}\nDescription: {desc_wrapped}"
        net.add_node(node, label=data.get("label", node), title=title)

    for source, target, data in G.edges(data=True):
        desc = data.get("description", "")[:200]
        weight = data.get("weight", "")
        title = f"Relation: \"{desc}\"\nWeight: {weight}"
        net.add_edge(source, target, title=title)

    return net.generate_html()