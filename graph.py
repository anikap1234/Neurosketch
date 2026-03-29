import json

def build_graph(nodes):
    # sort top-to-bottom, then left-to-right
    nodes = sorted(nodes, key=lambda x: (x["center"][1], x["center"][0]))

    edges = []

    for i in range(len(nodes) - 1):
        edges.append((nodes[i]["id"], nodes[i+1]["id"]))

    graph = {
        "nodes": nodes,
        "edges": edges
    }

    with open("debug/stage3_graph.json", "w") as f:
        json.dump(graph, f, indent=2)

    print("\nGRAPH:")
    print(graph)

    return graph