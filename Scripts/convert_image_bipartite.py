import igraph as ig
import xnetwork as xn

inputNetwork = "bip_g_user_imgs.gaza.20250204.01.graphml"
outputNetwork = "imagesv4.edges"

g = ig.Graph.Read_GraphML(inputNetwork)

types = g.vs["type"]
# list of edges in order: "img", "user", weight based on g.vs["type"]
edges = []
for e in g.es:
    weight = e["weight"]
    if types[e.source] == "img":
        edges.append((g.vs[e.source]["id"], g.vs[e.target]["id"], weight))
    else:
        edges.append((g.vs[e.target]["id"], g.vs[e.source]["id"], weight))
    

with open(outputNetwork, "w") as f:
    for e in edges:
        f.write(f"{e[0]} {e[1]} {e[2]}\n")
print("Done")
