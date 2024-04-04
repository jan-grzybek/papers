import networkx as nx
import matplotlib.pyplot as plt

# generate connections from H1 to H2 so that each H1 layer feeds equal number of H2 layers

x = {i: 0 for i in range(12)}
links = {i: [] for i in range(12)}

for t in range(12):
    for i, k in enumerate(sorted(x.items(), key=lambda x: x[1])):
        if i == 8:
            break
        x[k[0]] += 1
        links[k[0]].append(t)

G = nx.DiGraph()

top_nodes = []
bottom_nodes = []
for h2_, h1 in links.items():
    h2_name = f"H2_{h2_ + 1}"
    top_nodes.append(h2_name)
    for h1_ in h1:
        h1_name = f"H1_{h1_ + 1}"
        if h1_name not in bottom_nodes:
            bottom_nodes.append(h1_name)
        if h2_ % 3 == 0:
            G.add_edge(h1_name, h2_name, color="r")
        elif h2_ % 3 == 1:
            G.add_edge(h1_name, h2_name, color="g")
        elif h2_ % 3 == 2:
            G.add_edge(h1_name, h2_name, color="b")
        print(f"lenet->H2_{h2_ + 1}.forward(&lenet->H2_{h2_ + 1}, &lenet->H1_{h1_ + 1}.output);")
    print()

edges = G.edges()
colors = [G[u][v]['color'] for u, v in edges]

options = {
    "font_size": 8,
    "node_size": 1000,
    "node_color": "white",
    "edgecolors": "black",
    "linewidths": 2,
    "width": 2,
    "edge_color": colors
}
pos = {n: (i * 5000, 0) for i, n in enumerate(bottom_nodes)}
pos.update({n: (i * 5000, 1) for i, n in enumerate(top_nodes)})
nx.draw_networkx(G, pos, **options)

# Set margins for the axes so that nodes aren't clipped
ax = plt.gca()
ax.margins(0.20)
plt.axis("off")
plt.show()
