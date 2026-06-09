from matplotlib import pyplot as plt
import networkx as nx

layers = [5, 8, 8, 5]

G = nx.Graph()

pos = {}
node_id = 0

for l, n_nodes in enumerate(layers):
    for i in range(n_nodes):
        pos[node_id] = (l, -i)
        node_id += 1

offset = 0
for l in range(len(layers)-1):
    n1 = layers[l]
    n2 = layers[l+1]

    for i in range(n1):
        for j in range(n2):
            G.add_edge(
                offset+i,
                offset+n1+j
            )

    offset += n1

nx.draw(
    G,
    pos,
    node_size=600,
    width=1,
    with_labels=False
)

plt.axis("off")
plt.show()