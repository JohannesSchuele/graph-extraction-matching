
import networkx as nx
import matplotlib.pyplot as plt


def plot_graph_matches2(pos,image_size = 128):

    G = nx.Graph()
    for i,p in enumerate(pos):
        print(i,p)
        G.add_node(i,pos=p)
        if i<len(pos)-1:
            G.add_edge(i,i+1)
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos=pos, node_size=50, edge_color='g', width=5, node_color='r')

    image_size = 128
    extent = 0, image_size, 0, image_size
    plt.show()
    plt.savefig('../data/cow_mesh/save.png', format="PNG")


if __name__ == "__main__":
    pos = [[40, 10], [12, 100], [80, 90], [120, 120], [25, 120], [70, 40]]
    plot_graph_matches2(pos, image_size=128)