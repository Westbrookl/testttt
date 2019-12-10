import warnings

warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from numpy.random import random


def flip(p):
    return random() < p


def random_pairs(nodes, p):
    pair_list=list()
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if i < j and flip(p):
                pair_list.append((u,v))
    return pair_list


def make_random_graph(n, p):
    G = nx.Graph()
    nodes = range(n)
    G.add_nodes_from(nodes)
    edge_list = list(random_pairs(nodes, p))
    G.add_edges_from(edge_list)
    # print(nx.adjacency_matrix(G).todense())
    return G,1,1


def make_weighted_random_graph(n, p, weights,wmax):
    #print(str.format("weighted graph:{0} {1} {2} {3}",n,p,weights,wmax))
    G = nx.Graph()
    nodes = range(n)
    G.add_nodes_from(nodes)
    smax = wmax
    edge_list = list(random_pairs(nodes, p))
    while(len(edge_list)==0):
        edge_list = list(random_pairs(nodes, p))
    edge_weights= np.random.uniform(50, 100, len(edge_list))
    node_weights=np.random.uniform(50,100,n)
    if (weights == "normal"):
        edge_weights = np.random.uniform(50, 100, len(edge_list))
        smax=edge_weights.max()
    elif (weights=="mini"):
        edge_weights = np.random.uniform(15, 30, len(edge_list))
        node_weights = np.random.uniform(15, 30, n)
        smax=edge_weights.max()
    elif (weights == "mini_v"):
        edge_weights = np.random.uniform(0, 20, len(edge_list))
        node_weights = np.random.uniform(0, 20, n)
    elif (weights == "edge_v"):
        edge_weights = np.random.uniform(0, 50, len(edge_list))
        node_weights = np.random.uniform(0, 20, n)
    elif (weights == "node_v"):
        edge_weights = np.random.uniform(0, 20, len(edge_list))
        node_weights = np.random.uniform(0, 50, n)
    elif (weights == "intense"):
        edge_weights = np.random.uniform(0, 50, len(edge_list))
        node_weights = np.random.uniform(0, 50, n)
    elif (weights == "very_intense"):
        edge_weights = np.random.uniform(0, 100, len(edge_list))
        node_weights = np.random.uniform(0, 100, n)
    emin=edge_weights.min()
    emax=edge_weights.max()
    for i in range(len(edge_list)):
        G.add_edges_from([edge_list[i]], weight=edge_weights[i]/smax)
    #print(nx.adjacency_matrix(G).todense())
    return G,node_weights,emin,emax

def make_adj_matrix(G):
    return nx.to_numpy_matrix(G)


def make_laplacian_matrix(G):
    a = nx.normalized_laplacian_matrix(G)
    # print(a.todense())
    return a.todense()


def make_laplacian_list(graph_topology, node_size, orders):
    if graph_topology is None:
        print("Network topology is not initialized yet")
        return 0
    graph_laplacian_list = list()
    graph_laplacian_list.append(np.identity(node_size))
    lap = make_laplacian_matrix(graph_topology)
    base = make_laplacian_matrix(graph_topology)
    graph_laplacian_list.append(np.asarray(make_laplacian_matrix(graph_topology)))
    if orders > 2:
        for i in range(2, orders):
            lap = np.matmul(lap, base)
            graph_laplacian_list.append(np.asarray(lap))
    return graph_laplacian_list


def all_shortest_paths(G):
    return list(nx.all_pairs_shortest_path(G))


def simple_paths(G, source, target, cutoff):
    return list(nx.all_simple_paths(G, source, target, cutoff))


# random_graph = make_random_graph(100, 0.5)
# # print(all_shortest_paths(random_graph)[3][1])
# # print(simple_paths(random_graph,0,2,3))
# nx.draw_circular(random_graph,
#                  node_color=COLORS[3],
#                  node_size=1000,
#                  with_labels=True)
#
# # plt.show()
# matrix = nx.to_numpy_matrix(random_graph)
