import torch
import dgl
import copy

edges_src6 = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
              2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
              4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5]
edges_dst6 = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5,
              0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5,
              0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]


graph_dict = {
        6: dgl.graph((edges_src6, edges_dst6), num_nodes=6),
}


def get_dgl(data, target):
    num_batch = data.shape[0]
    num_nodes = data.shape[1]

    data = data.reshape(num_batch*num_nodes, 5)
    data = torch.tensor(data, dtype=torch.float32)

    graphs = [graph_dict[num_nodes] for ii in range(num_batch)]
    graphs = dgl.batch(graphs)

    graphs.ndata['features'] = data

    return graphs, target
