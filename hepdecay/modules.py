import torch
import torch.nn as nn
import dgl
from dgl.nn import GATv2Conv

class DeepGraphConvLayer(nn.Module):
    def __init__(self, inputs, nodes, num_heads, features):
        super(DeepGraphConvLayer, self).__init__()

        gat_nodes = nodes // num_heads

        self.nodes = nodes
        self.num_heads = num_heads

        self.conv = GATv2Conv(inputs, 
                              gat_nodes, 
                              num_heads=num_heads,
                              residual=True,
                              share_weights=True,
                              allow_zero_in_degree=True)

        self.bn = nn.BatchNorm1d(nodes)
        self.relu = nn.ReLU()

    def forward(self, g):
        in_feat = g.ndata['features']

        batch_size = g.batch_size
        node_size = in_feat.size(0) // batch_size

        in_feat, attn = self.conv(g, in_feat,  get_attention=True)
        num_edges = g.num_edges() // batch_size
        attn = torch.squeeze(attn)
        attn = attn.reshape(batch_size, num_edges, self.num_heads)
        attn = attn.permute(0, 2, 1)

        in_feat = in_feat.reshape(batch_size*node_size, self.nodes)
        in_feat = self.bn(in_feat)
        in_feat = self.relu(in_feat)

        g.ndata['features'] = in_feat

        return g, attn

class DeepGraphConvModel(nn.Module):
    def __init__(self, inputs, layers, nodes, num_heads, features):
        super(DeepGraphConvModel, self).__init__()

        convs = [DeepGraphConvLayer(inputs, nodes, num_heads, features)]

        for ii in range(layers-1):
            convs.append(DeepGraphConvLayer(nodes, nodes, num_heads, features))

        self.convs = nn.Sequential(*convs)  

    def forward(self, g, attn=False):

        rtn_attn = []
        for conv in self.convs:
            g, attn = conv(g)
            rtn_attn.append(attn)

        return g, rtn_attn

class GraphModel(nn.Module):
    def __init__(self, features=None, edge_layers=2, graph_layers=4, nodes=16, num_heads=4):
        super(GraphModel, self).__init__()
      
        self.features_edge = DeepGraphConvModel(5, edge_layers, nodes, num_heads, features)
        self.fc_edge0 = nn.Linear(nodes*2, nodes)
        self.bn_edge =nn.BatchNorm1d(nodes)
        self.relu_edge = nn.ReLU()
        self.fc_edge1 = nn.Linear(nodes, 6)

        self.features_graph = DeepGraphConvModel(nodes, graph_layers, nodes, num_heads, features)
        self.fc_graph0 = nn.Linear(nodes, 2)

    @staticmethod
    def cat_node_features(edges):
        return {'features': torch.cat([edges.src['features'], edges.dst['features']], axis=1)}

    def forward(self, g):
        batch_size = g.batch_size
        node_size = g.ndata['features'].size(0) // batch_size
        edge_size = node_size * node_size

        g, attn_edge = self.features_edge(g)
        attn_edge = torch.stack(attn_edge, axis=1)

        g.apply_edges(self.cat_node_features)
        g_edge = g.edata['features']
        g_edge = self.fc_edge0(g_edge)
        g_edge = self.relu_edge(g_edge)
        g_edge = self.fc_edge1(g_edge)
        g_edge = g_edge.reshape(batch_size, edge_size, g_edge.size(-1))
              
        g, attn_graph = self.features_graph(g)
        attn_graph = torch.stack(attn_graph, axis=1)
        g_graph = dgl.mean_nodes(g, feat='features')    
        g_graph = self.fc_graph0(g_graph)

        return g_edge, g_graph, attn_edge, attn_graph


class MLPModel(nn.Module):
    def __init__(self, inputs=30, layers=4, nodes=16):
        super(MLPModel, self).__init__()

        convs = [
            nn.Linear(inputs, nodes),
            nn.BatchNorm1d(nodes),
            nn.ReLU(),
        ]

        for ii in range(layers-2):
            convs.append(nn.Linear(nodes, nodes))
            convs.append(nn.BatchNorm1d(nodes))
            convs.append(nn.ReLU())

        convs.append(nn.Linear(nodes, 2))
       
        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)

        return x
