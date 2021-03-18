import dgl
import torch
import torch.nn as nn
import torchvision
import numpy as np

from .grnn import GRNN
from .config import CONFIGURATION
from .utils import MLP
import ipdb

class NodeUpdate(nn.Module):
    def __init__(self, CONFIG):
        super(NodeUpdate, self).__init__()
        self.fc = MLP(CONFIG.G_N_L_S_U, CONFIG.G_N_A_U, CONFIG.G_N_B_U, CONFIG.G_N_BN_U, CONFIG.G_N_D_U)
        self.fc_lang = MLP(CONFIG.G_N_L_S2_U, CONFIG.G_N_A2_U, CONFIG.G_N_B2_U, CONFIG.G_N_BN2_U, CONFIG.G_N_D2_U)

    def forward(self, node):
        feat = torch.cat([node.data['n_f_original'], node.data['new_n_f']], dim=1)
        feat_lang = torch.cat([node.data['word2vec_original'], node.data['new_n_f_lang']], dim=1)
        n_feat = self.fc(feat)
        n_feat_lang = self.fc_lang(feat_lang)

        return {'new_n_f': n_feat, 'new_n_f_lang': n_feat_lang}

class Predictor(nn.Module):
    def __init__(self, CONFIG):
        super(Predictor, self).__init__()
        self.classifier = MLP(CONFIG.G_ER_L_S, CONFIG.G_ER_A, CONFIG.G_ER_B, CONFIG.G_ER_BN, CONFIG.G_ER_D)
        self.sigmoid = nn.Sigmoid()

    def forward(self, edge):
        feat = torch.cat([edge.dst['new_n_f'], edge.dst['new_n_f_lang'], edge.data['s_f'], edge.src['new_n_f_lang'], edge.src['new_n_f']], dim=1)
        pred = self.classifier(feat)

        return {'pred': pred}

class AGRNN(nn.Module):
    def __init__(self, feat_type='fc7', bias=True, bn=True, dropout=None, multi_attn=False, layer=1, diff_edge=True):
        super(AGRNN, self).__init__()
 
        self.multi_attn = multi_attn
        self.layer = layer
        self.diff_edge = diff_edge
        self.CONFIG1 = CONFIGURATION(feat_type=feat_type, layer=1, bias=bias, bn=bn, dropout=dropout, multi_attn=multi_attn)
    
        self.grnn1 = GRNN(self.CONFIG1, multi_attn=multi_attn, diff_edge=diff_edge)
        self.edge_readout = Predictor(self.CONFIG1)

    def _build_graph(self, node_num, roi_label, node_space, diff_edge):

        graph = dgl.DGLGraph()
        graph.add_nodes(node_num)

        edge_list, h_node_list, obj_node_list, readout_edge_list, readout_h_h_e_list, readout_h_o_e_list = self._collect_edge(node_num, roi_label, node_space, diff_edge)
        src, dst = tuple(zip(*edge_list))
        graph.add_edges(src, dst)   # make the graph bi-directional

        return graph, h_node_list, obj_node_list, readout_edge_list, readout_h_h_e_list, readout_h_o_e_list

    def _collect_edge(self, node_num, roi_label, node_space, diff_edge):

        # get human nodes && object nodes
        h_node_list = np.where(roi_label == 1)[0]   # !NOTE: the type of roi_label must be numpy.array
        obj_node_list = np.where(roi_label != 1)[0]

        edge_list = []
        h_h_e_list = []
        o_o_e_list = []
        h_o_e_list = []
        readout_edge_list = []
        readout_h_h_e_list = []
        readout_h_o_e_list = []
        # get all edge in the fully-connected graph
        for src in range(node_num):
            for dst in range(node_num):
                if src == dst:
                    continue
                else:
                    edge_list.append((src, dst))

        # get corresponding readout edge in the graph
        src_box_list = np.arange(roi_label.shape[0])
        for dst in h_node_list:
            # if dst == roi_label.shape[0]-1:
            #     continue
            # src_box_list = src_box_list[1:]
            # for src in src_box_list:
            #     readout_edge_list.append((src, dst))
            for src in src_box_list:
                if dst != src:
                    readout_edge_list.append((src, dst))

        # get corresponding readout h_h edges && h_o edges
        temp_h_node_list = h_node_list[:]
        for dst in h_node_list:
            if dst == h_node_list.shape[0]-1:
                continue
            temp_h_node_list = temp_h_node_list[1:]
            for src in temp_h_node_list:
                if src == dst: continue
                readout_h_h_e_list.append((src, dst))

        readout_h_o_e_list = [x for x in readout_edge_list if x not in readout_h_h_e_list]

        # add node space to match the batch graph
        h_node_list = (np.array(h_node_list)+node_space).tolist()
        obj_node_list = (np.array(obj_node_list)+node_space).tolist()

        readout_h_h_e_list = (np.array(readout_h_h_e_list)+node_space).tolist()
        readout_h_o_e_list = (np.array(readout_h_o_e_list)+node_space).tolist()   
        readout_edge_list = (np.array(readout_edge_list)+node_space).tolist()

        return edge_list, h_node_list, obj_node_list, readout_edge_list, readout_h_h_e_list, readout_h_o_e_list

    def forward(self, node_num=None, feat=None, spatial_feat=None, word2vec=None, roi_label=None, validation=False, choose_nodes=None, remove_nodes=None):
        # set up graph
        batch_graph, batch_h_node_list, batch_obj_node_list, batch_readout_edge_list, batch_readout_h_h_e_list, batch_readout_h_o_e_list = [], [], [], [], [], []
        node_num_cum = np.cumsum(node_num) # !IMPORTANT
        for i in range(len(node_num)):
            # set node space
            node_space = 0
            if i != 0:
                node_space = node_num_cum[i-1]
            graph, h_node_list, obj_node_list, readout_edge_list, readout_h_h_e_list, readout_h_o_e_list = self._build_graph(node_num[i], roi_label[i], node_space, diff_edge=self.diff_edge)
            # updata batch graph,
            batch_graph.append(graph)
            batch_h_node_list += h_node_list
            batch_obj_node_list += obj_node_list
            batch_readout_edge_list += readout_edge_list
            batch_readout_h_h_e_list += readout_h_h_e_list
            batch_readout_h_o_e_list += readout_h_o_e_list
        batch_graph = dgl.batch(batch_graph)

        # pass throuh gnn/gcn
        self.grnn1(batch_graph, batch_h_node_list, batch_obj_node_list, feat, spatial_feat, word2vec, validation, initial_feat=True)
        batch_graph.apply_edges(self.edge_readout, tuple(zip(*(batch_readout_h_o_e_list+batch_readout_h_h_e_list))))

        # import ipdb; ipdb.set_trace()
        if self.training or validation:
            # return batch_graph.edges[tuple(zip(*(batch_readout_h_o_e_list+batch_readout_h_h_e_list)))].data['pred']
            # !NOTE: cannot use "batch_readout_h_o_e_list+batch_readout_h_h_e_list" because of the wrong order
            return batch_graph.edges[tuple(zip(*batch_readout_edge_list))].data['pred']
        else:
            return batch_graph.edges[tuple(zip(*batch_readout_edge_list))].data['pred'], \
                   batch_graph.nodes[batch_h_node_list].data['alpha'], \
                   batch_graph.nodes[batch_h_node_list].data['alpha_lang'] 

if __name__ == "__main__":
    model = AGRNN()

    node_num = 3
    edge_list = []
    for src in range(node_num):
        for dst in range(node_num):
            edge_list.append((src,dst))
    src, dst = tuple(zip(*edge_list))
    g = dgl.DGLGraph()
    g.add_nodes(node_num)
    g.add_edges(src, dst)
    import ipdb; ipdb.set_trace()
    e_data = torch.eye(9)
    n_data = torch.arange(9)
    g.edata['feat'] = e_data
    g.ndata['x'] = n_data