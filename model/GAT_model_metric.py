import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GATConv
from torch.nn import init
from dgl.utils import expand_as_pair
import dgl.function as fn
from dgl.ops import edge_softmax, segment
from dgl.base import DGLError
from dgl.readout import sum_nodes
from torch import Tensor
from typing import Optional, Tuple
from einops import rearrange
from torch.nn.modules.normalization import LayerNorm
from dgl import backend as Fu
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class gasa_classifier(torch.nn.Module):
    """
    GASA model has four attention layer, the first layer used to concat atom and bond features. 
    """

    def __init__(self, dropout, num_heads, hidden_dim1, hidden_dim3, hidden_dim2, n_tasks=11, in_dim=42):
        super(gasa_classifier, self).__init__()
        self.motif_trans = Motif_trans()
        # self.gat_trans_atten = Attention_weight_trans(in_dim, hidden_dim3 * num_heads, num_heads, edge_feat_size=6, dim=hidden_dim1)
        self.gnn = global_attention(in_dim, hidden_dim3 * num_heads, num_heads, edge_feat_size=6, dim=hidden_dim1)
        # self.gat_net1 = gat_net(hidden_dim1, hidden_dim3 * num_heads, num_heads=8, edge_feat_size=6, dim=hidden_dim1)
        # self.gat_net2 = gat_net_layer34(hidden_dim2, hidden_dim3 * num_heads, num_heads=8, edge_feat_size=6, dim=hidden_dim1)
        # self.gat_net3 = gat_net_layer34(hidden_dim2, hidden_dim3 * num_heads, num_heads=8, edge_feat_size=6, dim=hidden_dim1)
        self.gat1 = GATConv(hidden_dim1, hidden_dim2, num_heads,
                            negative_slope=0.2, bias=True)
        self.gat2 = GATConv(hidden_dim2 * num_heads, hidden_dim2, 1,
                            negative_slope=0.2, bias=True)
        self.gat3 = GATConv(hidden_dim2, hidden_dim2, 1,
                            negative_slope=0.2, bias=True)
        # self.gat4 = GATConv(hidden_dim2, hidden_dim2, 1,
        #                     negative_slope=0.2, bias=True)
        # self.gat5 = GATConv(hidden_dim2, hidden_dim2, 1,
        #                     negative_slope=0.2, bias=True)
        self.readout = WeightedSumAndMax(hidden_dim2)
        # self.readout = WeightedSumAndMax(hidden_dim2 + hidden_dim2 * num_heads)
        # self.trans_motif = TransformerEncoder(800,20,20,20,20,[6,20],20,20,4,2,0.5)
        # self.reduce_dim = nn.Linear(hidden_dim2 * 2, hidden_dim3)
        self.reduce_dim = nn.Linear(hidden_dim2, hidden_dim3)
        # self.reduce_dim = nn.Linear(hidden_dim2 * 4, hidden_dim3)
        # self.reduce_dim = nn.Linear((hidden_dim2+hidden_dim2 * num_heads) * 2, hidden_dim3)
        self.predict = nn.Linear(hidden_dim3 + 24, n_tasks)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, motif_vocab, maccs, feature_3d, get_node_weight=True):
        """Update node and edge representations.
        Parameters
        g: DGLGraph
           DGLGraph for a batch of graphs
        feats: FloatTensor of shape (N1, M1)
            * N1 is the total number of nodes in the batch of graphs
            * M1 is the input node feature size, which equals in_feats in initialization
        return
        hg : FloatTensor of shape (N2, M2)
            * N2 is the number of graph in the batch
            * M2 is the probilities of graph belong to each category.
        node_weights: FloatTensor
            atom weights learned by the model
        """
        edge_feats = g.edata['he']
        # h = g.in_degrees().view(-1, 1).float()
        h = g.ndata['hv']
        # h = self.gat_trans_atten(g, h, edge_feats)
        h = self.gnn(g, h, edge_feats)
        h = torch.flatten(self.dropout(F.elu(self.gat1(g, h))), 1)
        h = torch.flatten(self.dropout(F.elu(self.gat2(g, h))), 1)
        # h = torch.mean(self.dropout(F.elu(self.gat2(g, h))), 1)
        # h1 = h
        h = torch.mean(self.dropout(F.elu(self.gat3(g, h))), 1)
        # h = torch.flatten(self.dropout(F.elu(self.gat3(g, h))), 1)
        # h = torch.flatten(self.dropout(F.elu(self.gat4(g, h))), 1)
        # h = torch.mean(self.dropout(F.elu(self.gat5(g, h))), 1)
        # h = torch.cat((h1, h), 1)
        # h = self.gat_net1(g, h)
        # h = self.gat_net2(g, h)
        # h = self.gat_net3(g, h)
        g_feats, node_weights = self.readout(g, h, get_node_weight)
        reduce_g_feats = self.reduce_dim(g_feats)
        trans_motif_embed = self.motif_trans(motif_vocab)
        # print(trans_motif_embed.shape)
        # print(reduce_g_feats.shape)
        # print(maccs[:, [0, 1, 2, 3, 4, 5, 6]].shape)
        # print(feature_3d.shape)
        # exit()
        mixed_enbedding = torch.cat((reduce_g_feats, trans_motif_embed), 1)
        mixed_enbedding = torch.cat((mixed_enbedding, maccs[:, [0, 1, 2, 3, 4, 5, 6]]), 1)
        mixed_enbedding = torch.cat((mixed_enbedding, feature_3d), 1)
        # print(mixed_enbedding.shape)
        # # exit()
        hg = self.predict(mixed_enbedding)

        return hg, node_weights, mixed_enbedding


class global_attention(nn.Module):
    """
    The first layer of GASA model which is used to concat atom and bond features. 
    """

    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 dim,
                 edge_feat_size,
                 negative_slope=0.2,
                 feat_drop=0,
                 residual=False,
                 allow_zero_in_degree=False,
                 bias=True):
        super(global_attention, self).__init__()
        self._num_heads = num_heads
        self.dim = dim
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.edge_feat_size = edge_feat_size
        self.full = nn.Linear(out_feats * num_heads, out_feats)
        self.linears1 = nn.Linear(out_feats + edge_feat_size, out_feats)
        self.linears2 = nn.Linear(out_feats * 2, dim)
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)

        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, edge_feat):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')
                    # print('There are 0-in-degree nodes in the graph')
            # 非异构图
            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    -1, self._num_heads, self._out_feats)
                # exit()
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            graph.edata['a'] = edge_softmax(graph, e)
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            new = torch.flatten(F.leaky_relu(rst), 1)
            ft = self.full(new)
            graph.ndata['h'] = ft
            graph.edata['he'] = edge_feat
            graph.apply_edges(lambda edges: {'he1': torch.cat([edges.src['h'], edges.data['he']], dim=1)})
            graph.edata['he1'] = torch.tanh(self.linears1(graph.edata['he1']))
            graph.ndata['hv_new'] = ft
            graph.apply_edges(lambda egdes: {'he2': torch.cat([egdes.dst['hv_new'], graph.edata['he1']], dim=1)})
            graph.update_all(fn.copy_e('he2', 'm'), fn.mean('m', 'a'))
            hf = graph.ndata.pop('a')
            global_g = torch.tanh(self.linears2(hf))

            return global_g


class WeightAndSum(nn.Module):
    def __init__(self, in_feats):
        super(WeightAndSum, self).__init__()
        self.in_feats = in_feats
        self.atom_weighting = nn.Sequential(
            nn.Linear(in_feats, 1)
        )

    def forward(self, g, feats):
        with g.local_scope():
            g.ndata['h'] = feats
            atom_weights = self.atom_weighting(g.ndata['h'])
            g.ndata['w'] = torch.nn.Sigmoid()(self.atom_weighting(g.ndata['h']))
            h_g_sum = sum_nodes(g, 'h', 'w')

        return h_g_sum, atom_weights


class WeightAndSum_macro_ring(nn.Module):
    def __init__(self, in_feats):
        super(WeightAndSum_macro_ring, self).__init__()
        self.in_feats = in_feats
        self.atom_weighting = nn.Sequential(
            nn.Linear(in_feats, 1)
        )
        self.vir_atom_weighting = nn.Sequential(nn.Linear(in_feats, 1))

    def reset_parameters(self):
        gain = nn.init.calculate_gain('sigmoid')
        nn.init.xavier_normal_(self.atom_weighting, gain=gain)
        nn.init.xavier_normal_(self.vir_atom_weighting, gain=gain)

    def forward(self, g, feats, bg_vir_atom):
        with g.local_scope():
            # flag = g.vir_node
            g.ndata['h'] = feats
            # print(g.ndata['h'].shape)
            # print(bg_vir_atom)
            g_list = dgl.unbatch(g)
            g.atom = []
            g.vir_atom = []
            fea = feats.shape[1]
            for idx, g_one in enumerate(g_list):
                num = g_one.ndata['h'].shape[0]
                if len(bg_vir_atom[idx]) == 0:
                    g.atom.append(g_one.ndata['h'])
                    g.vir_atom.append(torch.zeros(1, fea, dtype=torch.float))
                else:
                    g.atom.append(g_one.ndata['h'][:num - len(bg_vir_atom[idx])])
                    g.vir_atom.append(g_one.ndata['h'][num - len(bg_vir_atom[idx]):])
            g.atom = torch.cat(g.atom, dim=0)
            g.vir_atom = torch.cat(g.vir_atom, dim=0)
            # print(g.atom.shape)
            # print(g.vir_atom.shape)
            atom_weights = self.atom_weighting(g.atom)
            # vir_atom_weights = self.vir_atom_weighting(g.vir_atom)
            w_atom = torch.nn.Sigmoid()(self.atom_weighting(g.atom))
            w_vir_atom = torch.nn.Sigmoid()(self.vir_atom_weighting(g.vir_atom))
            g.atom = g.atom * w_atom
            g.vir_atom = g.vir_atom * w_vir_atom
            offsets_vir_atom = torch.tensor([len(vir_atom) for vir_atom in bg_vir_atom], dtype=torch.int)
            offsets_atom = torch.tensor([vir_atom - len(bg_vir_atom[i]) for i, vir_atom in
                                         enumerate(g.batch_num_nodes().tolist())], dtype=torch.int)
            offsets_vir_atom = torch.tensor([vir_atom + 1 if vir_atom < 1 else vir_atom
                                             for vir_atom in offsets_vir_atom.tolist()], dtype=torch.int)
            # print(offsets_atom)
            # print(offsets_vir_atom)
            # exit()
            h_g_sum_atom = segment.segment_reduce(offsets_atom, g.atom, reducer='sum')
            h_g_sum_vir_atom = segment.segment_reduce(offsets_vir_atom, g.vir_atom, reducer='sum')
            # print(h_g_sum_atom.dtype)
            # print(h_g_sum_atom.shape)
            # print(h_g_sum_vir_atom.shape)
            # h_g_sum_atom = sum_nodes(g, 'h', 'w1')

        return h_g_sum_atom, h_g_sum_vir_atom, atom_weights


class WeightedSumAndMax(nn.Module):
    def __init__(self, in_feats):
        super(WeightedSumAndMax, self).__init__()
        self.weight_and_sum = WeightAndSum(in_feats)
        # self.weight_and_sum_macro_ring = WeightAndSum_macro_ring(in_feats)

    def forward(self, bg, feats, get_node_weight=True):
        h_g_sum, atom_weight = self.weight_and_sum(bg, feats)
        with bg.local_scope():
            bg.ndata['h'] = feats
            h_g_max = dgl.max_nodes(bg, 'h')
        # h_g = torch.cat([h_g_sum, h_g_max], dim=1)
        h_g = h_g_sum
        # atom_weight = self.weight_and_sum(bg, feats)[1]

        # h_g_sum_atom, h_g_sum_vir_atom, atom_weight = self.weight_and_sum_macro_ring(bg, feats, bg_vir_atom)
        # h_g = torch.cat([h_g_sum_atom, h_g_sum_vir_atom], dim=1)

        if get_node_weight:
            return h_g, atom_weight
        else:
            return h_g


class Motif_trans(nn.Module):
    def __init__(self, num_embeddings=800, embedding_dim=20, padding_idx=400, d_model=20, nhead=4,
                 dim_feedforward=40, dropout=0.5, num_layers=2, layer_norm_eps=1e-5, batch_first=True):
        super(Motif_trans, self).__init__()
        self.embed = nn.Embedding(num_embeddings, embedding_dim, padding_idx)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=dim_feedforward,
                                                        dropout=dropout, )
        self.encoder_trans = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.out = nn.Linear(embedding_dim * 7, 10)

    def forward(self, motif_vocab):
        motif_embed = self.embed(motif_vocab)
        # print(motif_embed.shape)
        X = motif_embed.permute(1, 0, 2)
        X = self.encoder_trans(X).permute(1, 0, 2)
        X = X.reshape((-1, motif_vocab.shape[1] * motif_embed.shape[-1]))
        X = self.out(X)


        return X


class AtomEmbedding(nn.Module):
    def __init__(self, n_fingerdict=4700, feature_dim=42):
        super(AtomEmbedding, self).__init__()
        self.embed = nn.Embedding(n_fingerdict, feature_dim)

    def forward(self, fingerprints):
        x_atoms = self.embed(fingerprints)

        return x_atoms


def reducer_k(nodes):
    return {'k': torch.tensor([eid if len(eid) == 5 else
                               eid + [[0 for i in range(len(eid[0]))] for id in range(5 - len(eid))]
                               for eid in nodes.mailbox['m'].tolist()]).to(device)}


def reducer_v(nodes):
    return {'v': torch.tensor([eid if len(eid) == 5 else
                               eid + [[0 for i in range(len(eid[0]))] for id in range(5 - len(eid))]
                               for eid in nodes.mailbox['m'].tolist()]).to(device)}


def count_pad(nodes):
    # print(torch.tensor(list(map(lambda x: x == 0, nodes.sum(2).numpy()))))
    return torch.tensor(list(map(lambda x: x == 0, nodes.sum(2).to('cpu').data.numpy()))).to(device)


class Self_atten(nn.Module):
    def __init__(self, embed_dim, num_heads=6, dropout=0., bias=False, **kwargs):
        super(Self_atten, self).__init__()
        self.embed_dim = embed_dim
        self.bias = bias
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim ** -0.5
        self.to_q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.to_k = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.to_v = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attend = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(embed_dim, embed_dim * 2)
        self.linear2 = nn.Linear(embed_dim * 2, embed_dim)
        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)
        self.activation = F.relu
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        nn.init.xavier_uniform_(self.to_v.weight)

        if self.bias:
            nn.init.constant_(self.to_q.bias, 0.)
            nn.init.constant_(self.to_k.bias, 0.)
            nn.init.constant_(self.to_v.bias, 0.)

    def forward(self, graph, q, k, v, return_attn=False):
        q1, k1, v1 = self.to_q(q), self.to_k(k), self.to_v(v)
        q1 = rearrange(q1, 'b n (h d) -> b h n d', h=self.num_heads)
        k1 = rearrange(k1, 'b n (h d) -> b h n d', h=self.num_heads)
        v1 = rearrange(v1, 'b n (h d) -> b h n d', h=self.num_heads)
        dots = torch.matmul(q1, k1.transpose(-1, -2)) * self.scale
        key_pad_mask = count_pad(graph.ndata['k']).unsqueeze(1).unsqueeze(1)
        if key_pad_mask is not None and key_pad_mask.dtype == torch.bool:
            dots = dots.masked_fill(key_pad_mask, float('-inf'))
        dots = self.attend(dots)
        dots = self.attn_dropout(dots)
        out = torch.matmul(dots, v1)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.out_proj(out)
        q = q + self.dropout1(out)
        q = self.norm1(q)
        out = self.linear2(self.dropout2(self.activation(self.linear1(q))))
        q = q + self.dropout3(out)
        q = self.norm2(q)

        if return_attn:
            return q, dots
        return q, None


class Attention_weight_trans(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 dim,
                 edge_feat_size,
                 negative_slope=0.2,
                 feat_drop=0,
                 residual=False,
                 allow_zero_in_degree=False,
                 bias=True, ):
        super(Attention_weight_trans, self).__init__()
        self.self_atten1 = Self_atten(in_feats, num_heads, dropout=0.5)
        self.self_atten2 = Self_atten(in_feats, num_heads, dropout=0.5)
        self._num_heads = num_heads
        self.dim = dim
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.atten = nn.MultiheadAttention(42, 6, dropout=0.5, )
        self.edge_feat_size = edge_feat_size
        self.full = nn.Linear(out_feats * num_heads, out_feats)
        self.linears1 = nn.Linear(in_feats + edge_feat_size, in_feats)
        self.linears2 = nn.Linear(out_feats * 2, dim)
        self.linears3 = nn.Linear(in_feats, dim)
        self.linears4 = nn.Linear(dim, int(dim / 2))

        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)

        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        gain_tanh = nn.init.calculate_gain('tanh')
        nn.init.xavier_normal_(self.linears1.weight, gain=gain_tanh)
        nn.init.xavier_normal_(self.linears2.weight, gain=gain_tanh)
        nn.init.xavier_normal_(self.linears3.weight, gain=gain_tanh)
        nn.init.xavier_normal_(self.linears4.weight, gain=gain_tanh)

        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, edge_feat, ):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')
                    # print('There are 0-in-degree nodes in the graph')
            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                # feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                # feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                # feat_src = feat_dst = self.fc(h_src).view(
                #     -1, self._num_heads, self._out_feats)
            graph.srcdata.update({'ft': h_src})
            # 暂存源点的特征作为key_src到相应单向边上
            # graph.apply_edges(fn.copy_u('ft', 'k_src'))
            # 合并点和边的特征线性转换为value并暂存在相应单向边上,点特征暂时只加源点特征
            graph.edata['he'] = edge_feat
            graph.apply_edges(lambda edges: {'v1': torch.cat([edges.src['ft'], edges.data['he']], dim=1)})
            graph.edata['v1'] = self.linears1(graph.edata['v1'])
            # graph.edata['v1'] = torch.tanh(self.linears1(graph.edata['v1']))
            # graph.apply_edges(lambda edges: {'v2': torch.cat([edges.dst['ft'], edges.data['v1']], dim=1)})
            # graph.edata['v'] = self.linears2(graph.edata['v2'])
            # graph.edata['v'] = torch.tanh(self.linears2(graph.edata['v2']))
            # 合并k、v储存在点'k''v'上
            graph.update_all(fn.copy_u('ft', 'm'), reducer_k)
            graph.update_all(fn.copy_e('v1', 'm'), reducer_v)
            # print(graph.ndata['v'])
            # 转换q的维度,暂存在点'q'上
            # q = graph.ndata['ft'].unsqueeze(1).permute(1, 0, 2)
            # key_pad_mask = count_pad(graph.ndata['k'])
            # # 多头单q注意力汇聚
            # output, output_weight = self.atten(q, graph.ndata['k'].permute(1, 0, 2),
            #                                    graph.ndata['v'].permute(1, 0, 2), key_pad_mask)
            # hf = output.permute(1, 0, 2).squeeze(1)
            q = graph.ndata['ft'].unsqueeze(1)
            k = graph.ndata['k']
            v = graph.ndata['v']
            q, _ = self.self_atten1(graph, q, k, v)
            out, _ = self.self_atten2(graph, q, k, v)
            hf = out.squeeze(1)
            global_g = torch.tanh(self.linears3(hf))

            return global_g


class gat_net(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 dim,
                 edge_feat_size,
                 negative_slope=0.2,
                 feat_drop=0, ):
        super(gat_net, self).__init__()
        self.self_atten1 = Self_atten(in_feats, num_heads, dropout=0.5)
        self.self_atten2 = Self_atten(in_feats, num_heads, dropout=0.5)
        self._num_heads = num_heads
        self.dim = dim
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self.atten = nn.MultiheadAttention(128, 8, dropout=0.5, )
        self.edge_feat_size = edge_feat_size
        self.full = nn.Linear(out_feats * num_heads, out_feats)
        self.linears1 = nn.Linear(in_feats + edge_feat_size, in_feats)
        self.linears2 = nn.Linear(out_feats * 2, dim)
        self.linears3 = nn.Linear(in_feats, dim)
        self.linears4 = nn.Linear(dim, int(dim / 2))
        self.linears5 = nn.Linear(int(dim / 2), int(dim / 2))

        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.feat_drop = nn.Dropout(feat_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        gain_tanh = nn.init.calculate_gain('tanh')
        nn.init.xavier_normal_(self.linears1.weight, gain=gain_tanh)
        nn.init.xavier_normal_(self.linears2.weight, gain=gain_tanh)
        nn.init.xavier_normal_(self.linears3.weight, gain=gain_tanh)
        nn.init.xavier_normal_(self.linears4.weight, gain=gain_tanh)
        nn.init.xavier_normal_(self.linears5.weight, gain=gain_tanh)
        # nn.init.xavier_normal_(self.attn_l, gain=gain)
        # nn.init.xavier_normal_(self.attn_r, gain=gain)
        # if isinstance(self.res_fc, nn.Linear):
        #     nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, ):
        with graph.local_scope():
            h_src = h_dst = self.feat_drop(feat)
            graph.srcdata.update({'ft': h_src})
            # 合并k、v储存在点'k''v'上
            graph.update_all(fn.copy_u('ft', 'm'), reducer_k)
            graph.ndata['v'] = graph.ndata['k']
            # 转换q的维度,暂存在点'q'上
            # q = graph.ndata['ft'].unsqueeze(1).permute(1, 0, 2)
            # key_pad_mask = count_pad(graph.ndata['k'])
            # # 多头单q注意力汇聚
            # output, output_weight = self.atten(q, graph.ndata['k'].permute(1, 0, 2),
            #                                    graph.ndata['v'].permute(1, 0, 2), key_pad_mask)
            # hf = output.permute(1, 0, 2).squeeze(1)
            # global_g = torch.tanh(self.linears4(hf))
            q = graph.ndata['ft'].unsqueeze(1)
            k = graph.ndata['k']
            v = graph.ndata['v']
            q, _ = self.self_atten1(graph, q, k, v)
            out, _ = self.self_atten2(graph, q, k, v)
            hf = out.squeeze(1)
            global_g = torch.tanh(self.linears4(hf))

            return global_g


class gat_net_layer34(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 dim,
                 edge_feat_size,
                 negative_slope=0.2,
                 feat_drop=0, ):
        super(gat_net_layer34, self).__init__()
        self.self_atten1 = Self_atten(in_feats, num_heads, dropout=0.5)
        self.self_atten2 = Self_atten(in_feats, num_heads, dropout=0.5)
        self._num_heads = num_heads
        self.dim = dim
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self.atten = nn.MultiheadAttention(64, 8, dropout=0.5, )
        self.edge_feat_size = edge_feat_size
        self.full = nn.Linear(out_feats * num_heads, out_feats)
        self.linears1 = nn.Linear(in_feats + edge_feat_size, in_feats)
        self.linears2 = nn.Linear(out_feats * 2, dim)
        self.linears3 = nn.Linear(in_feats, dim)
        self.linears4 = nn.Linear(dim, int(dim / 2))
        self.linears5 = nn.Linear(int(dim / 2), int(dim / 2))

        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.feat_drop = nn.Dropout(feat_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        gain_tanh = nn.init.calculate_gain('tanh')
        nn.init.xavier_normal_(self.linears1.weight, gain=gain_tanh)
        nn.init.xavier_normal_(self.linears2.weight, gain=gain_tanh)
        nn.init.xavier_normal_(self.linears3.weight, gain=gain_tanh)
        nn.init.xavier_normal_(self.linears4.weight, gain=gain_tanh)
        nn.init.xavier_normal_(self.linears5.weight, gain=gain_tanh)
        # nn.init.xavier_normal_(self.attn_l, gain=gain)
        # nn.init.xavier_normal_(self.attn_r, gain=gain)
        # if isinstance(self.res_fc, nn.Linear):
        #     nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, ):
        with graph.local_scope():
            h_src = h_dst = self.feat_drop(feat)
            graph.srcdata.update({'ft': h_src})
            # 合并k、v储存在点'k''v'上
            graph.update_all(fn.copy_u('ft', 'm'), reducer_k)
            graph.ndata['v'] = graph.ndata['k']
            # 转换q的维度,暂存在点'q'上
            # q = graph.ndata['ft'].unsqueeze(1).permute(1, 0, 2)
            # key_pad_mask = count_pad(graph.ndata['k'])
            # # 多头单q注意力汇聚
            # output, output_weight = self.atten(q, graph.ndata['k'].permute(1, 0, 2),
            #                                    graph.ndata['v'].permute(1, 0, 2), key_pad_mask)
            # hf = output.permute(1, 0, 2).squeeze(1)
            # global_g = torch.tanh(self.linears5(hf))
            q = graph.ndata['ft'].unsqueeze(1)
            k = graph.ndata['k']
            v = graph.ndata['v']
            q, _ = self.self_atten1(graph, q, k, v)
            out, _ = self.self_atten2(graph, q, k, v)
            hf = out.squeeze(1)
            global_g = torch.tanh(self.linears5(hf))

            return global_g
