import torch
from torch import nn
import cytoolz.curried as ct
import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn


class BatchEmbeddingBag(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, max_norm=None, norm_type=2,
                 scale_grad_by_freq=False, mode='mean', sparse=False, padding_idx=None):
        super().__init__()
        self.eb = nn.EmbeddingBag(num_embeddings, embedding_dim, max_norm=max_norm,
                                  norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq,
                                  mode=mode, sparse=sparse)
        if padding_idx is not None:
            self.eb.weight.data[padding_idx] = 0

    @property
    def weight(self):
        return self.eb.weight

    def forward(self, x):
        s = x.shape[:-1] + (-1,)
        return self.eb(x.reshape(-1, x.shape[-1])).reshape(s)


class Embed(nn.Module):
    def __init__(self, node_dim, edge_dim):
        super().__init__()
        self.emb_node_types = BatchEmbeddingBag(16, node_dim, padding_idx=0)
        self.emb_edge_types = nn.Embedding(5, edge_dim)
        
    def forward(self, g):
        emb = self.emb_node_types(g.ndata['type'])
        g.ndata['emb'] = emb
        
        emb = self.emb_edge_types(g.edata['type'])
        g.edata['emb'] = emb
        return g
      

class GraphCast(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.dtype = dtype
    
    def forward(self, g):
        g.ndata['emb'] = g.ndata['emb'].to(self.dtype)
        g.edata['emb'] = g.edata['emb'].to(self.dtype)
        return g
    

class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, init=ct.curry(nn.init.xavier_normal_)(gain=1.414)):
        self.init = init
        super().__init__(in_features, out_features, bias)
        
    def reset_parameters(self):
        self.init(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    

class MultiLinear(nn.Module):
    def __init__(self, in_features, out_features, n_linears=1, bias=True,
                init=ct.curry(nn.init.xavier_normal_)(gain=nn.init.calculate_gain('relu'))):
        super().__init__()
        self.out_features = out_features
        self.n_linears = n_linears
        self.in_features = in_features
        self.init = init
        weights = torch.zeros(n_linears, in_features, out_features, dtype=torch.float32)
        init(weights)
        self.lin = nn.Parameter(weights)
        self.init(self.lin.data)
        if bias:
            b = torch.zeros((n_linears, 1, self.out_features), dtype=torch.float32)
            self.bias = nn.Parameter(b)
        else:
            self.bias = None
            
    def extra_repr(self):
        return f'{self.in_features}, {self.out_features}, {self.n_linears}'
    
    def forward(self, x):
        batch = x.shape[0]
        x = x.view(batch, self.n_linears, -1).permute(1, 0, 2)
        if self.bias is not None:
            y = torch.baddbmm(self.bias.expand(self.n_linears, batch, self.out_features),
                             x,
                             self.lin)
        else:
            y = torch.bmm(input=x,mat2=self.lin.data)
        return y.permute(1, 0, 2).contiguous()
            

class Lambda(nn.Module):
    def __init__(self, fn):
        self.fn = fn
        
    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)
    

class GraphLambda(nn.Module):
    def __init__(self, fn, node_key='emb', edge_key='emb'):
        super().__init__()
        self.fn = fn
        self.edge_key = edge_key
        self.node_key = node_key
    
    def forward(self, g):
        if self.node_key:
            g.ndata[self.node_key] = self.fn(g.ndata[self.node_key])
        if self.edge_key:
            g.edata[self.edge_key] = self.fn(g.edata[self.edge_key])
        return g
    
ReduceMean = lambda: GraphLambda(lambda x: x.mean(dim=-2))
ReduceCat = lambda: GraphLambda(lambda x: x.view(x.shape[0], -1))


class Residual(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
    
    def forward(self, g):
        nemb = g.ndata['emb']
        eemb = g.edata['emb']
        g = self.module(g)
        g.ndata['emb'] += nemb
        g.edata['emb'] += eemb
        return g


class GatedResidual(nn.Module):
    def __init__(self, module, node_dim, edge_dim):
        super().__init__()
        self.module = module
        sig_gain = nn.init.calculate_gain('sigmoid')
        self.prev_node_gate = Linear(
            node_dim, node_dim, bias=False,
            init=ct.curry(nn.init.xavier_normal_)(gain=sig_gain))
        self.curr_node_gate = Linear(
            node_dim, node_dim, bias=True,
            init=ct.curry(nn.init.xavier_normal_)(gain=sig_gain))
        self.prev_edge_gate = Linear(
            edge_dim, edge_dim, bias=False,
            init=ct.curry(nn.init.xavier_normal_)(gain=sig_gain))
        self.curr_edge_gate = Linear(
            edge_dim, edge_dim, bias=True,
            init=ct.curry(nn.init.xavier_normal_)(gain=sig_gain))
        nn.init.zeros_(self.curr_node_gate.bias.data)
        nn.init.zeros_(self.curr_edge_gate.bias.data)
    
    def forward(self, g):
        prev_node = g.ndata['emb']
        prev_edge = g.edata['emb']
        
        g = self.module(g)
        
        node_z = torch.sigmoid(
            self.prev_node_gate(prev_node) + \
            self.curr_node_gate(g.ndata['emb']))
        edge_z = torch.sigmoid(
            self.prev_edge_gate(prev_edge) + \
            self.curr_edge_gate(g.edata['emb']))
        g.ndata['emb'] = node_z * g.ndata['emb'] + (1 - node_z) * prev_node
        g.edata['emb'] = edge_z * g.edata['emb'] + (1 - edge_z) * prev_edge
        
        return g


class TripletLinear(nn.Module):
    def __init__(self, in_node_dim, in_edge_dim, out_edge_dim, bias=False):
        super().__init__()
        self.lin = Linear(in_node_dim * 2 + in_edge_dim, out_edge_dim, bias)
        
    def triplet_linear(self, edges):
        triplets = torch.cat([edges.src['emb'], edges.data['emb'], edges.dst['emb']], dim=-1)
        return { 'triplets' : triplets }
    
    def forward(self, g):
        g.apply_edges(self.triplet_linear)
        g.edata['emb'] = self.lin(g.edata.pop('triplets'))
        return g

      
class TripletMultiLinear(nn.Module):
    def __init__(self, in_node_dim, in_edge_dim, out_edge_dim, n_lins, bias=False):
        super().__init__()
        self.lin = MultiLinear(in_node_dim * 2 + in_edge_dim, out_edge_dim, n_lins, bias)
        
    def triplet_linear(self, edges):
        triplets = torch.cat([edges.src['emb'], edges.data['emb'], edges.dst['emb']], dim=-1)
        return { 'triplets' : triplets }
    
    def forward(self, g):
        g.apply_edges(self.triplet_linear)
        g.edata['emb'] = self.lin(g.edata.pop('triplets'))
        return GraphLambda(lambda x: x.view(x.shape[0], -1), node_key=None)(g)


class TripletCat(nn.Module):
    def __init__(self, out='emb'):
        super().__init__()
        self.out = out

    def triplet_linear(self, edges):
        triplets = torch.cat([edges.src['emb'], edges.data['emb'], edges.dst['emb']], dim=-1)
        return { self.out : triplets }
    
    def forward(self, g):
        g.apply_edges(self.triplet_linear)
        return g
    

class MagicAttn(nn.Module):
    def __init__(self, node_dim, edge_dim, n_heads, attn_key='emb', msg_key='emb', alpha=.2):
        super().__init__()
        self.attn = MultiLinear(
            edge_dim, 1, n_heads, bias=False,
            init=ct.curry(nn.init.xavier_normal_)(gain=nn.init.calculate_gain('leaky_relu', alpha)))
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.n_heads = n_heads
#         self.softmax = EdgeSoftmax.apply
        self.attn_key = attn_key
        self.msg_key = msg_key
        
    def forward(self, g):
        alpha_prime = self.leaky_relu(self.attn(g.edata[self.attn_key]))
        g.edata['a'] = dglnn.edge_softmax(g, alpha_prime) * g.edata['emb'].view(g.edata['emb'].shape[0], self.n_heads, -1)
        attn_emb = g.ndata[self.msg_key]
        if attn_emb.ndimension() == 2:
            g.ndata[self.msg_key] = attn_emb.view(g.number_of_nodes(), self.n_heads, -1)
        g.update_all(fn.src_mul_edge(self.msg_key, 'a', 'm'), fn.sum('m', 'emb'))
        return GraphLambda(lambda x: x.view(x.shape[0], -1))(g)


class GraphAttnPool(nn.Module):
    def __init__(self, node_dim, n_heads, attn_key='emb', msg_key='emb', alpha=.2):
        super().__init__()
        self.attn = MultiLinear(
            node_dim, 1, n_heads, bias=False,
            init=ct.curry(nn.init.xavier_normal_)(gain=nn.init.calculate_gain('leaky_relu', alpha)))
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.n_heads = n_heads
        self.msg_key = msg_key
        self.attn_key = attn_key
        
    def forward(self, g):
        g.ndata['a'] = self.leaky_relu(self.attn(g.ndata[self.attn_key]))
        g.ndata['a'] = dgl.softmax_nodes(g, 'a')
        attn_emb = g.ndata[self.msg_key]
        if attn_emb.ndimension() == 2:
            g.ndata[self.msg_key] = attn_emb.view(g.number_of_nodes(), self.n_heads, -1)
        g.ndata['a'] = g.ndata[self.msg_key] * g.ndata['a'].unsqueeze(-1)
        graph_emb = dgl.sum_nodes(g, 'a')
        
        return graph_emb.view(graph_emb.shape[0], -1)
      

def NodeLinear(in_features, out_features, bias=False):
    return GraphLambda(Linear(in_features, out_features, bias), edge_key=None)


def EdgeLinear(in_features, out_features, bias=False):
    return GraphLambda(Linear(in_features, out_features, bias), node_key=None)


class ValidBCELoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction='none', *args, **kwargs)
        
    def forward(self, x, y):
        unreduced = self.loss(x, y['active'])
        valid = unreduced * y['valid']
        return valid.mean()

    
def AttnBlock(in_emb, out_emb, gattn_emb, gattn_heads, bias=False):
    return nn.Sequential(
        EdgeLinear(in_emb, out_emb),
        NodeLinear(in_emb, out_emb),
        GraphLambda(lambda x: x.view(x.shape[0], gattn_heads, -1)),
        TripletCat(out='triplet'),
        MagicAttn(gattn_emb, 3 * gattn_emb, gattn_heads, attn_key='triplet'),
        TripletMultiLinear(gattn_emb, gattn_emb, gattn_emb, gattn_heads, bias=bias),
        GraphLambda(torch.nn.LayerNorm(gattn_heads * gattn_emb), node_key=None),
        GraphLambda(torch.nn.LayerNorm(gattn_heads * gattn_emb), edge_key=None)
    )


def gr(emb, heads):
    return GatedResidual(AttnBlock(emb * heads, emb * heads, emb, heads), emb * heads, emb * heads)
