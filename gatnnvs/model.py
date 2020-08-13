import itertools
from torch import nn
from gatnnvs.modules import *


def noreg(x):
    return type(x) == nn.PReLU or (type(x) == GraphLambda and type(x.fn) == nn.PReLU)


def build_model(gattn_emb, gattn_heads, final_emb, num_classes, device, dropout=.5, final_relu=False):
    net = nn.Sequential(
        Embed(gattn_emb, gattn_emb),
        AttnBlock(gattn_emb, gattn_emb * gattn_heads, gattn_emb, gattn_heads), 
        GraphLambda(nn.PReLU(), node_key=None), GraphLambda(nn.PReLU(), edge_key=None),
        gr(gattn_emb, gattn_heads), GraphLambda(nn.PReLU(), node_key=None), GraphLambda(nn.PReLU(), edge_key=None),
        gr(gattn_emb, gattn_heads), GraphLambda(nn.PReLU(), node_key=None), GraphLambda(nn.PReLU(), edge_key=None),
        GraphLambda(nn.Dropout(dropout)),
        gr(gattn_emb, gattn_heads), GraphLambda(nn.PReLU(), node_key=None), GraphLambda(nn.PReLU(), edge_key=None),
        #GraphLambda(nn.Dropout(dropout)),
        gr(gattn_emb, gattn_heads), GraphLambda(nn.PReLU(), node_key=None), GraphLambda(nn.PReLU(), edge_key=None),
        GraphLambda(nn.Dropout(dropout)),
        gr(gattn_emb, gattn_heads), GraphLambda(nn.PReLU(), node_key=None), GraphLambda(nn.PReLU(), edge_key=None),
        gr(gattn_emb, gattn_heads), GraphLambda(nn.PReLU(), node_key=None), GraphLambda(nn.PReLU(), edge_key=None),
        NodeLinear(gattn_emb * gattn_heads, gattn_emb * gattn_heads, bias=False),
        GraphAttnPool(gattn_emb, gattn_heads), torch.nn.BatchNorm1d(gattn_heads * gattn_emb), nn.PReLU(),
        nn.Dropout(dropout),
        Linear(gattn_emb * gattn_heads, final_emb, bias=False),
        Linear(final_emb, num_classes, bias=True)
    )
    return net.to(device)
