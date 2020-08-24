from pathlib import Path
from omegaconf import OmegaConf
from gatnnvs.modules import Embed, GraphAttnPool, GraphLambda, GRAttnBlock, AttnBlock, Linear, NodeLinear
import torch
from torch import nn


def noreg(x):
    return type(x) == nn.PReLU or (type(x) == GraphLambda and type(x.fn) == nn.PReLU)


def build_model(gattn_emb, gattn_heads, final_emb, num_classes, device, dropout=.5):
    net = nn.Sequential(
        Embed(gattn_emb, gattn_emb),
        AttnBlock(gattn_emb, gattn_emb, gattn_heads),
        GraphLambda(nn.PReLU(), node_key=None), GraphLambda(nn.PReLU(), edge_key=None),
        GRAttnBlock(gattn_emb, gattn_heads), GraphLambda(nn.PReLU(), node_key=None), GraphLambda(nn.PReLU(), edge_key=None),
        GRAttnBlock(gattn_emb, gattn_heads), GraphLambda(nn.PReLU(), node_key=None), GraphLambda(nn.PReLU(), edge_key=None),
        GraphLambda(nn.Dropout(dropout)),
        GRAttnBlock(gattn_emb, gattn_heads), GraphLambda(nn.PReLU(), node_key=None), GraphLambda(nn.PReLU(), edge_key=None),
        GRAttnBlock(gattn_emb, gattn_heads), GraphLambda(nn.PReLU(), node_key=None), GraphLambda(nn.PReLU(), edge_key=None),
        GraphLambda(nn.Dropout(dropout)),
        GRAttnBlock(gattn_emb, gattn_heads), GraphLambda(nn.PReLU(), node_key=None), GraphLambda(nn.PReLU(), edge_key=None),
        GRAttnBlock(gattn_emb, gattn_heads), GraphLambda(nn.PReLU(), node_key=None), GraphLambda(nn.PReLU(), edge_key=None),
        NodeLinear(gattn_emb * gattn_heads, gattn_emb * gattn_heads, bias=False),
        GraphAttnPool(gattn_emb, gattn_heads), torch.nn.BatchNorm1d(gattn_heads * gattn_emb), nn.PReLU(),
        nn.Dropout(dropout),
        Linear(gattn_emb * gattn_heads, final_emb, bias=False),
        Linear(final_emb, num_classes, bias=True)
    )
    return net.to(device)


def load_model(model_path=None, device=None):

    if model_path is None:
        embed_cfg = OmegaConf.load(__file__ + '/../embed-config.yaml')
        model_path = embed_cfg.model_path

    mp = Path(model_path)
    model_cfg = OmegaConf.load(str(mp / 'config' / 'config.yaml'))
    net = build_model(
        model_cfg.net.gattn_emb,
        model_cfg.net.gattn_heads,
        model_cfg.net.final_emb,
        model_cfg.num_classes,
        device=device,
        dropout=model_cfg.net.dropout,
    )

    weights_path = mp / 'weights.torch'
    weights = torch.load(str(weights_path))
    net.load_state_dict(weights['model'])
    del net[-1]

    return net
