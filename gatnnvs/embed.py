import sys
import logging
import hydra
from omegaconf import OmegaConf
import torch
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from rdkit.Chem import AllChem

from gatnnvs.model import build_model
from gatnnvs.dataset import GraphDataset, make_graph_batch, is_mol_usable

log = logging.getLogger(__name__)


def desalt_smiles(smi):
    return max(smi.split('.'), key=len) if '.' in smi else smi


def run(cfg, rundir, epoch, infile, outfile):
    ## Build model and load checkpoint
    device = torch.device(cfg.device)
    net = build_model(
        cfg.net.gattn_emb, 
        cfg.net.gattn_heads, 
        cfg.net.final_emb, 
        cfg.num_classes, 
        device=device,
        dropout=cfg.net.dropout,
    )
    num_params = sum([p.numel() for p in net.parameters()])
    log.info(f'Parameter count: {num_params}')

    checkpoint = Path(rundir) / 'checkpoints' / f'{epoch}.last_10.torch'
    checkpoint = torch.load(str(checkpoint))
    net.load_state_dict(checkpoint['model'])
    del net[-1]

    ## Load dataset
    data = pd.read_table(infile, sep=',')[['smiles']]
    data['mol'] = data.smiles.map(lambda s: AllChem.MolFromSmiles(desalt_smiles(s)))
    data['usable'] = data.mol.map(is_mol_usable)
    usable = data[data.usable].reset_index(drop=True)
    skipped = data[~data.usable].index.values
#     skipped.tofile('skipped.csv', sep=',')
    log.info(f'Usable: {len(usable)}\t Unusable: {len(data[~data.usable])}')
    
    loader = torch.utils.data.DataLoader(
        GraphDataset(usable, embed_only=True), 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        num_workers=0, 
        collate_fn=make_graph_batch
    )

    net.eval()
    outs = []
    with torch.no_grad():
        for batch in tqdm(loader):
            g, a, v = batch
            g.to(device)
            out = net(g)
            outs.append(out.detach().cpu().numpy())
    outs = np.concatenate(outs)
    insert_at = skipped - np.arange(len(skipped))
    outs = np.insert(outs, insert_at, np.full(outs.shape[-1], np.nan), axis=0)
    np.save(outfile, outs)


def main(rundir, epoch, inp, out):
    cfg = OmegaConf.load(str(Path(rundir)  / '.hydra' / 'config.yaml'))
    log.info(cfg.pretty())
    run(cfg, rundir, epoch, inp, out)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print('Usage: embed.py <rundir> <epoch> <input> <output>')
        sys.exit(1)
    main(*sys.argv[1:])
