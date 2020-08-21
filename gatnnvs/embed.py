import logging
import hydra
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from rdkit.Chem import AllChem

from gatnnvs.model import build_model, load_model
from gatnnvs.dataset import GraphDataset, make_graph_batch, is_mol_usable
from gatnnvs.pretrained import ensure_model

log = logging.getLogger(__name__)


def desalt_smiles(smi):
    return max(smi.split('.'), key=len) if '.' in smi else smi


def run(net, device, batch_size, infile, outfile):

    num_params = sum([p.numel() for p in net.parameters()])
    log.info(f'Parameter count: {num_params}')


    ## Load dataset
    data = pd.read_table(infile, sep=',')[['smiles']]
    data['mol'] = data.smiles.map(lambda s: AllChem.MolFromSmiles(desalt_smiles(s)))
    data['usable'] = data.mol.map(is_mol_usable)
    usable = data[data.usable].reset_index(drop=True)
    skipped = data[~data.usable].index.values
    log.info(f'Usable: {len(usable)}\t Unusable: {len(data[~data.usable])}')
    
    loader = torch.utils.data.DataLoader(
        GraphDataset(usable, embed_only=True), 
        batch_size=batch_size,
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


@hydra.main(config_path="embed-config.yaml")
def main(cfg):
    log.info(cfg.pretty())
    inp = Path(cfg.input)
    device = cfg.device
    if not device:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    ensure_model(cfg.model_path)
    net = load_model(cfg.model_path, device)

    out = cfg.output
    if not out:
        out = inp.stem
    run(net, device, cfg.batch_size, inp, out)


if __name__ == "__main__":
    main()
