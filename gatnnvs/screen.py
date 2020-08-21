import logging
import hydra
import torch
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from rdkit.Chem import AllChem
from dgl import batch

from gatnnvs.model import build_model, load_model
from gatnnvs.dataset import make_dglg, is_mol_usable
from gatnnvs.pretrained import ensure_model

log = logging.getLogger(__name__)


def desalt_smiles(smi):
    return max(smi.split('.'), key=len) if '.' in smi else smi


def similarity_cosine(x, y):
    """
    Bulk cosine similarity sprod(A, B)/|A||B|
    :param x: Known / query fingerprints
    :param y: Fingerprints to be screened
    :return: np.array of shape (len(y), len(x))
    """
    cross = np.dot(x, y.T)
    x2, y2 = (x * x).sum(axis=1), (y * y).sum(axis=1)
    score = cross.T/np.sqrt((y2.reshape(-1, 1) * x2))
    score[np.isnan(score)] = -1
    return score


def vscreen_np(known, lib, simil_fn=similarity_cosine, chunk_size=1000):
    '''
    Perform virtual screen on two numpy objects scoring chunk_size at a time.
    Result is a one-dimensional array of size len(lib), same as reduce_fn(simil_fn(known, lib)).
    '''
    chunks, num_chunks = [], math.ceil(len(lib) / chunk_size)
    for cn in range(num_chunks):
        start = cn * chunk_size
        end = min(len(lib), start + chunk_size)
        chunks.append(simil_fn(known, lib[start:end]))

    return np.concatenate(chunks)


def run(net, device, infile, library, outfile):
    num_params = sum([p.numel() for p in net.parameters()])
    log.info(f'Parameter count: {num_params}')

    ## Load dataset
    data = pd.read_table(infile, sep=',')[['smiles']]
    data['mol'] = data.smiles.map(lambda s: AllChem.MolFromSmiles(desalt_smiles(s)))
    data['usable'] = data.mol.map(is_mol_usable)
    usable = data[data.usable].reset_index(drop=True)
    log.info(f'Usable: {len(usable)}\t Unusable: {len(data[~data.usable])}')

    graphs = [make_dglg(mol) for mol in usable.mol]
    net.eval()

    with torch.no_grad():
        g = batch(graphs)
        g.to(device)
        out = net(g)
    outs = out.detach().cpu().numpy()

    library = np.load(library)

    scores = vscreen_np(outs, library)
    score_columns = ['score_' + str(i) for i in data[data.usable].index]
    scores_df = pd.DataFrame(scores, columns=score_columns)
    for i in data[data.usable].index:
        scores_df['rank_' + str(i)] = scores_df['score_' + str(i)].rank(method='first', ascending=False, na_option='bottom').astype(int)

    scores_df['max_score'] = scores_df[score_columns].max(axis=1)
    scores_df['max_score_rank'] = scores_df.max_score.rank(method='first', ascending=False, na_option='bottom').astype(int)
    scores_df['argmax_score'] = data[data.usable].index[scores_df[score_columns].values.argmax(axis=1)].astype(int)

    columns = []
    for i in data[data.usable].index:
        columns.append('score_' + str(i))
        columns.append('rank_' + str(i))

    scores_df[columns + ['max_score', 'argmax_score', 'max_score_rank']].to_csv(outfile, index_label='index')


@hydra.main(config_path="screen-config.yaml")
def main(cfg):
    log.info(cfg.pretty())
    inp = Path(cfg.query)
    device = torch.device(cfg.device)

    ensure_model(cfg.model_path)
    net = load_model(cfg.model_path, device)

    out = cfg.output
    if not out:
        out = inp.stem + '-' + Path(cfg.library).stem + '-scores.csv'
    run(net, device, inp, cfg.library, out)


if __name__ == "__main__":
    main()
