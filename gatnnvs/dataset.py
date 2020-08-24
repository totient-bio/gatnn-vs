import torch
import dgl
from rdkit import Chem


ATOMS = 'X', 'C', 'N', 'O', 'S', 'Cl', 'F', 'Br', 'P', 'I'
CHIRALITY = {
    Chem.ChiralType.CHI_UNSPECIFIED:           0, 
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW: 4 + len(ATOMS),
    Chem.ChiralType.CHI_TETRAHEDRAL_CCW:       5 + len(ATOMS),
}
BOND_TYPE = Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC


def is_mol_usable(mol):
    if not mol or not isinstance(mol, Chem.Mol):
        return False
    for a in mol.GetAtoms():
        if a.GetSymbol() not in ATOMS:
            return False
        if a.GetChiralTag() not in CHIRALITY:
            return False
    return True


def make_dglg(mol):
    src = torch.as_tensor([b.GetBeginAtomIdx() for b in mol.GetBonds()], dtype=torch.int64)
    dst = torch.as_tensor([b.GetEndAtomIdx() for b in mol.GetBonds()], dtype=torch.int64)
    btype = torch.as_tensor([BOND_TYPE.index(b.GetBondType()) for b in mol.GetBonds()], dtype=torch.int64)

    ndata = {
        'type': [ATOMS.index(a.GetSymbol()) for a in mol.GetAtoms()],
        'hs': [a.GetTotalNumHs() + len(ATOMS) for a in mol.GetAtoms()],
        'chirality': [CHIRALITY[a.GetChiralTag()] for a in mol.GetAtoms()]
    }
    ndata = {k: torch.as_tensor(v, dtype=torch.int64) for k, v in ndata.items()}
    
    g = dgl.DGLGraph()
    g.add_nodes(mol.GetNumHeavyAtoms())
    g.ndata['type'] = torch.stack([ndata['type'], ndata['hs'], ndata['chirality']], dim=-1)
    g.add_edges(src, dst)
    g.add_edges(dst, src)
    g.add_edges(range(mol.GetNumHeavyAtoms()), range(mol.GetNumHeavyAtoms()))
    ones = torch.ones(mol.GetNumHeavyAtoms(), dtype=torch.int64) 
    g.edata['type'] = torch.cat([btype.repeat(2), ones * 4])
    return g


class GraphDataset:
    def __init__(self, df, mol_column='mol', target_columns=None, embed_only=False):
        self.df = df
        self.mol_column = mol_column
        if target_columns is None:
            target_columns = [c for c in df.columns if c.startswith('Status_')]
        if not embed_only:
            self.actives = df[target_columns].values == 'Active'
            self.valids = self.actives | (df[target_columns].values == 'Inactive')

    def __getitem__(self, idx):
        mol = self.df.loc[idx, self.mol_column]
        g = make_dglg(mol)
        if not hasattr(self, 'actives'):
            return g
        a = torch.tensor(self.actives[idx], dtype=torch.float32)
        v = torch.tensor(self.valids[idx], dtype=torch.float32)
        return g, a, v

    def __len__(self):
        return len(self.df)


def make_graph_batch(data):
    g = dgl.batch([d[0] for d in data])
    a = torch.stack([d[1] for d in data])
    v = torch.stack([d[2] for d in data])
    g.set_n_initializer(dgl.init.zero_initializer)
    g.set_e_initializer(dgl.init.zero_initializer)
    return g, a, v
