# Graph Attention Neural Network for Virtual Screening

**GATNN-VS** is a neural network architecture, and a learning framework designed to yield a generally applicable tool for ligand-based virtual screening. Our approach uses the molecular graph as input, and involves learning a representation that places compounds of similar biological profiles in close proximity within a hyperdimensional feature space; this is achieved by simultaneously leveraging historical screening data against a multitude of targets during training. Cosine distance between molecules in this space becomes a general similarity metric, and can readily be used to rank order database compounds in LBVS workflows.

Checkout our paper: [Improved Scaffold Hopping in Ligand-based Virtual Screening Using Neural Representation Learning](https://pubs.acs.org/doi/10.1021/acs.jcim.0c00622)

![GATTN Virtual Screen Visualization](https://github.com/totient-bio/gatnn-vs/raw/master/images/gatnn-screening.png)

## Instalation

Project requires python 3.6+ and uses following dependencies:

* **[pytorch](https://pytorch.org)** - deep learning framework
* **[dgl](https://www.dgl.ai/)** - for graph handling
* **[rdkit](https://rdkit.org)** - for molecule processing
* Several utility packages: *tqdm*, *hydra-core*, *omegaconf*, *pandas*, *cytoolz*, *feather-format*

Please make sure the first three dependencies are installed and working; just pip installing them probably isn't enough (rdkit doesn't even have a pypi package). The easyest way is probably through [conda](https://www.anaconda.com/products/individual) package manager.

Once the environment is set up, you can install GATNN-VS with pip:

    $ pip install git+https://github.com/totient-bio/gatnn-vs.git


## Command line tools

All command line tools use [hydra](https://hydra.cc)/[omegaconf](https://github.com/omry/omegaconf) for parsing command line as well as the configuration. That way you can easily override any config parameter, e.g. for training.

### Using a pretrained model

To use trained model for screening, first you should create a fingerprint library:

    $ python3 -m gatnnvs.embed input=INPUT_FILE [model=MODEL_PATH] [output=OUTPUT_PATH] [device=DEVICE]

**Input** should be a `csv` file with a '*smiles*' column. Molecules must consist only from *C*, *N*, *O*, *S*, *Cl*, *F*, *Br*, *P* and *I* atoms (and implicit *H*s). If compound is a salt, only the largest component will be used.

**Model** is a directory with a following structure:

    model-dir
    +---config
    |   +---config.yaml  
    \---weights.torch  

If omitted, this tool will automatically download and use pretrained model.

**Device** is a pytorch device specifier. If omitted, will default to the first CUDA device.
 
**Output** will be a `npy` file. Lines from the input file will correspond with rows in generated numpy array. If rdkit can't parse smiles, or molecule contains elements that are not accepted, appropriate rows will be filled with `nan`s. Single GPU can compute roughly 3000 embeddings/sec, so for 1-2 million compounds this step should be done in minutes.

When fingerprint library is computed, you can use it to screen molecules:

    $ python3 -m gatnnvs.screen query=QUERY_FILE library=LIBRARY_FILE [model=MODEL_PATH] [output=OUTPUT_PATH] [device=DEVICE]

Similarly to the `embed` command, a **query** file is a `csv` with a '*smiles*' column, **library** is a file generated by `embed` command and **model** is used to fingerprint query molecules.

**Output** is a `csv` file with a following columns:

* *index* - row number of the input **library** array
* *score_{i}* - where *{i}* is a row number of a molecule from a **query** file. If a query molecule is rejected for some reason, it's score column will be missing from the output file.
* *rank_{i}* - integer rank of a score: highest score -> lowest rank
* *max_score* - max score this library molecule achieved across all query molecules
* *argmax_score* - index of a query molecule for which library molecule achieved maximal score
* *max_score_rank* - integer rant of a *max_score* 

### Training your own model

You can train a model yourself with:

    $ python3 -m gatnnvs.train data=DATA rundir=RUNDIR [Options...]

Input **data** shoud be a feather file with smiles column and target columns. You can control what columns are considered 'target' by specifying `target_prefix` option. Target columns should have values '*Active*', '*Inactive*' and '*Unknown*'. For large datasets, it's best if target columns are of `category` datatype.

If provided with existing **rundir**, it will automatically resume training from the latest checkpoint.

## Using GAtNN-VS from code

### Building a network

The main building block is an `AttnBlock` module:

```python
from gatnnvs.modules import Embed, GraphAttnPool, GraphLambda, AttnBlock, Linear
import torch
from torch import nn

input_emb_size = 32
gattn_emb_size = 32
gattn_heads = 16
final_emb = 1024
num_classes = 200

net = nn.Sequential(
    # create node and edge embeddings
    Embed(input_emb_size, input_emb_size),
    # Size of an input embedding is input_emb_size, and of output embedding is gattn_emb_size * gattn_heads 
    AttnBlock(input_emb_size, gattn_emb_size, gattn_heads),
    # Node and edge activation 
    GraphLambda(nn.PReLU(), node_key=None), GraphLambda(nn.PReLU(), edge_key=None),
    # Now input_size == output_size ==  gattn_emb_size * gattn_heads
    AttnBlock(gattn_emb_size * gattn_heads, gattn_emb_size, gattn_heads),
    # Pooling node embeddings into unified molecular embedding
    GraphAttnPool(gattn_emb_size, gattn_heads), torch.nn.BatchNorm1d(gattn_heads * gattn_emb_size), nn.PReLU(),
    Linear(gattn_emb_size * gattn_heads, final_emb, bias=False),
    # Final prediction layer
    Linear(final_emb, num_classes, bias=True)
)
 
```

When `emb_input_size == emb_out_size` you can use skip connections. There are `Residual` and `GatedResidual` modules implemented for graph, as well as `GRAttnBlock` that wrap regular `AttnBlock` with gated connections:

```python
from gatnnvs.modules import Embed, GraphAttnPool, GraphLambda, AttnBlock, GRAttnBlock, Linear
import torch
from torch import nn

input_emb_size = 32
gattn_emb_size = 32
gattn_heads = 16
final_emb = 1024
num_classes = 200

net = nn.Sequential(
    # create node and edge embeddings
    Embed(input_emb_size, input_emb_size),
    # Size of an input embedding is input_emb_size, and of output embedding is gattn_emb_size * gattn_heads 
    AttnBlock(input_emb_size, gattn_emb_size, gattn_heads),
    # Node and edge activation 
    GraphLambda(nn.PReLU(), node_key=None), GraphLambda(nn.PReLU(), edge_key=None),
    GRAttnBlock(gattn_emb_size, gattn_heads),
    # Pooling node embeddings into unified molecular embedding
    GraphAttnPool(gattn_emb_size, gattn_heads), torch.nn.BatchNorm1d(gattn_heads * gattn_emb_size), nn.PReLU(),
    Linear(gattn_emb_size * gattn_heads, final_emb, bias=False),
    # Final prediction layer
    Linear(final_emb, num_classes, bias=True)
)

```

### Training

It is expected that the available data is sparse, that is that not every molecule has known activity. That's why a `GraphDataset` class returns a triplet: `(graph, active, valid)`, and a loss function masks out all invalid datapoints from loss calculation:

```python
from gatnnvs.dataset import GraphDataset, make_graph_batch
from gatnnvs.modules import ValidBCELoss

train_ds = GraphDataset(train_data, target_columns=targets)
train_loader = torch.utils.data.DataLoader(
    train_ds, drop_last=True, batch_size=256, shuffle=True, collate_fn=make_graph_batch)

loss_fn = ValidBCELoss()

for batch in train_loader:
    optim.zero_grad()

    g, a, v = batch
    g.to(device)
    a = a.to(device)
    v = v.to(device)

    out = net(g)
    loss = loss_fn(out, {'active': a, 'valid': v})
    loss.backward()
    optim.step()
```

Look at the [CLI training section](#training-your-own-model) for description of the expected data format.

### Stochastic weight averaging

[Stochastic weight averaging](https://arxiv.org/abs/1803.05407) is a simple procedure that improves generalization:

```python
from gatnnvs.swa import last_n_swa

if epoch > cfg.train.start_swa:

    # Average previous N checkpoints
    swa_chk = last_n_swa(cfg.train.n_swa, epoch, checkpoints_dir, device='cpu')
    
    # Evaluate and save SWA weights
    net.load_state_dict(swa_chk['model'])
    net = net.to(device)
    run_eval(cfg, net, loss_fn, swa_writer, epoch, i, eval_loader, label='SWA')
    torch.save(swa_chk, checkpoints_dir / f'{epoch}.last_{cfg.train.n_swa}.torch')
    
    # Continue training from the last saved checkpoint
    net.load_state_dict(torch.load(checkpoints_dir / f'{epoch}.torch')['model'])
```

### LambW optimizer

Changes weight decay handling of Lamb optimizer similar to, analog to Adam -> AdamW. In some tests, it was slightly better than AdamW, although not used for final model.

```python
from gatnnvs.optim import LambW
optim = LambW(net.parameters())
```

## License

Code and pretrained models are licenced under [GNU GPL v3 License](LICENSE).

## Citation

If you use this code or our pretrained models for your publication, please cite the original paper:

    @article{doi:10.1021/acs.jcim.0c00622,
        author = {Stojanović, Luka and Popović, Miloš and Tijanić, Nebojša and Rakočević, Goran and Kalinić, Marko},
        title = {Improved Scaffold Hopping in Ligand-Based Virtual Screening Using Neural Representation Learning},
        journal = {Journal of Chemical Information and Modeling},
        year = {2020},
        doi = {10.1021/acs.jcim.0c00622},
        note ={PMID: 32786700},
        URL = { 
            https://doi.org/10.1021/acs.jcim.0c00622
        },
        eprint = { 
            https://doi.org/10.1021/acs.jcim.0c00622
        
        }
    }
