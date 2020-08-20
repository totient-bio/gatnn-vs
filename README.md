# Graph Attention Neural Network for Virtual Screening

GATNN-VS is a neural network architecture, and a learning framework designed to yield a generally applicable tool for ligand-based virtual screening. Our approach uses the molecular graph as input, and involves learning a representation that places compounds of similar biological profiles in close proximity within a hyperdimensional feature space; this is achieved by simultaneously leveraging historical screening data against a multitude of targets during training. Cosine distance between molecules in this space becomes a general similarity metric, and can readily be used to rank order database compounds in LBVS workflows

Checkout our paper: [Improved Scaffold Hopping in Ligand-based Virtual Screening Using Neural Representation Learning](https://pubs.acs.org/doi/10.1021/acs.jcim.0c00622)

![GATTN Virtual Screen Visualization](https://github.com/totient-bio/gatnn-vs/raw/master/images/gatnn-screening.png)

## Instalation

Project requires python 3.6+ and uses following dependencies:

* *pytorch*
* *dgl* - for graph handlind
* *rdkit* - molecule processing
* Several utility packages: *tqdm*, *hydra-core*, *omegaconf*, *pandas*, *cytoolz*, *feather-format*

Please make sure the first three dependencies are installed and working; just pip installing them probably isn't enough (rdkit doesn't even have a pip package). The easyest way would be through [conda](https://www.anaconda.com/products/individual) package manager.

Once the environment is set up, you can install GATNN-VS with pip:

    $ pip install git+https://github.com/totient-bio/gatnn-vs.git

## Pretrained model

You can download [pretrained model](https://totient-pretrained.s3.amazonaws.com/gatnn-vs/gatnn-model-final.tar.gz)

## Usage

### Command line

All command line tools use omegaconf for parsing command line as well as the configuration. That way you can easily override any config parameter, e.g. for training.

To use trained model for screening, first you should create a fingerprint database:

    $ python3 -m gatnnvs.embed INPUT_FILE [--model MODEL_PATH] [--output OUTPUT_PATH] [--device DEVICE]

Input should be a `csv` file with 'smiles' column. Molecules must consist only from C, N, O, S, Cl, F, Br, P and I atoms (and implicit Hs). If compound is a salt, only the largest component will be used. Output will be a `npy` file. Lines from the input file will correspond with rows in generated numpy array. If rdkit can't parse smiles, or molecule contains elements that are not accepted, appropriate rows will be filled with `nan`s. Single GPU can compute roughly 3000 embeddings/sec, so for 1-2 million compounds this step should be done in minutes.

When fingerprint database is computed, you can use it to screen molecules:

    TODO

### Library



## Training

You can train a model yourself with:

    $ python3 -m gatnnvs.train data=DATA rundir=RUNDIR

Input shoud be a feather file with smiles column and target columns. You can control what columns are considered 'target' by specifying `target_prefix` option. Target columns should have values 'Active', 'Inactive' and 'Unknown'. For large datasets, it's best if target columns are of `category` datatype.

Train command uses hydra-core / omegaconf for command line parsin / sonfiguration. You can get a full 

## License

Code and pretrained models are licenced under [GNU GPL v3 License](LICENSE).

## Citation

If you use this code or our pretrained models for your publication, please cite the original paper:

    @article{doi:10.1021/acs.jcim.0c00622,
        author = {Stojanović, Luka and Popović, Miloš and Tijanić, Nebojša and Rakočević, Goran and Kalinić, Marko},
        title = {Improved Scaffold Hopping in Ligand-Based Virtual Screening Using Neural Representation Learning},
        journal = {Journal of Chemical Information and Modeling},
        volume = {0},
        number = {0},
        pages = {null},
        year = {0},
        doi = {10.1021/acs.jcim.0c00622},
        note ={PMID: 32786700},
        URL = { 
            https://doi.org/10.1021/acs.jcim.0c00622
        },
        eprint = { 
            https://doi.org/10.1021/acs.jcim.0c00622
        
        }
    }
