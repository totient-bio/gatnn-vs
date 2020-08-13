import logging
import hydra
import torch
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from rdkit.Chem import AllChem
import torch.utils.tensorboard as tb

from gatnnvs.model import build_model, noreg
from gatnnvs.optim import LambW
from gatnnvs.dataset import GraphDataset, make_graph_batch
from gatnnvs.modules import ValidBCELoss
from gatnnvs.metrics import get_metrics
from gatnnvs.scheduler import CyclicLR
from gatnnvs.swa import last_n_swa

log = logging.getLogger(__name__)


def desalt_smiles(smi):
    return max(smi.split('.'), key=len) if '.' in smi else smi


def run_eval(cfg, net, loss_fn, tb_writer, epoch, iteration, eval_loader, label='Eval'):
    i = iteration
    device = next(net.parameters()).device
    net.eval()
    with torch.no_grad():
        outs = []
        actives = []
        valids = []
        for batch in tqdm(eval_loader, desc=f'{label} epoch: {epoch}', leave=False):
            g, a, v = batch
            g.to(device)
            a = a.to(device)
            v = v.to(device)
            out = net(g)
            
            outs.append(out)
            actives.append(a)
            valids.append(v)
        
        outs = torch.cat(outs)
        actives = torch.cat(actives)
        valids = torch.cat(valids)
        loss = loss_fn(outs, {'active': actives, 'valid': valids})
        metrics, histograms = get_metrics(
            outs.cpu().numpy(), actives.cpu().numpy(), valids.cpu().numpy()
        )
        tb_writer.add_scalar('loss', loss.item(), i)
        for k, v in metrics.items():
            tb_writer.add_scalar(k, v, global_step=i)
        for k, v in histograms.items():
            tb_writer.add_histogram(k, v, global_step=i)
        tb_writer.file_writer.flush()
        

def run(cfg):
    cycle_epochs = cfg.train.cycle.epochs_up + cfg.train.cycle.epochs_down

    ## Load dataset
    data = pd.read_feather(cfg.data)
    rng = np.random.RandomState(seed=cfg.seed)
    is_eval = rng.rand(len(data)) > cfg.train_split
    targets = [c for c in data.columns if c.startswith(cfg.targets_prefix)]
    if len(targets) < cfg.num_classes:
        raise(Exception(f'Not enough targets in dataset: got {len(targets)}, expected at least {cfg.num_classes}'))
    if len(targets) > cfg.num_classes:
        targets = rng.choice(targets, cfg.num_classes)
    
    ## Build model and load latest checkpoint if resuming
    i, start_epoch = -1, 1
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

    regs = [p for p in itertools.chain.from_iterable(x.parameters() for x in net if not noreg(x))]
    noregs = [p for p in itertools.chain.from_iterable(x.parameters() for x in net if noreg(x))]
    params = [
        {'params': regs, 'lr': cfg.train.lr, 'weight_decay': cfg.train.wd.pre_cycle}, #'betas':[0.99, 0.95]},
        {'params': noregs, 'lr': cfg.train.lr, 'weight_decay': 0}, # 'betas': [0.99, 0.95]}
    ]
    optim = {
        'LambW': LambW,
        'AdamW': torch.optim.AdamW,
    }[cfg.train.optim](params)
    
    checkpoints_dir = Path('checkpoints')
    if not checkpoints_dir.is_dir():
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
    else:
        last = sorted([int(c.name.split('.')[0]) for c in checkpoints_dir.glob('*.torch') if len(c.name.split('.')) == 2])[-1]
        log.info(f'Loading checkpoint for epoch {last}')
        checkpoint = torch.load(str(checkpoints_dir / f'{last}.torch'))
        net.load_state_dict(checkpoint['model'])
        optim.load_state_dict(checkpoint['optim'])
        i = checkpoint['iteration']
        start_epoch = checkpoint['epoch'] + 1
        if 'batch_size' in checkpoint:
            assert cfg.batch_size == checkpoint['batch_size']

    data['mol'] = data.smiles.map(lambda s: AllChem.MolFromSmiles(desalt_smiles(s)))
    train_data = data[~is_eval].reset_index(drop=True)
    eval_data = data[is_eval].reset_index(drop=True)
    assert (len(train_data) + len(eval_data)) == len(data)
    train_ds = GraphDataset(train_data, target_columns=targets)
    eval_ds = GraphDataset(eval_data, target_columns=targets)
    train_loader = torch.utils.data.DataLoader(
        train_ds, drop_last=True, batch_size=cfg.batch_size, shuffle=True, num_workers=4, collate_fn=make_graph_batch)
    eval_loader = torch.utils.data.DataLoader(
        eval_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4, collate_fn=make_graph_batch)


    train_log = Path('train')
    eval_log = Path('eval')
    swa_log = Path('swa')
    train_log.mkdir(parents=True, exist_ok=True)
    eval_log.mkdir(parents=True, exist_ok=True)
    swa_log.mkdir(parents=True, exist_ok=True)

    train_writer = tb.SummaryWriter(log_dir=str(train_log))
    eval_writer = tb.SummaryWriter(log_dir=str(eval_log))
    swa_writer = tb.SummaryWriter(log_dir=str(swa_log))
    
    if cfg.train.cycle.enabled and start_epoch <= cycle_epochs:
        scheduler = CyclicLR(optim, 
                             base_lr=cfg.train.cycle.base_lr,
                             max_lr=cfg.train.cycle.max_lr,
                             cycle_momentum=cfg.train.cycle.momentum,
                             step_size_up=cfg.train.cycle.epochs_up * len(train_loader),
                             step_size_down=cfg.train.cycle.epochs_down * len(train_loader),
                             last_epoch=i)

    def cycle_step(epoch):
        if epoch <= cycle_epochs and cfg.train.cycle.enabled:
            scheduler.step()

        if epoch == cycle_epochs + 1:
            optim.param_groups[0]['weight_decay'] = cfg.train.wd.post_cycle
            optim.param_groups[0]['lr'] = cfg.train.lr
            optim.param_groups[1]['lr'] = cfg.train.lr  

    loss_fn = ValidBCELoss(pos_weight=torch.full((cfg.num_classes,), cfg.train.pos_class_weight))
    device = next(net.parameters()).device
    loss_fn.to(device)

    for epoch in tqdm(range(start_epoch, start_epoch + cfg.train.epochs)):
        train_writer.add_scalar('Optim/LR', optim.param_groups[0]['lr'], i)
        train_writer.add_scalar('Optim/WD', optim.param_groups[0]['weight_decay'], i)
        net.train()
        for batch in tqdm(train_loader, desc=f'Epoch: {epoch}', leave=False):
            i += 1

            optim.zero_grad()

            g, a, v = batch
            g.to(device)
            a = a.to(device)
            v = v.to(device)

            out = net(g)
            loss = loss_fn(out, {'active': a, 'valid': v})
            loss.backward()
            optim.step()
            cycle_step(epoch)

            if i > 0 and ((i % cfg.train.metric_freq == 0) or ((i+1) % len(train_loader) == 0)):
                train_writer.add_scalar('loss', loss.item(), i)
                train_writer.file_writer.flush()
        run_eval(cfg, net, loss_fn, eval_writer, epoch, i, eval_loader)
        torch.save({
            'model': net.state_dict(),
            'optim': optim.state_dict(),
            'iteration': i,
            'epoch': epoch,
            'batch_size': cfg.batch_size,
        }, checkpoints_dir / f'{epoch}.torch')
    
        if epoch < cfg.train.start_swa:
            continue

        swa_chk = last_n_swa(cfg.train.n_swa, epoch, checkpoints_dir, device='cpu')
        assert swa_chk['epoch'] == epoch
        net.load_state_dict(swa_chk['model'])
        net = net.to(device)
        run_eval(cfg, net, loss_fn, swa_writer, epoch, i, eval_loader, label='SWA')
        torch.save(swa_chk, checkpoints_dir / f'{epoch}.last_{cfg.train.n_swa}.torch')
        net.load_state_dict(torch.load(checkpoints_dir / f'{epoch}.torch')['model'])

        to_del = checkpoints_dir / f'{epoch-cfg.train.keep_swa}.last_{cfg.train.n_swa}.torch'
        if to_del.exists():
            to_del.unlink()

        to_del = checkpoints_dir / f'{epoch-cfg.train.n_swa}.torch'
        if to_del.exists():
            to_del.unlink()

    train_writer.close()
    eval_writer.close()
    swa_writer.close()
            
@hydra.main(config_path="../config/train-config.yaml")
def main(cfg):
    log.info(cfg.pretty())
    run(cfg)
    

if __name__ == "__main__":
    main()
