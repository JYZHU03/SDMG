
import numpy as np

import argparse

import shutil
import time
import os.path as osp

import dgl

from dgl.dataloading import GraphDataLoader

import torch
from torch.utils.data.sampler import SubsetRandomSampler


from utils.utils import (create_optimizer, create_pooler, set_random_seed, compute_ppr)

from datasets.data_util import load_graph_classification_dataset

from models import SDMG



from utils import comm
from utils.logger import setup_logger
from utils.misc import mkdir

from evaluator import graph_classification_evaluation
import yaml
from easydict import EasyDict as edict
import torch.nn.functional as F
from models.positional_encoding import get_rw_landing_probs


parser = argparse.ArgumentParser(description='Graph DGL Training')
parser.add_argument('--resume', '-r', action='store_true', default=False, help='resume from checkpoint')
parser.add_argument("--local_rank", type=int, default=0, help="local rank")
parser.add_argument("--seed", type=int, default=1234, help="random seed")
parser.add_argument("--yaml_dir", type=str, default='./yamls/MUTAG.yaml')
parser.add_argument("--output_dir", type=str, default='./output')
parser.add_argument("--checkpoint_dir", type=str, default='./ckpt')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
args = parser.parse_args()


def pretrain(model, train_loader, optimizer, device, epoch, logger=None):
    model.train()
    loss_list = []
    for batch in train_loader:
        batch_g, _ = batch
        batch_g = batch_g.to(device)
        feat = batch_g.ndata["attr"]
        # feat_mask = batch_g.ndata["attr_mask"]
        loss, loss_dict = model(batch_g, feat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
    lr = optimizer.param_groups[0]['lr']
    logger.info(f"Epoch {epoch} | train_loss: {np.mean(loss_list):.4f} | lr: {lr:.6f}")


def collate_fn(batch):
    graphs = [x[0] for x in batch]
    labels = [x[1] for x in batch]
    batch_g = dgl.batch(graphs)
    labels = torch.cat(labels, dim=0)
    return batch_g, labels


def save_checkpoint(state, is_best, filename):
    ckp = osp.join(filename, 'checkpoint.pth.tar')
    # ckp = filename + "checkpoint.pth.tar"
    torch.save(state, ckp)
    if is_best:
        shutil.copyfile(ckp, filename+'/model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch, alpha, decay, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 80 epochs"""
    lr = lr * (alpha ** (epoch // decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main(cfg):
    best_acc = float('-inf')
    best_acc_epoch = float('inf')

    if cfg.output_dir:
        mkdir(cfg.output_dir)
        mkdir(cfg.checkpoint_dir)

    logger = setup_logger("graph", cfg.output_dir, comm.get_rank(), filename='train_log.txt')
    logger.info("Rank of current process: {}. World size: {}".format(comm.get_rank(), comm.get_world_size()))
    # logger.info("Environment info:\n" + collect_env_info())
    logger.info("Command line arguments: " + str(args))

    # shutil.copyfile('./params.yaml', cfg.output_dir + '/params.yaml')
    # shutil.copyfile('./main_graph.py', cfg.output_dir + '/graph.py')
    # shutil.copyfile('./models/SDMG.py', cfg.output_dir + '/SDMG.py')
    # shutil.copyfile('./models/mlp_gat.py', cfg.output_dir + '/mlp_gat.py')

    graphs, (num_features, num_classes) = load_graph_classification_dataset(cfg.DATA.data_name,
                                                                            deg4feat=cfg.DATA.deg4feat,
                                                                            PE=False)
    ksteps = list(range(1, cfg.MODEL.RW_step))
    for graph, label in graphs:

        src, dst = graph.edges()
        edge_index = torch.stack([src, dst], dim=0)
        num_nodes = graph.num_nodes()
        adj_pos = get_rw_landing_probs(ksteps=ksteps, edge_index=edge_index, num_nodes=num_nodes)
        adj_pos = F.layer_norm(adj_pos, (adj_pos.shape[-1],))
        graph.ndata["adj_pos"] = adj_pos

    cfg.num_features = num_features

    train_idx = torch.arange(len(graphs))
    train_sampler = SubsetRandomSampler(train_idx)
    train_loader = GraphDataLoader(graphs, sampler=train_sampler, collate_fn=collate_fn,
                                   batch_size=cfg.DATALOADER.BATCH_SIZE, pin_memory=True)
    eval_loader = GraphDataLoader(graphs, collate_fn=collate_fn, batch_size=len(graphs), shuffle=False)

    pooler = create_pooler(cfg.MODEL.pooler)

    acc_list = []
    for i, seed in enumerate(cfg.seeds):
        logger.info(f'Run {i}th for seed {seed}')
        set_random_seed(seed)

        ml_cfg = cfg.MODEL
        ml_cfg.update({'in_dim': num_features})
        model = SDMG(**ml_cfg)
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info('Total trainable params num : {}'.format(total_trainable_params))
        model.to(cfg.DEVICE)

        optimizer = create_optimizer(cfg.SOLVER.optim_type, model, cfg.SOLVER.LR, cfg.SOLVER.weight_decay)

        start_epoch = 0
        if args.resume:
            if osp.isfile(cfg.pretrain_checkpoint_dir):
                logger.info("=> loading checkpoint '{}'".format(cfg.checkpoint_dir))
                checkpoint = torch.load(cfg.checkpoint_dir, map_location=torch.device('cpu'))
                start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                logger.info("=> loaded checkpoint '{}' (epoch {})"
                            .format(cfg.checkpoint_dir, checkpoint['epoch']))

        logger.info("----------Start Training----------")

        for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
            adjust_learning_rate(optimizer, epoch=epoch, alpha=cfg.SOLVER.alpha, decay=cfg.SOLVER.decay, lr=cfg.SOLVER.LR)
            pretrain(model, train_loader, optimizer, cfg.DEVICE, epoch, logger)
            if ((epoch + 1) % 1 == 0) & (epoch > cfg.eva_epoch):
                model.eval()
                test_acc = graph_classification_evaluation(model, cfg.eval_T, pooler, eval_loader,
                                                          cfg.DEVICE, logger)
                is_best = test_acc > best_acc
                if is_best:
                    best_acc_epoch = epoch
                best_acc = max(test_acc, best_acc)
                logger.info(f"Epoch {epoch}: get test acc score: {test_acc: .3f}")
                logger.info(f"best_acc {best_acc:.3f} at epoch {best_acc_epoch}")
                save_checkpoint({'epoch': epoch + 1,
                                 'state_dict': model.state_dict(),
                                 'best_acc': best_acc,
                                 'optimizer': optimizer.state_dict()},
                                is_best, filename=cfg.checkpoint_dir)
        acc_list.append(best_acc)
    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    logger.info((f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}"))
    return final_acc


if __name__ == "__main__":
    root_dir = osp.abspath(osp.dirname(__file__))
    yaml_dir = osp.join(root_dir, 'params.yaml')
    output_dir = osp.join(root_dir, 'log')
    checkpoint_dir = osp.join(output_dir, "checkpoint")

    yaml_dir = args.yaml_dir if args.yaml_dir else yaml_dir
    output_dir = args.output_dir if args.output_dir else output_dir
    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir else checkpoint_dir

    with open(yaml_dir, "r") as f:
        config = yaml.load(f, yaml.FullLoader)
    cfg = edict(config)

    cfg.output_dir, cfg.checkpoint_dir = output_dir, checkpoint_dir
    print(cfg)
    acc = main(cfg)













