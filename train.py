import random
import time
import torch
from copy import deepcopy
import torch.nn.functional as F
import torch.nn as nn
from colorama import init, Fore, Back, Style
import numpy as np
from tqdm import tqdm
from data_all import getattr_d, get_dataset_or_loader
from data_unit.utils import blind_other_gpus
from models import LogReg, GRLC_GCN_test
from munkres import Munkres
from sklearn import metrics
from tensorboardX import SummaryWriter
import os
import argparse
from ruamel.yaml import YAML
from termcolor import cprint


def get_args_key(args):
    return "-".join([args.model_name, args.dataset_name, args.custom_key])


def get_args(model_name, dataset_class, dataset_name, custom_key="", yaml_path=None) -> argparse.Namespace:
    yaml_path = yaml_path or os.path.join(os.path.dirname(os.path.realpath(__file__)), "args.yaml")
    custom_key = custom_key.split("+")[0]
    parser = argparse.ArgumentParser(description='Parser for Supervised Graph Attention Networks')
    # Basics
    parser.add_argument("--m", default="", type=str, help="Memo")
    parser.add_argument("--num-gpus-total", default=0, type=int)
    parser.add_argument("--num-gpus-to-use", default=0, type=int)
    parser.add_argument("--black-list", default=None, type=int, nargs="+")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--model-name", default=model_name)
    parser.add_argument("--task-type", default="", type=str)
    parser.add_argument("--perf-type", default="accuracy", type=str)
    parser.add_argument("--custom-key", default=custom_key)
    parser.add_argument("--save-model", default=True)
    parser.add_argument("--verbose", default=2)
    parser.add_argument("--save-plot", default=False)
    parser.add_argument("--seed", default=0)

    # Dataset
    parser.add_argument('--data-root', default="~/graph-data", metavar='DIR', help='path to dataset')
    parser.add_argument("--dataset-class", default=dataset_class)
    parser.add_argument("--dataset-name", default=dataset_name)
    parser.add_argument("--data-sampling-size", default=None, type=int, nargs="+")
    parser.add_argument("--data-sampling-num-hops", default=None, type=int)
    parser.add_argument("--data-num-splits", default=1, type=int)
    parser.add_argument("--data-sampler", default=None, type=str)

    # Training
    parser.add_argument('--lr', '--learning-rate', default=0.0025, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--batch-size', default=128, type=int,
                        metavar='N',
                        help='mini-batch size (default: 128), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument("--loss", default=None, type=str)
    parser.add_argument("--l1-lambda", default=0., type=float)
    parser.add_argument("--l2-lambda", default=0., type=float)
    parser.add_argument("--num-layers", default=2, type=int)
    parser.add_argument("--use-bn", default=False, type=bool)
    parser.add_argument("--perf-task-for-val", default="Node", type=str)  # Node or Link

    # Early stop
    parser.add_argument("--use-early-stop", default=False, type=bool)
    parser.add_argument("--early-stop-patience", default=-1, type=int)
    parser.add_argument("--early-stop-queue-length", default=100, type=int)
    parser.add_argument("--early-stop-threshold-loss", default=-1.0, type=float)
    parser.add_argument("--early-stop-threshold-perf", default=-1.0, type=float)

    # Pretraining
    parser.add_argument("--usepretraining", default=False, type=bool)
    parser.add_argument("--total-pretraining-epoch", default=0, type=int)
    parser.add_argument("--pretraining-noise-ratio", default=0.0, type=float)

    # Baseline
    parser.add_argument("--is-link-gnn", default=False, type=bool)
    parser.add_argument("--link-lambda", default=0., type=float)

    # Test
    parser.add_argument("--val-interval", default=10)

    parser.add_argument('--dateset', type=str, default='Cora', help='')
    parser.add_argument('--UsingLiner', type=bool, default=True, help='')
    parser.add_argument('--useNewA', type=bool, default=True, help='')
    parser.add_argument('--NewATop', type=int, default=0, help='')
    parser.add_argument('--usingact', type=bool, default=True, help='')
    parser.add_argument('--notation', type=str, default=None, help='')

    # Experiment specific parameters loaded from .yamls
    with open(yaml_path) as args_file:
        args = parser.parse_args()
        args_key = "-".join([args.model_name, args.dataset_name or args.dataset_class, args.custom_key])
        try:
            parser.set_defaults(**dict(YAML().load(args_file)[args_key].items()))
        except KeyError:
            raise AssertionError("KeyError: there's no {} in yamls".format(args_key), "red")

    # Update params from .yamls
    args = parser.parse_args()
    return args


def get_important_args(_args: argparse.Namespace) -> dict:
    important_args = [
        "lr",
        "batch_size",
        "data_sampling_num_hops",
        "data_sampling_size",
        "data_sampler",
        "data_num_splits",
        "to_undirected_at_neg",
        "num_hidden_features",
        "num_layers",
        "use_bn",
        "l1_lambda",
        "l2_lambda",
        "dropout",
        "is_super_gat",
        "is_link-gnn",
        "attention_type",
        "logit_temperature",
        "use_pretraining",
        "total_pretraining_epoch",
        "pretraining_noise_ratio",
        "neg_sample_ratio",
        "edge_sampling_ratio",
        "use_early_stop",
    ]
    ret = {}
    for ia_key in important_args:
        if ia_key in _args.__dict__:
            ret[ia_key] = _args.__getattribute__(ia_key)
    return ret


def save_args(model_dir_path: str, _args: argparse.Namespace):
    if not os.path.isdir(model_dir_path):
        raise NotADirectoryError("Cannot save arguments, there's no {}".format(model_dir_path))
    with open(os.path.join(model_dir_path, "args.txt"), "w") as arg_file:
        for k, v in sorted(_args.__dict__.items()):
            arg_file.write("{}: {}\n".format(k, v))


def pprint_args(_args: argparse.Namespace):
    cprint("Args PPRINT: {}".format(get_args_key(_args)), "yellow")
    for k, v in sorted(_args.__dict__.items()):
        print("\t- {}: {}".format(k, v))


def pdebug_args(_args: argparse.Namespace, logger):
    logger.debug("Args LOGGING-PDEBUG: {}".format(get_args_key(_args)))
    for k, v in sorted(_args.__dict__.items()):
        logger.debug("\t- {}: {}".format(k, v))


class clustering_metrics():
    "from https://github.com/Ruiqi-Hu/ARGA"

    def __init__(self, true_label, predict_label):
        self.true_label = true_label
        self.pred_label = predict_label

    def clusteringAcc(self):
        # best mapping between true_label and predict label
        l1 = list(set(self.true_label))
        numclass1 = len(l1)

        l2 = list(set(self.pred_label))
        numclass2 = len(l2)
        if numclass1 != numclass2:
            print('Class Not equal, Error!!!!')
            return 0

        cost = np.zeros((numclass1, numclass2), dtype=int)
        for i, c1 in enumerate(l1):
            mps = [i1 for i1, e1 in enumerate(self.true_label) if e1 == c1]
            for j, c2 in enumerate(l2):
                mps_d = [i1 for i1 in mps if self.pred_label[i1] == c2]

                cost[i][j] = len(mps_d)

        # match two clustering results by Munkres algorithm
        m = Munkres()
        cost = cost.__neg__().tolist()

        indexes = m.compute(cost)

        # get the match results
        new_predict = np.zeros(len(self.pred_label))
        for i, c in enumerate(l1):
            # correponding label in l2:
            c2 = l2[indexes[i][1]]

            # ai is the index with label==c2 in the pred_label list
            ai = [ind for ind, elm in enumerate(self.pred_label) if elm == c2]
            new_predict[ai] = c

        acc = metrics.accuracy_score(self.true_label, new_predict)
        f1_macro = metrics.f1_score(self.true_label, new_predict, average='macro')
        precision_macro = metrics.precision_score(self.true_label, new_predict, average='macro')
        recall_macro = metrics.recall_score(self.true_label, new_predict, average='macro')
        f1_micro = metrics.f1_score(self.true_label, new_predict, average='micro')
        precision_micro = metrics.precision_score(self.true_label, new_predict, average='micro')
        recall_micro = metrics.recall_score(self.true_label, new_predict, average='micro')
        return acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro

    def evaluationClusterModelFromLabel(self):
        nmi = metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
        adjscore = metrics.adjusted_rand_score(self.true_label, self.pred_label)
        acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro = self.clusteringAcc()

        return acc, nmi, adjscore


init(autoreset=True)


def cosine_dist(x, y):
    bs1 = x.size()[0]
    bs2 = y.size()[0]
    frac_up = torch.matmul(x, y.transpose(0, 1))
    frac_down = (torch.sqrt(torch.sum(torch.pow(x, 2) + 0.00000001, 1))).view(bs1, 1).repeat(1, bs2) * \
                (torch.sqrt(torch.sum(torch.pow(y, 2) + 0.00000001, 1))).view(1, bs2).repeat(bs1, 1)
    cosine = frac_up / frac_down
    return cosine


def normalize_graph(A):
    eps = 2.2204e-16
    deg_inv_sqrt = (A.sum(dim=-1).clamp(min=0.) + eps).pow(-0.5)
    if A.size()[0] != A.size()[1]:
        A = deg_inv_sqrt.unsqueeze(-1) * (deg_inv_sqrt.unsqueeze(-1) * A)
    else:
        A = deg_inv_sqrt.unsqueeze(-1) * A * deg_inv_sqrt.unsqueeze(-2)
    return A


def get_dataset(args, dataset_kwargs):
    train_d, val_d, test_d = get_dataset_or_loader(
        args.dataset_class, args.dataset_name, args.data_root,
        batch_size=args.batch_size, seed=args.seed, num_splits=args.data_num_splits,
        **dataset_kwargs,
    )
    train_d.data.x = torch.FloatTensor(train_d.data.x)
    eps = 2.2204e-16
    norm = train_d.data.x.norm(p=1, dim=1, keepdim=True).clamp(min=0.) + eps
    train_d.data.x = train_d.data.x.div(norm.expand_as(train_d.data.x))

    i = torch.LongTensor([train_d.data.edge_index[0].numpy(), train_d.data.edge_index[1].numpy()])
    v = torch.FloatTensor(torch.ones([train_d.data.num_edges]))

    A_sp = torch.sparse.FloatTensor(i, v, torch.Size([train_d.data.num_nodes, train_d.data.num_nodes]))
    A = A_sp.to_dense()
    I = torch.eye(A.shape[1]).to(A.device)
    A_I = A + I
    A_I_nomal = normalize_graph(A_I)
    return train_d, [A_I, A_I_nomal, A_sp], [train_d.data.x]


def run_GCN(args, gpu_id=None, exp_name=None, number=0, return_model=False, return_time_series=False):
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    writename = "-" + exp_name[4:] + "_" + str(number)
    logname = os.path.join(exp_name, str(number) + "_" + str(args.seed) + ".txt")
    logfile = open(logname, 'a')
    writer = SummaryWriter(comment=writename)
    final_acc = 0
    best_acc = 0
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    running_device = "cpu" if gpu_id is None \
        else torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')
    dataset_kwargs = {}

    train_d, adj_list, x_list = get_dataset(args, dataset_kwargs)

    lable = train_d.data.y
    A_I = adj_list[0]
    A_I_nomal = adj_list[1]

    nb_edges = train_d.data.num_edges
    nb_nodes = train_d.data.num_nodes
    nb_feature = train_d.data.num_features
    nb_classes = int(lable.max() - lable.min()) + 1

    lable_matrix = (lable.view(nb_nodes, 1).repeat(1, nb_nodes) == lable.view(1, nb_nodes).repeat(nb_nodes, 1)) + 0
    I = (torch.eye(nb_nodes).to(lable_matrix.device) == 1)
    lable_matrix[I] = 0
    zero_vec = 0.0 * torch.ones_like(A_I_nomal)
    if args.dataset_name in ['Photo', 'DBLP', 'Crocodile', 'CoraFull', 'WikiCS']:
        useA = True
    else:
        useA = False
    model = GRLC_GCN_test(nb_nodes, nb_feature, args.dim,
                          dim_x=args.dim_x, useact=args.usingact, liner=args.UsingLiner,
                          dropout=args.dropout, useA=useA)

    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)

    model.to(running_device)
    lable = lable.to(running_device)
    if args.dataset_name == 'WikiCS':
        train_lbls = lable[train_d.data.train_mask[:, args.NewATop]]  # capture
        test_lbls = lable[train_d.data.test_mask]
    elif args.dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
        train_lbls = lable[train_d.data.train_mask]
        test_lbls = lable[train_d.data.test_mask]
    elif args.dataset_name in ['Photo', 'DBLP', 'Crocodile', 'CoraFull']:
        train_index = []
        test_index = []
        for j in range(lable.max().item() + 1):
            num = ((lable == j) + 0).sum().item()
            index = torch.range(0, len(lable) - 1)[(lable == j)]
            x_list0 = random.sample(list(index), int(len(index) * 0.1))
            for x in x_list0:
                train_index.append(int(x))
        for c in range(len(lable)):
            if int(c) not in train_index:
                test_index.append(int(c))
        train_lbls = lable[train_index]
        test_lbls = lable[test_index]
        val_lbls = lable[train_index]

    A_I_nomal_dense = A_I_nomal
    I_input = torch.eye(A_I_nomal.shape[1])  # .to(A_I_nomal.device)
    if args.dataset_name in ['PubMed', 'CoraFull', 'DBLP']:
        pass
    elif args.dataset_name in ['Crocodile', 'Photo', 'WikiCS']:
        A_I_nomal_dense = A_I_nomal_dense.to(running_device)
    ######################sparse################
    if args.dataset_name in ['PubMed', 'Crocodile', 'CoraFull', 'DBLP', 'Photo', 'WikiCS']:
        A_I_nomal = A_I_nomal.to_sparse()
        model.sparse = True
        I_input = I_input.to_sparse()
    ######################sparse################
    A_I_nomal = A_I_nomal.to(running_device)
    I_input = I_input.to(A_I_nomal.device)
    mask_I = I.to(running_device)
    zero_vec = zero_vec.to(running_device)
    my_margin = args.margin1
    my_margin_2 = my_margin + args.margin2
    margin_loss = torch.nn.MarginRankingLoss(margin=my_margin, reduce=False)
    num_neg = args.NN
    print(num_neg)
    if args.usepretraining:
        if os.path.exists(args.checkpoint_dir + '/' + args.dataset_name + '_weights.pth'):
            load_params = torch.load(args.checkpoint_dir + '/' + args.dataset_name + '_weights.pth')
            model_params = model.state_dict()
            same_parsms = {k: v for k, v in load_params.items() if k in model_params.keys()}
            model_params.update(same_parsms)
            model.load_state_dict(model_params)
            model.eval()
            feature_X = x_list[0].to(running_device)
            lbl_z = torch.tensor([0.]).to(running_device)
            feature_a = feature_X
            feature_p = feature_X
            feature_n = []
            idx_list = []
            idx_lable = []
            for i in range(num_neg):
                idx_0 = np.random.permutation(nb_nodes)
                idx_list.append(idx_0)
                idx_lable.append(lable[idx_0])
                feature_temp = feature_X[idx_0]
                feature_n.append(feature_temp)
            h_a, h_p = model.embed(feature_a, feature_p, feature_n, A_I_nomal, I=I_input)
            if args.useNewA:
                embs = h_p
            else:
                embs = h_a
            if args.dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
                embs = embs / embs.norm(dim=1)[:, None]

            if args.dataset_name == 'WikiCS':
                train_embs = embs[train_d.data.train_mask[:, args.NewATop]]
                test_embs = embs[train_d.data.test_mask]
            elif args.dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
                train_embs = embs[train_d.data.train_mask]
                test_embs = embs[train_d.data.test_mask]
            elif args.dataset_name in ['Photo', 'DBLP', 'Crocodile', 'CoraFull']:
                train_embs = embs[train_index]
                test_embs = embs[test_index]
            accs = []
            accs_small = []
            xent = nn.CrossEntropyLoss()
            for _ in range(2):
                log = LogReg(args.dim, nb_classes)
                opt = torch.optim.Adam(log.parameters(), lr=1e-2, weight_decay=args.wd)
                log.to(running_device)
                for _ in range(args.num1):
                    log.train()
                    opt.zero_grad()
                    logits = log(train_embs)
                    loss = xent(logits, train_lbls)
                    loss.backward()
                    opt.step()
                logits = log(test_embs)
                preds = torch.argmax(logits, dim=1)
                acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
                accs.append(acc * 100)
                ac = []
                for i in range(nb_classes):
                    acc_small = torch.sum(preds[test_lbls == i] == test_lbls[test_lbls == i]).float() / \
                                test_lbls[test_lbls == i].shape[0]
                    ac.append(acc_small * 100)
                accs_small = ac
            accs = torch.stack(accs)
            string_3 = ""
            for i in range(nb_classes):
                string_3 = string_3 + "|{:.1f}".format(accs_small[i].item())
            string_2 = Fore.GREEN + "accs: {:.1f},std: {:.2f} ".format( accs.mean().item(),
                                                                                  accs.std().item())
            tqdm.write(string_2 + string_3)
            final_acc = accs.mean().item()
            best_acc = max(best_acc, final_acc)
        return final_acc, best_acc
    for current_iter, epoch in enumerate(tqdm(range(args.start_epoch, args.start_epoch + args.epochs + 1))):
        model.train()
        optimiser.zero_grad()
        idx = np.random.permutation(nb_nodes)
        feature_X = x_list[0].to(running_device)
        lbl_z = torch.tensor([0.]).to(running_device)
        feature_a = feature_X
        feature_p = feature_X
        feature_n = []
        idx_list = []
        idx_lable = []
        for i in range(num_neg):
            idx_0 = np.random.permutation(nb_nodes)
            idx_list.append(idx_0)
            idx_lable.append(lable[idx_0])
            feature_temp = feature_X[idx_0]
            feature_n.append(feature_temp)
        h_a, h_p, h_n_lsit, h_a_0, h_p_0, h_n_0_list = model(feature_a, feature_p, feature_n, A_I_nomal, I=I_input)
        s_p = F.pairwise_distance(h_a, h_p)
        cos_0_list = []
        for h_n_0 in h_n_0_list:
            cos_0 = F.pairwise_distance(h_a_0, h_n_0)
            cos_0_list.append(cos_0)
        cos_0_stack = torch.stack(cos_0_list).detach()
        cos_0_min = cos_0_stack.min(dim=0)[0]
        cos_0_max = cos_0_stack.max(dim=0)[0]
        gap = cos_0_max - cos_0_min
        weight_list = []
        for i in range(cos_0_stack.size()[0]):
            weight_list.append((cos_0_stack[i] - cos_0_min) / gap)
        s_n_list = []
        s_n_cosin_list = []
        for h_n in h_n_lsit:
            if args.dataset_name in ['Cora', 'CiteSeer']:
                s_n_cosin_list.append(cosine_dist(h_a, h_n)[mask_I].detach())
            s_n = F.pairwise_distance(h_a, h_n)
            s_n_list.append(s_n)
        margin_label = -1 * torch.ones_like(s_p)
        loss_mar = 0
        mask_margin_N = 0
        i = 0
        for s_n in s_n_list:
            loss_mar += (margin_loss(s_p, s_n, margin_label) * weight_list[i]).mean()
            mask_margin_N += torch.max((s_n - s_p.detach() - my_margin_2), lbl_z).sum()
            i += 1
        mask_margin_N = mask_margin_N / num_neg
        string_1 = " loss_1: {:.3f}||loss_2: {:.3f}||".format(loss_mar.item(), mask_margin_N.item())
        loss = loss_mar * args.w_loss1 + mask_margin_N * args.w_loss2 / nb_nodes
        if args.dataset_name in ['Cora']:
            loss = loss_mar * args.w_loss1 + mask_margin_N * args.w_loss2
        loss.backward()
        optimiser.step()

        model.eval()
        if args.dataset_name in ['Crocodile', 'WikiCS', 'Photo']:
            h_p_d = h_p.detach()
            S_new = cosine_dist(h_p_d, h_p_d)
            model.A = normalize_graph(torch.mul(S_new, A_I_nomal_dense)).to_sparse()
        elif args.dataset_name in ['Cora', 'CiteSeer']:
            h_a, h_p = model.embed(feature_a, feature_p, feature_n, A_I_nomal, I=I_input)
            s_a = cosine_dist(h_a, h_a).detach()
            S = (torch.stack(s_n_cosin_list).mean(dim=0).expand_as(A_I) - s_a).detach()
            # zero_vec = -9e15 * torch.ones_like(S)
            one_vec = torch.ones_like(S)
            s_a = torch.where(A_I_nomal > 0, one_vec, zero_vec)
            attention = torch.where(S < 0, s_a, zero_vec)
            attention_N = normalize_graph(attention)
            attention[I] = 0
            model.A = attention_N

        if epoch % 50 == 0:
            model.eval()
            h_a, h_p = model.embed(feature_a, feature_p, feature_n, A_I_nomal, I=I_input)
            if args.useNewA:
                embs = h_p  # torch.cat((h_a,h_p),dim=1)
            else:
                embs = h_a
            if args.dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
                embs = embs / embs.norm(dim=1)[:, None]

            if args.dataset_name == 'WikiCS':
                train_embs = embs[train_d.data.train_mask[:, args.NewATop]]
                test_embs = embs[train_d.data.test_mask]
            elif args.dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
                train_embs = embs[train_d.data.train_mask]
                test_embs = embs[train_d.data.test_mask]
            elif args.dataset_name in ['Photo', 'DBLP', 'Crocodile', 'CoraFull']:
                train_embs = embs[train_index]
                test_embs = embs[test_index]

            accs = []
            accs_small = []
            xent = nn.CrossEntropyLoss()
            for _ in range(2):
                log = LogReg(args.dim, nb_classes)
                opt = torch.optim.Adam(log.parameters(), lr=1e-2, weight_decay=args.wd)
                log.to(running_device)
                for _ in range(args.num1):
                    log.train()
                    opt.zero_grad()
                    logits = log(train_embs)
                    loss = xent(logits, train_lbls)
                    loss.backward()
                    opt.step()
                logits = log(test_embs)
                preds = torch.argmax(logits, dim=1)
                acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
                accs.append(acc * 100)
                ac = []
                for i in range(nb_classes):
                    acc_small = torch.sum(preds[test_lbls == i] == test_lbls[test_lbls == i]).float() / \
                                test_lbls[test_lbls == i].shape[0]
                    ac.append(acc_small * 100)
                accs_small = ac
            accs = torch.stack(accs)
            string_3 = ""
            for i in range(nb_classes):
                string_3 = string_3 + "|{:.1f}".format(accs_small[i].item())
            string_2 = Fore.GREEN + " epoch: {},accs: {:.1f},std: {:.2f} ".format(epoch, accs.mean().item(),
                                                                                  accs.std().item())
            tqdm.write(string_1 + string_2 + string_3)
            final_acc = accs.mean().item()
            best_acc = max(best_acc, final_acc)
    return final_acc, best_acc


def run_with_many_seeds(args, num_seeds, gpu_id=None, name=None, **kwargs):
    results_acc = []
    results_best = []
    results_epoch = []
    for i in range(num_seeds):
        cprint("## TRIAL {} ##".format(i), "yellow")
        _args = deepcopy(args)
        _args.seed = _args.seed + random.randint(0, 100)
        acc, best = run_GCN(_args, gpu_id=gpu_id, number=i, exp_name=name, **kwargs)
        results_acc.append(torch.as_tensor(acc, dtype=torch.float32))
        results_best.append(torch.as_tensor(best, dtype=torch.float32))
    results_acc = torch.stack(results_acc)
    return results_acc.mean().item()


if __name__ == '__main__':

    num_total_runs = 10
    main_args = get_args(
        model_name="GRLC",
        dataset_class="Planetoid",
        dataset_name="Cora",
        custom_key="classification",
    )
    pprint_args(main_args)
    filePath = "log"
    exp_ID = 0
    for filename in os.listdir(filePath):
        file_info = filename.split("_")
        file_dataname = file_info[0]
        if file_dataname == main_args.dataset_name:
            exp_ID = max(int(file_info[1]), exp_ID)
    exp_name = main_args.dataset_name + "_" + str(exp_ID + 1)
    exp_name = os.path.join(filePath, exp_name)
    os.makedirs(exp_name)
    if len(main_args.black_list) == main_args.num_gpus_total:
        alloc_gpu = [None]
        cprint("Use CPU", "yellow")
    else:
        alloc_gpu = blind_other_gpus(num_gpus_total=main_args.num_gpus_total,
                                     num_gpus_to_use=main_args.num_gpus_to_use,
                                     black_list=main_args.black_list)
        if not alloc_gpu:
            alloc_gpu = [int(np.random.choice([g for g in range(main_args.num_gpus_total)
                                               if g not in main_args.black_list], 1))]
        cprint("Use GPU the ID of which is {}".format(alloc_gpu), "yellow")

    # noinspection PyTypeChecker

    # main_args.ir = 0.01
    many_seeds_result = run_with_many_seeds(main_args, num_total_runs, gpu_id=alloc_gpu[0], name=exp_name)
    newname = exp_name + "_" + '%.2f' % many_seeds_result
    if main_args.notation:
        newname = newname + "_" + main_args.notation
    os.rename(exp_name, newname)
