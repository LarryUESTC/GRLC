import numpy as np
import scipy.sparse as sp
import torch
import sys
import pickle as pkl
import networkx as nx
from scipy.io import loadmat
# from normalization import fetch_normalization, row_normalize
import torch.nn.functional as F
from time import perf_counter
# from knn import knn_hnsw,knn_faiss
import random
from sklearn.cluster import KMeans
# from metrics import cosine_dist
import os
import json
eps = 2.2204e-16
def normalize_graph(A):
    deg_inv_sqrt = A.sum(dim=-1).clamp(min=0.).pow(-0.5)
    A = deg_inv_sqrt.unsqueeze(-1) * A * deg_inv_sqrt.unsqueeze(-2)
    return A

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def preprocess_citation(adj, features, normalization="FirstOrderGCN"):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    # features = row_normalize(features)
    return adj, features

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def load_citation(dataset_str="cora", normalization="AugNormAdj", cuda=True):
    """
    Load Citation Networks Datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    adj, features = preprocess_citation(adj, features, normalization)

    # porting to pytorch
    features = torch.FloatTensor(np.array(features.todense())).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    return adj, features, labels, idx_train, idx_val, idx_test

def load_citation_kongfi(dataset_str="cora", normalization="AugNormAdj", cuda=True,rate=80):

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)
    if dataset_str == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    size = labels.shape[0]
    k = int(size * rate / 100)
    idx_train = range(k)
    idx_test = range(k, size)
    idx_val = range(k, size)
    adj, features = preprocess_citation(adj, features, normalization)
    # porting to pytorch
    features = torch.FloatTensor(np.array(features.todense())).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    return adj, features, labels, idx_train, idx_val, idx_test


def load_citation_out(dataset_str="cora", normalization="AugNormAdj", cuda=True,  rate=10):
    """
    Load Citation Networks Datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    # features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    # idx_test = test_idx_range.tolist()
    size = labels.shape[0]
    k = int(size * rate / 100 )
    idx_train = range(k)
    idx_test = range(k, size)
    idx_val = range(k, size)

    # adj, features = preprocess_citation(adj, features, normalization)

    # porting to pytorch
    features = torch.FloatTensor(np.array(features.todense())).float()
    features = F.normalize(features)
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    adj = adj.to_dense()+torch.eye(size)
    adj[:k][:, k:] = 0.0
    adj = adj / adj.sum(dim=1)
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    features_train = features[:k]
    adj_train = adj[:k,:k]



    if cuda:
        features = features.cuda()
        features_train = features_train.cuda()
        adj = adj.cuda()
        adj_train = adj_train.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    return adj, adj_train, features, features_train, labels, idx_train, idx_test


def sgc_precompute(features, adj, degree):
    t = perf_counter()
    features_temp = features
    for i in range(degree):
        features_temp = torch.spmm(adj, features_temp)
    precompute_time = perf_counter()-t
    return features_temp, precompute_time

def set_seed(seed=13, cuda=True):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda: torch.cuda.manual_seed(seed)

def loadRedditFromNPZ(dataset_dir):
    adj = sp.load_npz(dataset_dir+"reddit_adj.npz")
    data = np.load(dataset_dir+"reddit.npz")

    return adj, data['feats'], data['y_train'], data['y_val'], data['y_test'], data['train_index'], data['val_index'], data['test_index']

def load_reddit_data(data_path="data/", normalization="AugNormAdj", cuda=True):
    adj, features, y_train, y_val, y_test, train_index, val_index, test_index = loadRedditFromNPZ("data/")
    labels = np.zeros(adj.shape[0])
    labels[train_index]  = y_train
    labels[val_index]  = y_val
    labels[test_index]  = y_test
    adj = adj + adj.T + sp.eye(adj.shape[0])
    train_adj = adj[train_index, :][:, train_index]
    features = torch.FloatTensor(np.array(features))
    features = (features-features.mean(dim=0))/features.std(dim=0)
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    train_adj = adj_normalizer(train_adj)
    train_adj = sparse_mx_to_torch_sparse_tensor(train_adj).float()
    labels = torch.LongTensor(labels)
    if cuda:
        adj = adj.cuda()
        train_adj = train_adj.cuda()
        features = features.cuda()
        labels = labels.cuda()
    return adj, train_adj, features, labels, train_index, val_index, test_index


def load_Caltech101_data(dataset='flower',cuda=True, seed = 6 ,rate = 10 ):  ##handwritten citeseer cora flower
    dir = 'data/data_set/' + dataset + "/"
    mat_dir = dir + dataset
    info2 = loadmat(mat_dir) #data/handwritten.mat

    data = []
    idx_file = dir +  '{}_{}.idx'.format(rate, seed)
    f = open(idx_file, "r")
    train_test_idx  = json.load(f)
    train_idx = train_test_idx['train']
    test_idx = train_test_idx['test']

    f.close()
    if dataset == 'Caltech101-7':
        data_temp = info2['X'].squeeze()
        label_temp = info2['Y'].squeeze()
        for features in data_temp:
            data.append(features)
        label = label_temp.squeeze()
    elif dataset == 'handwritten':
        data_temp = info2['X'].squeeze()
        label_temp = info2['gt'].squeeze()
        for features in data_temp:
            data.append(features)
        label = label_temp.squeeze()
    elif dataset == 'flower':
        data_temp = info2['X'].squeeze()
        label_temp = info2['gt'].squeeze()
        for features in data_temp:
            data.append(features)
        label = label_temp.squeeze()
    elif dataset == 'citeseer':
        data_temp = info2['X'].squeeze()
        label_temp = info2['gt'].squeeze()
        for features in data_temp:
            data.append(features)
        label = label_temp.squeeze()
    elif dataset == 'cora':
        data_temp = info2['X'].squeeze()
        label_temp = info2['gt'].squeeze()
        for features in data_temp:
            data.append(features)
        label = label_temp.squeeze()
    elif dataset == 'AWA':
        data_temp = info2['X'].squeeze()
        label_temp = info2['gt'].squeeze()
        for features in data_temp:
            data.append(features.toarray())
        label = label_temp.squeeze()
    elif dataset == '3sources' :
        data_temp = info2['data'].squeeze()
        label_temp = info2['truelabel'].squeeze()
        for features in data_temp:
            features = features.swapaxes(1, 0)
            data.append(features)
        label = label_temp[0].squeeze()
    elif dataset == 'BBCSport' :
        data_temp = info2['data'].squeeze()
        label_temp = info2['truelabel'].squeeze()
        for features in data_temp:
            features = features.toarray().swapaxes(1, 0)
            data.append(features)
        label = label_temp[0].squeeze()
    elif dataset == 'BBC' :
        data_temp = info2['data'].squeeze()
        label_temp = info2['truelabel'].squeeze()
        for features in data_temp:
            features = features.toarray().swapaxes(1, 0)
            data.append(features)
        label = label_temp[0].squeeze()
    elif dataset == 'WebKB':
        data_temp = info2['data'].squeeze()
        label_temp = info2['truelabel'].squeeze()
        for features in data_temp:
            features = features.swapaxes(1, 0)
            data.append(features)
        label = label_temp[0].squeeze()

    feature_list = []
    adj_list = []
    k = 5
    for m in range(len(data)):
        features = data[m]
        features = (features - features.mean(0)) / (features.std(0)+0.00000001)

        npyname = "npy/" + dataset + "_K" + str(k) + "_Rate" + str(rate) + "_V" + str(m) + "_semi.npy"
        if not os.path.exists(npyname):
            index1 = knn_hnsw(features, k)
            num_ins = len(index1.knns)
            adj = np.zeros((num_ins, num_ins))
            for i in range(num_ins):
                for j in range(k):
                    index = index1.knns[i][0][j]
                    adj[i][index] = 1
            adj = torch.LongTensor(adj).float()
            adj = normalize_graph(adj)
            np.save(npyname, adj.numpy())

        adj = torch.from_numpy(np.load(npyname))


        # index1 = knn_hnsw(features, k)
        # # index2 = knn_faiss(features, k)
        # num_ins = len(index1.knns)
        # adj = np.zeros((num_ins,num_ins))
        # for i in range(num_ins):
        #     for j in range(k):
        #         index = index1.knns[i][0][j]
        #         adj[i][index] = 1
        #
        # adj = torch.LongTensor(adj).float()
        # adj = normalize_graph(adj)
        features = torch.FloatTensor(np.array(features))
        if cuda:
            adj = adj.cuda()
            features = features.cuda()
        adj_list.append(adj)
        feature_list.append(features)

    if label.min() == 1 :
        label = torch.LongTensor(np.array(label)-1)
    else:
        label = torch.LongTensor(np.array(label))

    label_num = []
    label_rate = 0.1
    label_train_num = []
    train_index = []
    test_index = []
    # for k in range(int(label.max()+1)):
    #     num = ((label == k) + 0).sum()
    #     index = torch.range(0, len(label) - 1)[label == k]
    #     label_num.append(int(num))
    #     label_train_num.append(int(num*label_rate))
    #
    #     x_list = random.sample(list(index), label_train_num[-1])
    #     for x in x_list:
    #         train_index.append(int(x))
    #
    # for i in range(len(label)):
    #     if int(i) not in train_index:
    #         test_index.append(int(i))

    train_index = torch.LongTensor(train_idx)
    test_index = torch.LongTensor(test_idx)

    label = label.cuda()
    train_index = train_index.cuda()
    test_index = test_index.cuda()

    #print info
    print('[DataName]:{}'.format(dataset))
    print('[View number]: {} [instance]: {} [label]: {}'.format(len(adj_list),len(label), int(label.max())+1))
    for i in range(len(adj_list)):
        print('[View {} ]: {} feature'.format(int(i), feature_list[i].size(1)))
    # for i in range(int(label.max())+1):
    #     print('[Label {} ]: {} instance'.format(int(i), label_num[i]))
    return adj_list, feature_list, label, train_index, test_index, len(adj_list), len(label)



def spectralclustering(data, n_clusters):
	N = data.shape[0]
	maxiter = 1000  # max iteration times
	replic = 100  # number of time kmeans will be run with diff centroids

	DN = np.diag(1/np.sqrt(np.sum(data, axis=0) + eps))
	lapN = np.eye(N) - DN.dot(data).dot(DN)
	U, A, V = np.linalg.svd(lapN)
	V = V.T
	kerN = V[:, N - n_clusters :N ]
	normN = np.sum(kerN**2, 1)**(0.5)
	kerNS = (kerN.T/(normN + eps)).T
	# kmeans
	clf = KMeans(n_clusters=n_clusters, max_iter=maxiter, n_init=replic)
	return clf.fit_predict(kerNS)

if __name__ == '__main__':

    out = load_Caltech101_data(dataset="flower")
    print("end")
