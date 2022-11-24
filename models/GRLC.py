import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def cos(x, y):
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


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 1) / torch.sum(msk)


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=True)
        self.fc_1 = nn.Linear(in_ft, out_ft * 2, bias=True)
        self.fc_2 = nn.Linear(out_ft * 2, out_ft, bias=False)
        self.act = nn.PReLU() if act is not None else None

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    #   Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.mm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        if self.act:
            out = self.act(out)
        return out  # , out /out.norm(dim=2)[:, :,None]


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj, do_att=False):
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        if do_att:
            a_input = self._prepare_attentional_mechanism_input(Wh)
            e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)
            attention = F.softmax(attention, dim=1)
            attention = F.dropout(attention, self.dropout, training=self.training)
        else:
            attention = adj
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  # number of nodes

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.nonzero().t()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.out_att = GraphAttentionLayer(nfeat, nhid, dropout=0.1, alpha=0.0, concat=False)

    def forward(self, x, adj, do_att=False):
        x = self.out_att(x, adj, do_att)
        return x


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi):

        sc_1 = torch.squeeze(self.f_k(h_pl, c), 2)
        sc_2 = torch.squeeze(self.f_k(h_mi, c), 2)
        logits = torch.cat((sc_1, sc_2), 1)

        return logits


class Discriminator_cluster(nn.Module):
    def __init__(self, n_in, n_h, n_nb, num_clusters):
        super(Discriminator_cluster, self).__init__()

        self.n_nb = n_nb
        self.n_h = n_h
        self.num_clusters = num_clusters

    def forward(self, c, h_0, h_pl, h_mi):
        c_x = c.expand_as(h_0)
        sc_1 = cos(h_pl, c_x)
        sc_2 = cos(h_mi, c_x)
        mask = (torch.eye(self.n_nb) == 1).to(h_0.device)
        sc_1 = sc_1[mask]
        sc_2 = sc_2[mask]
        logits = torch.cat((sc_1, sc_2), 0).view(1, -1)

        return logits


def cluster(data, k, temp, num_iter, init, cluster_temp):
    cuda0 = torch.cuda.is_available()  # False

    if cuda0:
        mu = init.cuda()
        data = data.cuda()
        cluster_temp = cluster_temp.cuda()
    else:
        mu = init
    n = data.shape[0]
    d = data.shape[1]
    #
    data = data / data.norm(dim=1)[:, None]
    for t in range(num_iter):
        mu = mu / mu.norm(dim=1)[:, None]
        dist = torch.mm(data, mu.transpose(0, 1))

        # cluster responsibilities via softmax
        r = F.softmax(cluster_temp * dist, dim=1)
        # total responsibility of each cluster
        cluster_r = r.sum(dim=0)
        # mean of points in each cluster weighted by responsibility
        cluster_mean = r.t() @ data
        # update cluster means
        new_mu = torch.diag(1 / cluster_r) @ cluster_mean
        mu = new_mu

    r = F.softmax(cluster_temp * dist, dim=1)

    return mu, r


class Clusterator(nn.Module):
    '''
    The ClusterNet architecture. The first step is a 2-layer GCN to generate embeddings.
    The output is the cluster means mu and soft assignments r, along with the
    embeddings and the the node similarities (just output for debugging purposes).

    The forward pass inputs are x, a feature matrix for the nodes, and adj, a sparse
    adjacency matrix. The optional parameter num_iter determines how many steps to
    run the k-means updates for.
    '''

    def __init__(self, nout, K):
        super(Clusterator, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.K = K
        self.nout = nout

        self.init = torch.rand(self.K, nout)

    def forward(self, embeds, cluster_temp, num_iter=10):
        mu_init, _ = cluster(embeds, self.K, 1, num_iter, cluster_temp=torch.tensor(cluster_temp), init=self.init)
        # self.init = mu_init.clone().detach()
        mu, r = cluster(embeds, self.K, 1, 1, cluster_temp=torch.tensor(cluster_temp), init=mu_init.clone().detach())

        return mu, r


class GRLC_GCN_test(nn.Module):
    def __init__(self, n_nb, n_in, n_h, dim_x=2, useact=False, liner=False, dropout=0.2, useA=True):
        super(GRLC_GCN_test, self).__init__()
        self.gcn_0 = GCN(n_in, n_h * dim_x, act=None)
        self.gcn_1 = GCN(n_h * dim_x, n_h, act=None)
        self.act = nn.ReLU()
        self.sigm = nn.Sigmoid()
        self.liner = liner
        self.useact = useact
        self.S = None
        self.dropout = dropout
        self.A = None
        self.sparse = False
        self.useA = useA

    def forward(self, seq_a, seq_p, seq_n, adj=None, diff=None, I=None):
        if self.A is None:
            self.A = adj
        seq_p = F.dropout(seq_p, self.dropout, training=self.training)
        seq_n_list = []
        for seq_n_temp in seq_n:
            seq_n_list.append(F.dropout(seq_n_temp, self.dropout, training=self.training))

        h_a_0 = self.gcn_0(seq_a, I)
        h_a_0 = self.act(h_a_0)

        h_a = self.gcn_1(h_a_0, I)
        h_p_0 = self.gcn_0(seq_p, I)
        h_p_0 = self.act(h_p_0)

        h_p = F.dropout(h_p_0, self.dropout, training=self.training)
        h_p = self.gcn_1(h_p, self.A)

        h_n_list = []
        h_n_0_list = []
        for seq_n_temp in seq_n_list:
            h_n_0 = self.gcn_0(seq_n_temp, I)
            h_n_0 = self.act(h_n_0)
            h_n_0_list.append(h_n_0)
            h_n = F.dropout(h_n_0, self.dropout, training=self.training)
            if self.useA:
                h_n = self.gcn_1(h_n, self.A)
            else:
                h_n = self.gcn_1(h_n, I)

            if self.useact:
                h_n = self.act(h_n)
            h_n_list.append(h_n)

        if self.useact:
            return self.act(h_a), self.act(h_p), h_n_list, h_a_0, h_p_0, h_n_0_list
        else:
            return h_a, h_p, h_n_list

    def embed(self, seq_a, seq_p, seq_n, adj=None, diff=None, I=None):
        h_a = self.gcn_0(seq_a, I)
        h_a = self.gcn_1(self.act(h_a), I)
        h_p = self.gcn_0(seq_p, I)
        h_p = self.act(h_p)
        h_p = self.gcn_1(h_p, self.A)

        if self.useact:
            return self.act(h_a).detach(), self.act(h_p).detach()
        else:
            return h_a.detach(), h_p.detach()
