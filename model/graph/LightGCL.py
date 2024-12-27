import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE


class LightGCL(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(LightGCL, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['LightGCL'])
        self.cl_rate = float(args['-lambda1'])
        self.lambda2 = float(args['-lambda2'])
        svd_data = conf.config['svd']
        self.temp = float(args['-temp'])
        self.n_layers = int(args['-n_layer'])
        self.device = conf.config['device']
        self.model = LGCL_Encoder(self.data, self.emb_size, self.n_layers, svd_data, self.device)

    def train(self):
        if self.device == 'gpu':
            model = self.model.cuda()
        else:
            model = self.model.cpu()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[
                    neg_idx]
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                # cl_loss = self.cl_rate * self.cal_cl_loss([user_idx, pos_idx])
                cl_loss = self.cl_rate * self.cal_cl_loss([user_idx, pos_idx])
                batch_loss = rec_loss + l2_reg_loss(self.lambda2, user_emb, pos_item_emb) + cl_loss
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100 == 0 and n > 0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def cal_cl_loss(self, idx):
        if self.device == 'gpu':
            u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
            i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        else:
            u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cpu()
            i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cpu()
        user_view_1, user_view_2, item_view_1, item_view_2 = self.model(perturbed=True)
        # user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], self.temp)
        # item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], self.temp)
        user_cl_loss = self.lightgcl_cl_loss(user_view_1, user_view_2, u_idx)
        item_cl_loss = self.lightgcl_cl_loss(item_view_1, item_view_2, i_idx)
        return user_cl_loss + item_cl_loss

    def lightgcl_cl_loss(self, view1, view2, ids):
        neg_score = torch.log(torch.exp(view1[ids] @ view2[ids].T / self.temp).sum(1) + 1e-8).mean()
        pos_score = (torch.clamp((view1[ids] * view2[ids]).sum(1) / self.temp, -5.0, 5.0)).mean()
        loss_s = -pos_score + neg_score
        return loss_s

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class LGCL_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers, svd_data, device):
        super(LGCL_Encoder, self).__init__()
        self.n_u, self.n_i, self.u_mul_s, self.v_mul_s, self.ut, self.vt, self.train_csr, self.adj_norm = svd_data
        self.data = data
        self.emb_size = emb_size
        self.l = n_layers
        self.device = device
        # self.E_u_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_u, self.latent_size)))
        # self.E_i_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_i, self.latent_size)))
        # self.E_u_list = [None] * (self.l + 1)
        # self.E_i_list = [None] * (self.l + 1)
        # self.E_u_list[0] = self.E_u_0
        # self.E_i_list[0] = self.E_i_0
        # self.Z_u_list = [None] * (self.l + 1)
        # self.Z_i_list = [None] * (self.l + 1)
        # self.G_u_list = [None] * (self.l + 1)
        # self.G_i_list = [None] * (self.l + 1)
        # self.G_u_list[0] = self.E_u_0
        # self.G_i_list[0] = self.E_i_0
        # self.dropout = 0.0

        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        if self.device == 'gpu':
            self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()
        else:
            self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cpu()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict

    def forward(self, perturbed=False):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = []
        svd_user_embs, svd_item_embs = [],[]
        for layer in range(1, self.l + 1):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
            if perturbed:
                user_emb, item_emb = torch.split(ego_embeddings, [self.data.user_num, self.data.item_num])
                # svd_adj propagation
                ut_eu = self.ut @ user_emb
                vt_ei = self.vt @ item_emb

                user_emb = (self.u_mul_s @ vt_ei)
                item_emb = (self.v_mul_s @ ut_eu)
                svd_user_embs.append(user_emb)
                svd_item_embs.append(item_emb)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])
        if perturbed:
            svd_user_embs = sum(svd_user_embs)
            svd_item_embs = sum(svd_item_embs)
            return user_all_embeddings,svd_user_embs, item_all_embeddings, svd_item_embs
        else:
            return user_all_embeddings,item_all_embeddings




        # for layer in range(1, self.l + 1):
        #     # GNN propagation
        #     self.Z_u_list[layer] = (
        #         torch.spmm(self.sparse_dropout(self.adj_norm, self.dropout), self.E_i_list[layer - 1]))
        #     self.Z_i_list[layer] = (
        #         torch.spmm(self.sparse_dropout(self.adj_norm, self.dropout).transpose(0, 1), self.E_u_list[layer - 1]))
        #     if not test:
        #         # svd_adj propagation
        #         vt_ei = self.vt @ self.E_i_list[layer - 1]
        #         self.G_u_list[layer] = (self.u_mul_s @ vt_ei)
        #         ut_eu = self.ut @ self.E_u_list[layer - 1]
        #         self.G_i_list[layer] = (self.v_mul_s @ ut_eu)
        #
        #     # aggregate
        #     self.E_u_list[layer] = self.Z_u_list[layer]
        #     self.E_i_list[layer] = self.Z_i_list[layer]
        # if not test:
        #     # self.G_u = sum(self.G_u_list)
        #     # self.G_i = sum(self.G_i_list)
        #     G_u = sum(self.G_u_list)
        #     G_i = sum(self.G_i_list)
        #
        # # aggregate across layers
        # # self.E_u = sum(self.E_u_list)
        # # self.E_i = sum(self.E_i_list)
        # E_u = sum(self.E_u_list)
        # E_i = sum(self.E_i_list)
        #
        # E_u_norm = E_u
        # E_i_norm = E_i
        # # cl loss
        # if not test:
        #     G_u_norm = G_u
        #     G_i_norm = G_i
        #
        # if test:
        #     return E_u_norm, E_i_norm
        # if not test:
        #     return G_u_norm, E_u_norm, G_i_norm, E_i_norm

    def sparse_dropout(self, mat, dropout):
        if dropout == 0.0:
            return mat
        indices = mat.indices()
        values = nn.functional.dropout(mat.values(), p=dropout)
        size = mat.size()
        return torch.sparse.FloatTensor(indices, values, size)
