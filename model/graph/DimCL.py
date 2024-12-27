import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE


# KDD'24 Submission: DimCL


class DimCL(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(DimCL, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['DimCL'])
        self.cl_rate = float(args['-lambda'])
        self.eps = float(args['-eps'])
        self.n_layers = int(args['-n_layer'])
        self.temp = float(args['-temp'])
        self.tau1 = float(args['-tau1'])
        self.tau2 = float(args['-tau2'])
        self.tau3 = float(args['-tau3'])
        self.taus = (self.tau1, self.tau2, self.tau3)
        self.device = conf.config['device']
        self.fac = int(args['-fac'])
        self.model = DimCL_Encoder(self.data, self.emb_size, self.temp, self.taus, self.eps, self.n_layers, self.device, self.fac)

    def train(self):
        if self.device == 'gpu':
            model = self.model.cuda()
        else:
            model = self.model.cpu()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model(user_idx, pos_idx, neg_idx)
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[
                    neg_idx]
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                cl_loss = self.cl_rate * self.cal_cl_loss([user_idx, pos_idx], user_idx, pos_idx, neg_idx, self.temp)
                batch_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) + cl_loss
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

    def cal_cl_loss(self, idx, user_idx, pos_idx, neg_idx, temp):
        if self.device == 'gpu':
            u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
            i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        else:
            print('cpu warning!')
            u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cpu()
            i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cpu()
        user_view_1, item_view_1,user_view_2, item_view_2 = self.model(user_idx, pos_idx, neg_idx, perturbed=True)
        # user_view_2, item_view_2 = self.model(user_idx, pos_idx, neg_idx, perturbed=True)
        user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], temp)
        item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], temp)
        return user_cl_loss + item_cl_loss

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class DimCL_Encoder(nn.Module):
    def __init__(self, data, emb_size, temp, taus, eps, n_layers, device, fac):
        super(DimCL_Encoder, self).__init__()
        self.data = data
        self.eps = eps
        self.d = emb_size
        self.n_layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.device = device
        if self.device == 'gpu':
            self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()
        else:
            print('cpu warning')
            self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cpu()
        self.sigmoid = nn.Sigmoid()
        self.temp = temp
        self.tau1, self.tau2, self.tau3 = taus
        self.fac = fac
        self.parameter_init()

    def parameter_init(self):
        self.user_noise_1 = nn.Sequential(
            nn.Linear(self.d, self.d, bias=False),
            # nn.Sigmoid()
            nn.GELU()
        )

        self.user_noise_2 = nn.Sequential(
            nn.Linear(self.d, self.d, bias=False),
            nn.GELU(),
            # nn.Sigmoid()

        )

        self.item_noise_1 = nn.Sequential(
            nn.Linear(self.d, self.d, bias=False),
            nn.GELU()
            # nn.Sigmoid()
        )

        self.item_noise_2 = nn.Sequential(
            nn.Linear(self.d, self.d, bias=False),
            nn.GELU(),
            # nn.Sigmoid()
        )

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.d))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.d))),
        })
        return embedding_dict

    def forward(self, uids=0, pos=0, neg=0, perturbed=False):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = []
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])

        if perturbed:
            views_user, views_item = self.augmentation(user_all_embeddings, item_all_embeddings, uids, pos, neg)
            user_view1, user_view2 = views_user
            item_view1, item_view2 = views_item
            return user_view1, item_view1, user_view2, item_view2

        else:
            return user_all_embeddings, item_all_embeddings

    def generate_noise(self, embs, type='user'):
        # embs = self.norm(embs)
        if type == 'user':
            noise1 = self.user_noise_1(embs)
            noise2 = self.user_noise_2(embs)
        elif type == 'item':
            noise1 = self.item_noise_1(embs)
            noise2 = self.item_noise_2(embs)
        else:
            print('NO IMPLEMENTATION!')
            assert 1 == 2
        return noise1, noise2

    def generate_views(self, embs, noises, w=(0, 0), type='ge'):
        noise1, noise2 = noises
        if type == 'ge':
            # noise1 = F.normalize(noise1)
            # noise2 = F.normalize(noise2)
            view1 = embs + noise1
            view2 = embs + noise2
        elif type == 'de':
            # noise1 = F.normalize(noise1)
            # noise2 = F.normalize(noise2)
            view1 = embs + w[0] * noise1
            view2 = embs + w[1] * noise2
        else:
            print('NO IMPLEMENTATION!')
            assert 1 == 2
        return view1, view2

    def generate_factors(self, views, emb, noises, ids, type='user'):
        view1, view2 = views
        noise1, noise2 = noises
        loss_ssl = self.SSL(ids, view1, view2, self.temp, type=type)
        loss_ssl.backward(retain_graph=True)
        grad1, grad2 = noise1.grad, noise2.grad

        view1_fac1 = self.fac1(grad1)  # view1's factors
        view1_fac2 = self.fac2(emb, noise1)

        view2_fac1 = self.fac1(grad2)  # view2's factors
        view2_fac2 = self.fac2(emb, noise2)

        return (view1_fac1, view2_fac1), (view1_fac2, view2_fac2)

    def SSL(self, idx, view1, view2, temp, type='user'):
        if self.device == 'gpu':
            if type == 'user':
                u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
                user_cl_loss = InfoNCE(view1[u_idx], view2[u_idx], temp)
                return user_cl_loss
            elif type == 'item':
                i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
                item_cl_loss = InfoNCE(view1[i_idx], view2[i_idx], temp)
                return item_cl_loss
        else:
            print('cpu warning')
            if type == 'user':
                u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cpu()
                user_cl_loss = InfoNCE(view1[u_idx], view2[u_idx], temp)
                return user_cl_loss
            elif type == 'item':
                i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cpu()
                item_cl_loss = InfoNCE(view1[i_idx], view2[i_idx], temp)
                return item_cl_loss

    def REC(self, emb_user, emb_item, uids, pos, neg):
        user_emb = emb_user[uids]
        pos_item_emb = emb_item[pos]
        neg_item_emb = emb_item[neg]
        pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
        neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)
        loss = -torch.log(10e-6 + torch.sigmoid(pos_score - neg_score))
        return torch.mean(loss)

    def fac1(self, grad):
        # grad: nxd
        grad = grad.unsqueeze(1)  # n x 1 x d
        grad_T = grad.transpose(2, 1)
        temp1 = grad_T @ torch.ones(grad.shape, device=grad.device)  # n x d x d
        temp2 = torch.ones(temp1.shape, device=grad.device) - torch.eye(self.d, self.d, device=grad.device)
        grad_wo_dim = temp2.transpose(2, 1)
        grad = grad.repeat(1, self.d, 1)
        factor1 = F.cosine_similarity(grad, grad_wo_dim, dim=2)
        return factor1

    # def fac2(self, emb, noise):
    #     act = nn.Sigmoid()
    #     sim = torch.clamp(emb * noise / self.temp, min=-5.0, max=5.0)
    #     factor2 = act(torch.exp(sim))
    #     return factor2

    # 0719
    def fac2(self, emb, noise):
        act = nn.Sigmoid()
        sim = (emb * noise) / self.temp
        factor2 = act(torch.exp(sim))
        return factor2

    # def fac3(self, user_view, item_view, uid, pos, neg):
    #     user_view1, user_view2 = user_view
    #     item_view1, item_view2 = item_view
    #     loss_1 = self.REC(user_view1, item_view1, uid, pos, neg)
    #     loss_2 = self.REC(user_view2, item_view2, uid, pos, neg)
    #     user_base_emb = self.embedding_dict['user_emb']
    #     item_base_emb = self.embedding_dict['item_emb']
    #     grad1_user = torch.autograd.grad(loss_1, user_base_emb, retain_graph=True, create_graph=True)[0]
    #     grad1_item = torch.autograd.grad(loss_1, item_base_emb, retain_graph=True, create_graph=True)[0]
    #     grad2_user = torch.autograd.grad(loss_2, user_base_emb, retain_graph=True, create_graph=True)[0]
    #     grad2_item = torch.autograd.grad(loss_2, item_base_emb, retain_graph=True, create_graph=True)[0]
    #     sim_user = grad1_user * grad2_user  # u x d
    #     sim_item = grad1_item * grad2_item  # i x d
    #     sim_user = self.sigmoid(torch.clamp(sim_user, -5.0, 5.0))
    #     sim_item = self.sigmoid(torch.clamp(sim_item, -5.0, 5.0))
    #     return sim_user, sim_item

    def fac3(self, user_view, item_view, uid, pos, neg):
        user_view1, user_view2 = user_view
        item_view1, item_view2 = item_view
        loss_1 = self.REC(user_view1, item_view1, uid, pos, neg)
        loss_2 = self.REC(user_view2, item_view2, uid, pos, neg)
        user_base_emb = self.embedding_dict['user_emb']
        item_base_emb = self.embedding_dict['item_emb']
        grad1_user = torch.autograd.grad(loss_1, user_base_emb, retain_graph=True, create_graph=True)[0]
        grad1_item = torch.autograd.grad(loss_1, item_base_emb, retain_graph=True, create_graph=True)[0]
        grad2_user = torch.autograd.grad(loss_2, user_base_emb, retain_graph=True, create_graph=True)[0]
        grad2_item = torch.autograd.grad(loss_2, item_base_emb, retain_graph=True, create_graph=True)[0]
        sim_user = grad1_user * grad2_user  # u x d
        sim_item = grad1_item * grad2_item  # i x d
        sim_user = self.sigmoid(sim_user)
        sim_item = self.sigmoid(sim_item)
        return sim_user, sim_item

    # def denoise_intra_view(self, factors):
    #     # input: n x d
    #     factor1, factor2 = factors
    #     factor1 = torch.stack((factor1, 1 - factor1), -1)  # (sim, dis-sim)
    #     factor2 = torch.stack((factor2, 1 - factor2), -1)
    #
    #     w1 = F.gumbel_softmax(factor1, tau=self.tau1, hard=True)
    #     w2 = F.gumbel_softmax(factor2, tau=self.tau2, hard=True)
    #     rst = torch.clamp(w1 + w2, max=1.0)  # double-check high-similarity = 0
    #     # output: n x d
    #     w = rst[:, :, 0]
    #     return w.squeeze()
    #
    # 0719
    def denoise_intra_view(self, factors):
        # input: n x d
        factor1, factor2 = factors
        factor1 = torch.stack((factor1, 1 - factor1), -1)  # (sim, dis-sim)
        factor2 = torch.stack((factor2, 1 - factor2), -1)

        w1 = F.gumbel_softmax(factor1, tau=self.tau1, hard=True)
        w2 = F.gumbel_softmax(factor2, tau=self.tau2, hard=True)

        if self.fac == 1 or self.fac == 22:
            w12 = w2
        elif self.fac == 2 or self.fac==11:
            w12 = w1
        else:
            w12 = w1+w2

        rst = w12 - 1 * torch.gt(w12,1.0)
        # rst = torch.clamp(w1 + w2, max=1.0)  # double-check high-similarity = 0
        # output: n x d
        w = rst[:, :, 0]
        return w.squeeze()

    # def denoise(self, factor_views, factor3, type='same'):
    #     fac_view1, fac_view2 = factor_views
    #     factor3 = torch.stack((1 - factor3, factor3), -1)
    #     w3 = F.gumbel_softmax(factor3, tau=self.tau3, hard=True)[:, :, 0]
    #     w_view1 = self.denoise_intra_view(fac_view1)
    #     w_view2 = self.denoise_intra_view(fac_view2)
    #
    #     if type == 'same':  # each view =0  => 0
    #         w = w_view1 * w_view2
    #         w1 = w
    #         w2 = w
    #     elif type == 'diff':  # two view = 0  => 0
    #         w1 = w_view1
    #         w2 = w_view2
    #     else:
    #         assert 1 == 2
    #     return w1 * w3, w2 * w3

    def denoise(self, factor_views, factor3, type='same'):
        fac_view1, fac_view2 = factor_views
        factor3 = torch.stack((1 - factor3, factor3), -1)
        w3 = F.gumbel_softmax(factor3, tau=self.tau3, hard=True)[:, :, 0]
        if self.fac == 33:
            w_view1=w_view2=1
        else:
            w_view1 = self.denoise_intra_view(fac_view1)
            w_view2 = self.denoise_intra_view(fac_view2)
        if self.fac == 3 or self.fac==11 or self.fac==22:
            w3 = 1

        if type == 'same':  # each view =0  => 0
            w = w_view1 * w_view2
            w1 = w
            w2 = w
        elif type == 'diff':  # two view = 0  => 0
            w1 = w_view1
            w2 = w_view2
        else:
            assert 1 == 2
        return w1 * w3, w2 * w3

    def augmentation(self, emb_user, emb_item, uid, pos, neg):
        noise_user = self.generate_noise(emb_user, type='user')
        noise_item = self.generate_noise(emb_item, type='item')
        noise_user[0].retain_grad()
        noise_user[1].retain_grad()
        noise_item[0].retain_grad()
        noise_item[1].retain_grad()
        view_user_ge = self.generate_views(emb_user, noise_user)
        view_item_ge = self.generate_views(emb_item, noise_item)
        ids = [uid,pos]
        factor_user = self.generate_factors(view_user_ge, emb_user, noise_user, ids, type='user')
        factor_item = self.generate_factors(view_item_ge, emb_item, noise_item, ids, type='item')
        factor3_user, factor3_item = self.fac3(view_user_ge, view_item_ge, uid, pos, neg)
        w_user = self.denoise(factor_user, factor3_user, type='same')
        w_item = self.denoise(factor_item, factor3_item, type='same')
        view_user_de = self.generate_views(emb_user, noise_user, w_user, type='de')
        view_item_de = self.generate_views(emb_item, noise_item, w_item, type='de')
        return view_user_de, view_item_de
