import numpy as np
import torch
import pickle
import torch.utils.data as data

def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def get_svd(dataset,device):
    # load data
    path = 'dataset/' +dataset + '/'
    f = open(path+'trnMat.pkl','rb')
    train = pickle.load(f)
    train_csr = (train!=0).astype(np.float32)

    # normalizing the adj matrix
    rowD = np.array(train.sum(1)).squeeze()
    colD = np.array(train.sum(0)).squeeze()
    for i in range(len(train.data)):
        train.data[i] = train.data[i] / pow(rowD[train.row[i]]*colD[train.col[i]], 0.5)

    # construct data loader
    train = train.tocoo()
    print('Adj matrix normalized.')

    adj_norm = scipy_sparse_mat_to_torch_sparse_tensor(train)
    if device == 'gpu':
        adj_norm = adj_norm.coalesce().cuda()
        adj = scipy_sparse_mat_to_torch_sparse_tensor(train).coalesce().cuda()

    else:
        adj_norm = adj_norm.coalesce().cpu()
        adj = scipy_sparse_mat_to_torch_sparse_tensor(train).coalesce().cpu()



    # perform svd reconstruction
    # adj = scipy_sparse_mat_to_torch_sparse_tensor(train).coalesce().cuda()
    print('Performing SVD...')
    svd_u,s,svd_v = torch.svd_lowrank(adj, q=5)
    u_mul_s = svd_u @ (torch.diag(s))
    v_mul_s = svd_v @ (torch.diag(s))
    del s
    print('SVD done.')

    svd_data = adj_norm.shape[0], adj_norm.shape[1], u_mul_s, v_mul_s, svd_u.T, svd_v.T, train_csr, adj_norm
    return svd_data