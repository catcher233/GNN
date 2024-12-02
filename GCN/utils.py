import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd

def encode_onehot(labels):
    classes = set(labels)
    dict_class= {c:np.identity(len(classes))[i,:] for i,c in enumerate(classes)}
    labels_onehot = np.array(list(map(dict_class.get, labels)),dtype=np.int32)
    return labels_onehot
def load_data(path="data/cora",data="cora"):
    print("Loading {} dataset".format(data))
    idx_features_labels = np.genfromtxt("{}/{}.content".format(path,data),dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:,1:-1],dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:,-1])
    idx = np.array(idx_features_labels[:,0], dtype=np.int32)
    idx_map = {j:i for i,j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}/{}.cites".format(path,data),dtype=np.int32)
    edges = np.array(list(map(idx_map.get,edges_unordered.flatten())),dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]),(edges[:,0],edges[:,1])),shape=(labels.shape[0],labels.shape[0]),dtype=np.float32)
    adj = adj+adj.T.multiply(adj.T>adj)-adj.multiply(adj.T>adj)
    fearures = normalize(features)
    adj = normalize(adj+sp.eye(adj.shape[0]))
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    return adj, features, labels, idx_train, idx_val, idx_test
def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(mx)
def accuracy(output,labels):
    pred = output.max(1)[1].type_as(labels)
    correct = pred.eq(labels).double()
    correct = correct.sum()
    return correct/len(labels)
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


