import torch
import h5py
import joblib
import random
import anndata
import platform
import warnings
import scipy as sp
import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.colors as mcs
from scipy.sparse import coo_matrix
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from gcorr import pearson

warnings.filterwarnings('ignore')





########################################## Data Loader ############################################


dataset_dir = "./datasets/"


def read_h5(file):
    def empty_safe(fn, dtype):
        def _fn(x):
            if x.size:
                return fn(x)
            return x.astype(dtype)

        return _fn

    decode = empty_safe(np.vectorize(lambda _x: _x.decode("utf-8")), str)
    f = h5py.File(file, "r")
    exprs_handle = f["exprs"]
    mat = sp.sparse.csr_matrix((exprs_handle["data"][...], exprs_handle["indices"][...], exprs_handle["indptr"][...]),
                               shape=exprs_handle["shape"][...])
    X = np.array(mat.toarray())
    cell_names = decode(f["obs"]["cell_type1"][...])
    genes = decode(f["var_names"][...])
    df = pd.DataFrame(X, columns=genes)
    df["y1"] = cell_names
    return df.copy()


def read_h5ad(file):
    f = anndata.read_h5ad(file)
    X = f.X
    cell_names = f.obs["barcode"].to_list()
    genes = f.var["gene_name"].to_list()
    df = pd.DataFrame(X, columns=genes)
    df.index = cell_names
    df["y1"] = f.obs["cell_type"].to_list()
    return df.copy()


def load_data(filename):
    import os
    if filename.count(".") == 0:
        if os.path.exists("%s/%s.h5" % (dataset_dir, filename)):
            return load_data(filename + ".h5")
        if os.path.exists("%s/%s.h5ad" % (dataset_dir, filename)):
            return load_data(filename + ".h5ad")
        elif os.path.exists("%s/%s.csv" % (dataset_dir, filename)):
            return load_data(filename + ".csv")
        else:
            raise ValueError(f"unknown datatype! file: {filename}")
    if filename.count(":") == 0:
        filename = "%s/%s" % (dataset_dir, filename)
    if filename[-4:] == ".csv":
        df = pd.read_csv(filename)
    elif filename[-3:] == ".h5":
        df = read_h5(filename)
    elif filename[-5:] == ".h5ad":
        df = read_h5ad(filename)
    else:
        raise ValueError(f"unknown datatype! file: {filename}")
    y = df.loc[:, ["y1"]].to_numpy().flatten()
    ys = [y for y in ["y1", "y2", "y3"] if y in df.columns]
    df.drop(ys, axis=1, inplace=True)
    x = df.astype(float)
    return x, y


def lab2fac(label):
    _, y = np.unique(label, return_inverse=True)
    return y





########################################## Visualization ############################################



def _show(data, y, title="Channel", mask=None, label=False, red=None):
    """
    data:  ndarray or list, with a shape of (-1, n, m) or (n, m), representing several subplots or one plot accordingly.
    y:     list of number from 0 to 9 with a shape of (n), representing the color for each sample.
    title: str or list of str with a shape of (-1), representing title for each subplot.
    mask:  list of number, for which sample the color is ambiguous and will be presented as color 'C9'
    """
    if mask is None:
        mask = []
    y = list(y)
    for i in mask:
        y[i] = -1

    if data[0].ndim == 1:
        data = data.reshape(1, data.shape[0], data.shape[1])
    N, (n, m) = len(data), data[0].shape

    if type(title) == str:
        if N > 1:
            title = ["%s %i" % (title, i) for i in range(N)]
        else:
            title = [title]

    rows, cols = (2, (N + 1) // 2) if N > 5 else (1, N)
    fig = plt.figure(figsize=(5 * cols, 4 * rows))

    pointSize = max(min(60 // (n**0.5),3),0.3)

    colors = plt.get_cmap('tab20')

    for i, x in enumerate(data):
        ax = fig.add_subplot(rows, cols, i + 1)
        x_reduction = red(x)
        ax.set_title(title[i])
        ax.scatter(x_reduction[:, 0], x_reduction[:, 1], s=pointSize, color=[colors(i) for i in y])
        if label is not None and label:
            if label is True:
                label = [str(s) for s in range(len(x))]
            for x_i, y_i, s_i in zip(x_reduction[:, 0], x_reduction[:, 1], label):
                plt.text(x_i, y_i, s_i)
    return fig


def reducer(method=None, random_state=0, min_dist=0.1):
    def null_reducer(x):
        return x
    if method is None:
        method = "umap"
    if method == "raw":
        return null_reducer
    elif method == "umap":
        import umap
        red = umap.UMAP(random_state=random_state, min_dist=min_dist).fit_transform
    elif method == "t_sne":
        from sklearn.manifold import TSNE
        red = TSNE(random_state=random_state).fit_transform
    else:
        raise ValueError(f"unknown method {method}!")
    return red


def t_sne(data, y, title="Channel", mask=None, random_state=None, label=False):
    if mask is None:
        mask = []
    return _show(data, y, title=title, mask=mask, label=label,
                 red=reducer(method="t_sne", random_state=random_state))


def u_map(data, y, title="Channel", mask=None, random_state=None, label=False, min_dist=0.5):
    if mask is None:
        mask = []
    return _show(data, y, title=title, mask=mask, label=label,
                 red=reducer(method="umap", random_state=random_state, min_dist=min_dist))





########################################## Benchmark ############################################


def calc_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(np.max(w) - w)
    acc = sum([w[i, j] for i, j in zip(*ind)]) * 1.0 / y_pred.size
    return np.round(acc, 5)


def calc_ari(y_true, y_pred):
    return np.around(adjusted_rand_score(y_true, y_pred), 5)


def calc_nmi(y_true, y_pred):
    return np.around(normalized_mutual_info_score(y_true, y_pred), 5)


def benchmark(y_true, y_pred, d=True):
    acc = calc_acc(y_true, y_pred)
    ari = calc_ari(y_true, y_pred)
    nmi = calc_nmi(y_true, y_pred)
    res = f"acc: {acc} \tari: {ari} \tnmi: {nmi}"
    if d:
        print(res)
    return res


def c_kmeans(x, cluster_num, n_init=10, random_state=None):
    from sklearn.cluster import KMeans
    model = KMeans(n_clusters=cluster_num, init="k-means++", n_init=n_init, random_state=random_state, max_iter=500)
    score = model.fit_transform(x)
    pred = model.predict(x)
    return pred, score


def label_trans(y_true, pred):
    """
    y_true: shape (N,1) or (N)
    pred: shape (N,K)
    """
    y_true = y_true.astype(np.int64)
    y_true = y_true.reshape(-1, 1)
    pred = pred.reshape(len(y_true), -1)
    for i in range(pred.shape[1]):
        y_pred = pred[:, i]
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for j in range(y_pred.size):
            w[y_pred[j], y_true[j]] += 1
        ind = linear_assignment(np.max(w) - w)
        ind = np.array(ind).T
        pred[:, i] = ind[y_pred, 1]





########################################## General ############################################


def mklog(s):
    print(s, end='\r')


def set_seed(seed=1111):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def preprocess(expr_df, cell_type, highly_genes=8000):
    adata = sc.AnnData(
        X=expr_df.values,
        var=pd.DataFrame(index=expr_df.columns),
        obs=pd.DataFrame(data={"cell_type": cell_type, "batch": [0] * len(cell_type)},
                         index=[f"cell_{i}" for i in range(len(cell_type))])
    )
    sc.pp.filter_genes(adata, min_counts=1)
    sc.pp.normalize_total(adata, target_sum=1e5)
    sc.pp.log1p(adata)
    n_top_genes = highly_genes if type(highly_genes) == int else int(highly_genes * adata.X.shape[1])
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, subset=True)
    sc.pp.scale(adata)
    adata.raw = sc.AnnData(
        X=expr_df[adata.var.index].values.astype(int),
        var=pd.DataFrame(index=adata.var.index),
        obs=pd.DataFrame(data={"cell_type": cell_type, "batch": [0] * len(cell_type)},
                         index=[f"cell_{i}" for i in range(len(cell_type))])
    )
    return adata





########################################## GFM ############################################


class UnionFind:
    def __init__(self, size):
        self.parent = [i for i in range(size)]
        self.rank = [1] * size

    def find(self, p):
        if self.parent[p] != p:
            self.parent[p] = self.find(self.parent[p])
        return self.parent[p]

    def union(self, p, q):
        rootP = self.find(p)
        rootQ = self.find(q)

        if rootP != rootQ:
            if self.rank[rootP] > self.rank[rootQ]:
                self.parent[rootQ] = rootP
            elif self.rank[rootP] < self.rank[rootQ]:
                self.parent[rootP] = rootQ
            else:
                self.parent[rootQ] = rootP
                self.rank[rootP] += 1

    def connected(self, p, q):
        return self.find(p) == self.find(q)


def get_core_sub_graph_genes(adata, t=0.2):
    esp = 1e-6
    expr_matrix = adata.X
    n_genes = expr_matrix.shape[1]
    corr = pearson(expr_matrix).to_numpy()
    corr = np.clip(corr, -1 + esp, 1 - esp)
    corr = 0.5 * np.log((1 + corr) / (1 - corr))
    pairs = np.where(abs(corr) >= t)
    del corr
    uf = UnionFind(n_genes)
    for i, (a, b) in enumerate(np.array(pairs).T):
        if a >= b:
            continue
        uf.union(a, b)
    ids_map = {i: [] for i in range(n_genes)}
    for i in range(n_genes):
        ids_map[uf.find(i)].append(i)
    key, size = sorted([[k, len(v)] for k, v in ids_map.items()], key=lambda x: x[1], reverse=True)[0]
    print("The size of origin gfm is:", size)
    return adata.var.index[np.array(ids_map[key])].to_numpy()


def extend_gfm(adata, genes, t=0.2, d=3):
    esp = 1e-6
    expr_matrix = adata.X
    corr = pearson(expr_matrix).to_numpy()
    corr = np.clip(corr, -1 + esp, 1 - esp)
    corr = 0.5 * np.log((1 + corr) / (1 - corr))
    old_gene_count = len(genes)
    for i in range(d):
        mask = np.any(pd.DataFrame(data=corr, index=adata.var.index).loc[genes] >= t, axis=0)
        genes = adata.var.index[mask].to_numpy()
        if old_gene_count == len(genes):
            break
        old_gene_count = len(genes)
    return genes


def get_cutoff(adata, gfm, target=3000):
    low, high = 5, 100
    gfm = list(set(gfm) & set(adata.var.index))
    while low<high:
        mid = (low+high+1)//2
        gene_list = extend_gfm(adata, gfm, mid/100)
        if len(gene_list)>= target:
            low = mid
        else:
            high = mid - 1
    return low/100


def load_gfm(dbname="muraro", index=0):
    with open(f"./GFMs/{dbname}/{index + 1}.txt", "r") as fg:
        gfm = set([s.replace("\n", "").replace("\t", "") for s in fg.readlines() if s[0] != "#"])
    return list(gfm)
