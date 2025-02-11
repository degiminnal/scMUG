import numpy as np
import multiprocessing


def get_mat1_job(pred ,n_sample, kmeans_times, n_channel, cluster_number, score, thread_num, i):
    mat1 = np.zeros(shape=(n_sample, n_sample), dtype=np.float32)
    idx = -1
    for c in range(n_channel):
        for t in range(kmeans_times):
            if(np.sum(score[t,c,:])==0):
                continue
            idx += 1
            if idx%thread_num==i:
                p = pred[t,c,:].flatten()
                tmp_score = score[t,c,:].reshape(-1,1)*score[t,c,:].reshape(1,-1)
                mask = np.zeros(shape=(n_sample, n_sample), dtype=bool)
                for j in range(cluster_number):
                    idxs = np.argwhere(pred[t,c,:] == j).flatten()
                    mask[idxs,:]=True
                    mask = mask * (mask.T)
                    mat1 += tmp_score * mask
                    mask[idxs,:]=False
                del p, tmp_score, mask
    return mat1

def get_mat1_job1(z):
	return get_mat1_job(z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7])

def get_mat1(pred, n_sample, kmeans_times, n_channel, cluster_number, score, thread_num):
    pool = multiprocessing.Pool(thread_num)
    data_list=[(pred, n_sample, kmeans_times, n_channel, cluster_number, score, thread_num, i) for i in range(thread_num)]
    res = pool.map(get_mat1_job1,data_list)
    pool.close()
    pool.join()
    ans = np.zeros(shape=res[0].shape, dtype=float)
    for i in range(thread_num):
        ans += res[i]
    return ans


##############################################################################################################################################

def recipMinSqrt(x, y, p=0.5):
    """
    x, y: ndarray with the same shape
    return:
        ndarray with the same shape as x and y, the value in which is pow(min(x/y, y/x), p)
    """
    x, y = np.array([x]), np.array([y])
    x_eps, y_eps = x+1e-5*(x==0), y+1e-5*(y==0)
    return np.min(np.vstack([x/y_eps,y/x_eps]), axis=0)**p

def get_mat2(n_sample, neighbourDist, dist, neighbourDistScore):
    mat2 = np.zeros(shape=(n_sample, n_sample))
    nch, ng = neighbourDist.shape
    neighProbs = recipMinSqrt(neighbourDist.reshape([nch,1,ng]), dist)
    neighProbsT = np.array([x.T for x in  neighProbs])
    score = neighbourDistScore.reshape((-1, n_sample, 1))*neighbourDistScore.reshape((-1, 1, n_sample))
    mat2 = neighProbs*neighProbsT*score
    mat2 = np.max(mat2, axis=0)
    mat2 = mat2 + np.eye(n_sample)*np.mean(mat2, axis=0)
    return mat2


































