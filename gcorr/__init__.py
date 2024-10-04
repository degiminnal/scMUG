import pandas as pd
import numpy as np
from ctypes import *
from time import time

def pearson(data): 
    '''
    data: str, DataFrame, nadrray
    the shape of data is limited to rows as cells ans columns as genes
    '''
    t0 = time()
    dllApi =  cdll.LoadLibrary('gcorr')
    M = c_int32()
    dllApi.get_m(byref(M))
    M = M.value

    if type(data)==str or type(data)==pd.DataFrame:
        df = pd.read_csv(data, header=0) if type(data)==str else data
        ys = [y for y in ["y1", "y2"] if y in df.columns]
        df.drop(ys, axis=1, inplace=True)
        names = df.columns
        array = df.T.to_numpy()
    elif type(data)==np.ndarray:
        names = [i for i in range(data.shape[1])]
        array = data.T
    else:
        raise ValueError("data type should be csv filename, pd.DataFrame or np.ndarray")
    
    m, n = array.shape
    print("cells:%d genes:%d" % (n, m))

    float_arr = c_float*(m*M)

    array = array.astype(np.float32)

    _a = array.ctypes.data_as(c_char_p)

    corrs = [float_arr() for _ in range((m+M-1)//M)]

    print("load data: %.2fs" % (time()-t0))

    dllApi.pearson(m, n, _a,*tuple([byref(corr) for corr in corrs])) 

    print("calculate: %.2fs" % (time()-t0))

    corrs_np = [np.frombuffer(buf, dtype=np.float32).reshape(-1, m) for buf in corrs]  
    corr_np = np.concatenate(corrs_np, axis=0)[:m]
    corr = pd.DataFrame(corr_np, columns=names, index=names).fillna(0)

    print("return df: %.2fs" % (time()-t0))

    return corr


if __name__ == '__main__':
    data = "camp1.csv"
    df = pd.read_csv(data, header=0) if type(data)==str else data
    ys = [y for y in ["y1", "y2"] if y in df.columns]
    df.drop(ys, axis=1, inplace=True)
    corr = pearson(df)
    corr = pearson(df)
    corr = pearson(df)
    corr = pearson(df)
    corr = pearson(df)
    corr = pearson(df)
    corr = pearson(df)
    print(corr)
    input()


