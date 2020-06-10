import numpy as np
import pickle as p

def generate_data(N, d, betas=None, betas_Y=None, prior_Y=0.5,
                  max_parents=4, stdnoise=0.1):
    if betas is not None:
        d = betas.shape()[0]
    else:
        if prior_Y is not None:
            betas_Y = np.float32(np.random.randn(1, d))
        betas = np.zeros((d, d), np.float32)
        for j in range(1, d):
            nj = np.random.choice(min(j, max_parents))
            pj = np.random.permutation(range(j))[:nj]
            betas[pj, j] = np.random.randn(nj)/nj

    if prior_Y is not None:
        Y = np.float32(np.less_equal(np.random.rand(N, 1), prior_Y))
        mus = Y*betas_Y
    else:
        Y = None
        mus = np.zeros((N, d), dtype=np.float32)

    X = np.zeros((N, d), dtype=np.float32)
    X[:, 0] = (2.0*np.random.randint(2, size=(N))-1.0)*mus[:, 0] + stdnoise*np.random.randn(N)
    for j in range(d):
        mus[:, j] += np.matmul(X[:, :j], betas[:j, j])
        X[:, j] = (2.0*np.random.randint(2, size=(N))-1.0)*mus[:, j] + stdnoise*np.random.randn(N)

    return X, Y, betas, betas_Y

if __name__ == '__main__':
    np.set_printoptions(linewidth=200)
    n = 300000
    data = generate_data(n, 7, prior_Y=None)
    print(data[2])
    splits = (np.cumsum([0.7, 0.15, 0.15]) * n).astype(np.int)
    data_split = np.split(data[0], splits)
    print(data_split[0].shape)
    with open('/share/jolivaunc/data/bayes_data/bayes_data4.p', 'wb') as f:
        p.dump({
            'train': data_split[0], 
            'valid': data_split[1], 
            'test': data_split[2],
            'betas': data[2]
        }, f)