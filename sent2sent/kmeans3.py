import numpy as np
import pickle
import matplotlib.pyplot as plt

def dist(x,y):
    return np.sum((x-y)**2)

def ChooseInitialMeans(data,k):
    means = []
    for _ in range(k):
        random_centroid = []
        for i in range(data.shape[1]):
            a = min(data[:, i])
            b = max(data[:, i])
            random_centroid.append( np.random.uniform(a,b) )
        means.append(random_centroid)
    #means, clusters = mykmns.kmeans_main(X, k)
    return means

home_dir = "/media/bruno/data/chatbot_project/sent2sent"

k = 2

data = pickle.load( open( home_dir + "/data.pickle", "rb" ) )
n = data.shape[0]

n_per_cluster = int(np.ceil(n/k))


def kmeansOnce(data,k,n,n_per_cluster):
    means = ChooseInitialMeans(data, k)

    for iter in range(50):
        #print(iter)

        if iter>0:
            means = []
            for k0 in ids:
                indices = [i for i, cl in enumerate(closest_cluster) if cl == k0]
                if len(indices) > 0:
                    cut = np.take(data, indices, axis=0)
                    means.append(np.apply_along_axis(np.mean, axis=0, arr=cut))

        clusters = dict(enumerate(means))
        ids = list(clusters.keys())
        diffs = []
        for id in ids:
            diffs.append(np.apply_along_axis(lambda x: dist(x, clusters[id]), axis=1, arr=data))

        diffs = np.asarray(diffs)

        clust_sizes = dict(zip(ids, np.zeros(len(ids))))
        closest_cluster = []
        for i in range(n):
            row = diffs[:, i]
            inds_sorted = np.argsort(row)
            for id_opt in inds_sorted:
                if clust_sizes[id_opt] < n_per_cluster:
                    closest_cluster.append(id_opt)
                    clust_sizes[id_opt] += 1
                    break

    inner_diffs = []
    for k0 in ids:
        indices = [i for i, cl in enumerate(closest_cluster) if cl == k0]
        if len(indices) > 0:
            cut = np.take(diffs, indices, axis=1)
            inner_diffs.append(np.apply_along_axis(np.mean, axis=1, arr=cut)[k0])

    return ids, closest_cluster, sum(inner_diffs)



def kmeans(data,k,n,n_per_cluster,B):

    results = []
    for b in range(B):
        print(b)
        results.append(kmeansOnce(data, k, n, n_per_cluster))

    inner_diffs = [r[2] for r in results]
    opt = np.argmin(inner_diffs)
    return results[opt][0], results[opt][1]


B = 10
print(data.shape)
ids, closest_cluster = kmeans(data,k,n,n_per_cluster,B)

for k0 in ids:
    indices = [i for i, cl in enumerate(closest_cluster) if cl == k0]
    cut = np.take(data, indices, axis=0)
    x, y = cut[:,0], cut[:,1]
    v = np.random.rand(3, 1)
    plt.scatter(x, y, c=tuple(v[:, 0]))
    #print("cluster " + str(cl) + " size = " + str(len(clusters[cl])))

plt.show()