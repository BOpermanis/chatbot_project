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

k = 3

data = pickle.load( open( home_dir + "/data.pickle", "rb" ) )




means = ChooseInitialMeans(data,k)


for iter in range(10):
    print(iter)

    clusters = dict(enumerate(means))
    ids = list(clusters.keys())
    diffs = []
    for id in ids:
        diffs.append(np.apply_along_axis(lambda x: dist(x, clusters[id]), axis=1, arr=data))

    diffs = np.asarray(diffs)
    closest_cluster = np.apply_along_axis(lambda row: ids[np.argmin(row)], axis=0, arr=diffs)

    means = []
    for k0 in ids:
        indices = [i for i, cl in enumerate(closest_cluster) if cl == k0]
        if len(indices) > 0:
            cut = np.take(data, indices, axis=0)
            means.append(np.apply_along_axis(np.mean, axis=0, arr=cut))



clusters = dict(enumerate(means))
ids = list(clusters.keys())

for k0 in ids:
    indices = [i for i, cl in enumerate(closest_cluster) if cl == k0]
    cut = np.take(data, indices, axis=0)
    x, y = cut[:,0], cut[:,1]
    v = np.random.rand(3, 1)
    plt.scatter(x, y, c=tuple(v[:, 0]))
    #print("cluster " + str(cl) + " size = " + str(len(clusters[cl])))

plt.show()