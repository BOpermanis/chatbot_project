import numpy as np
import sklearn.datasets as skl
#import sklearn.cluster as skl_cl
import matplotlib.pyplot as plt
import pickle
import matplotlib.cm as cm

home_dir = "/home/bruno/PycharmProjects/test/module"

data, cl = skl.make_blobs(n_samples=100,centers=3)

# ind0 = [i for i in range(cl.shape[0]) if cl[i]==0]
# ind1 = [i for i in range(cl.shape[0]) if cl[i]==1]
# colors0 = cm.rainbow(np.linspace(0, 0.5, len(ind0)))
# colors1 = cm.rainbow(np.linspace(0.5,1, len(ind1))

def get_data(k):
    x = []
    y = []
    for i in range(cl.shape[0]):
        if cl[i]==k:
            x.append(data[i,0])
            y.append(data[i,1])
    
    return x, y


for k in range(3):
    x, y = get_data(k)
    v = np.random.rand(3,1)
    plt.scatter(x, y,c = tuple(v[:, 0])) #, s=area, c=colors, alpha=0.5)

plt.show()
pickle.dump(data, open("data.pickle","wb"))

#np.savetxt("data.csv", data, fmt='%.18e', delimiter=',')

m0 = np.mean(data[:,0])
m1 = np.mean(data[:,1])
weights = []
for i in range(data.shape[0]):
    weights.append(10 if data[i,0]>m0 else 1)

pickle.dump(weights, open("weights.pickle","wb"))
