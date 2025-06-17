from torchkde import KernelDensity
import torch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


# TODO, new samples should only come from existing datas used for density estimation"
# TODO test this method on higher dimension data

multivariate_normal = torch.distributions.MultivariateNormal(torch.zeros(3), torch.eye(3))
X = multivariate_normal.sample((50000,)) # create data
plt.figure()
if X.shape[1] == 1:
    plt.hist(X, bins=30)
elif X.shape[1] == 2:
    plt.hist2d(X[:,0],X[:,1], bins=100)
    plt.colorbar()
elif X.shape[1] == 3:
    hist, edges = np.histogramdd(X, bins=10)
    x_centers = (edges[0][:-1] + edges[0][1:]) / 2
    y_centers = (edges[1][:-1] + edges[1][1:]) / 2
    z_centers = (edges[2][:-1] + edges[2][1:]) / 2
    x, y, z = np.meshgrid(x_centers, y_centers, z_centers, indexing="ij")
    x = x.ravel()
    y = y.ravel()
    z = z.ravel()
    hist = hist.ravel()
    mask = hist > 0
    x, y, z, hist = x[mask], y[mask], z[mask], hist[mask]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=hist, c=hist, alpha=0.4, cmap='viridis', depthshade=True)
else:
    pass
plt.title("from normal")
plt.show()

kde = KernelDensity(bandwidth=0.3, kernel='gaussian') # create kde object with isotropic bandwidth matrix
_ = kde.fit(X) # fit kde to data

# X_new = torch.distributions.Uniform(-6,6).sample((1000,))
# X_new = torch.distributions.Uniform(torch.tensor([-6,-6],dtype=torch.float32),torch.tensor([6,6],dtype=torch.float32)).sample((50000,))
# X_new = torch.distributions.Uniform(torch.tensor([-6,-6,-6],dtype=torch.float32),torch.tensor([6,6,6],dtype=torch.float32)).sample((50000,))

X_new = X
if len(X_new.shape) == 1:
    score = kde.score_samples(X_new.unsqueeze(1)) # kde score samples calculate log(p(x))
else:
    score = kde.score_samples(X_new)

p_x = torch.exp(score) # calculate pdf of x

inv_px = (1.0 / p_x+1e-5)**1.2 # calculate inverse of the pdf
inv_px = inv_px / inv_px.sum()

# rand_num = torch.rand_like(inv_px)
# X_accepted = X_new[rand_num < inv_px] # Mask to accept points more with higher likelihood in the invense distribution
# if len(X_accepted.shape) == 1:
#     X_new = X_accepted.unsqueeze(1)
# else:
#     X_new = X_accepted

indices = torch.multinomial(inv_px, num_samples=50000, replacement=True)
X_new = X[indices]


plt.figure() 
if X_new.shape[1] == 1:
    plt.hist(X_new, bins=30)
elif X_new.shape[1] == 2:    
    plt.hist2d(X_new[:,0],X_new[:,1], bins=100)
    plt.colorbar()
elif X_new.shape[1] == 3:
    hist, edges = np.histogramdd(X_new, bins=10)
    x_centers = (edges[0][:-1] + edges[0][1:]) / 2
    y_centers = (edges[1][:-1] + edges[1][1:]) / 2
    z_centers = (edges[2][:-1] + edges[2][1:]) / 2
    x, y, z = np.meshgrid(x_centers, y_centers, z_centers, indexing="ij")
    x = x.ravel()
    y = y.ravel()
    z = z.ravel()
    hist = hist.ravel()
    mask = hist > 0
    x, y, z, hist = x[mask], y[mask], z[mask], hist[mask]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=hist, c=hist,  alpha=0.4,cmap='viridis', depthshade=True)
else:
    pass
plt.title("from kde")
plt.show()
