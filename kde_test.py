from torchkde import KernelDensity
import torch
import matplotlib.pyplot as plt
import numpy as np

multivariate_normal = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
X = multivariate_normal.sample((50000,)) # create data
plt.figure()
plt.hist2d(X[:,0],X[:,1], bins=100)
plt.colorbar()
plt.title("from normal")
plt.show()

kde = KernelDensity(bandwidth=1.0, kernel='gaussian') # create kde object with isotropic bandwidth matrix
_ = kde.fit(X) # fit kde to data

X_new = kde.sample(50000) # sample from estimated density
plt.figure() 
plt.hist2d(X_new[:,0],X_new[:,1], bins=100)
plt.colorbar()
plt.title("from kde")
plt.show()
