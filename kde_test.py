from torchkde import KernelDensity
import torch
import matplotlib.pyplot as plt
import numpy as np

multivariate_normal = torch.distributions.MultivariateNormal(torch.zeros(1), torch.eye(1))
X = multivariate_normal.sample((50000,)) # create data
plt.figure()
if X.shape[1] == 1:
    plt.hist(X, bins=30)
elif X.shape[1] == 2:
    plt.hist2d(X[:,0],X[:,1], bins=100)
    plt.colorbar()
else:
    pass
plt.title("from normal")
plt.show()

kde = KernelDensity(bandwidth=0.1, kernel='gaussian') # create kde object with isotropic bandwidth matrix
_ = kde.fit(X) # fit kde to data

# X_new = kde.sample(5) # sample from estimated density
X_new = torch.distributions.Uniform(-4,4).sample((10000,))
score = kde.score_samples(X_new.unsqueeze(1))
n_score = torch.exp(score)
X_accepted = X_new[n_score > 3e-1]
X_new = X_accepted.unsqueeze(1)


plt.figure() 
if X_new.shape[1] == 1:
    plt.hist(X_new, bins=30)
elif X_new.shape[1] == 2:    
    plt.hist2d(X_new[:,0],X_new[:,1], bins=100)
    plt.colorbar()
else:
    pass
plt.title("from kde")
plt.show()
