from torchkde import KernelDensity
import torch
import matplotlib.pyplot as plt
import numpy as np

multivariate_normal = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
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

kde = KernelDensity(bandwidth=0.3, kernel='gaussian') # create kde object with isotropic bandwidth matrix
_ = kde.fit(X) # fit kde to data

# X_new = kde.sample(5) # sample from estimated density
X_new = torch.distributions.Uniform(torch.tensor([-6,-6],dtype=torch.float32),torch.tensor([6,6],dtype=torch.float32)).sample((100,))
print(X_new.shape)
score = kde.score_samples(X_new.unsqueeze(1))
n_score = torch.exp(score)
n_score = n_score / n_score.max()
n_score = torch.clamp(n_score, min=1e-5)

inv = 1.0 / n_score**0.5
inv = inv / inv.max()
print(inv)

rand_num = torch.rand_like(inv)
X_accepted = X_new[rand_num < inv]
print(X_accepted.shape)
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
