#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn import datasets
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from sklearn.datasets import make_spd_matrix
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.special import digamma
from scipy.special import gammaln


# # Question 1

# In[2]:


path_data = "p4dataset2023.txt"
genomes = np.loadtxt(path_data,dtype=str)
print(genomes)

#neglect the first 3 columns
genomes_ = genomes[:, 3:]

#binary matrix
modes = stats.mode(genomes_, axis=0).mode
X = (genomes_ != modes).astype(int)

X.shape
print(X)


# In[3]:


pca = PCA()
pca.fit(X)
X_pca = pca.transform(X)
print(X_pca.shape)


# In[18]:


population_abb = {
    "ACB": "African Caribbean",
    "GWD": "Gambian",
    "ESN": "Esan",
    "MSL": "Mende",
    "YRI": "Yoruba",
    "LWK": "Luhya",
    "ASW": "African American"
}


population_list = {
    "ACB": [],
    "GWD": [],
    "ESN": [],
    "MSL": [],
    "YRI": [],
    "LWK": [],
    "ASW": []
}


# In[17]:


pca = PCA(n_components=2)
X_PCA_2 = pca.fit_transform(X)

for i in range(len(genomes)):
    population = genomes[i][2] 
    population_list[population].append(X_PCA_2[i])

plt.figure(figsize=(10, 8))
for population, pca_values in population_list.items():
    pca_values = np.array(pca_values)
    plt.scatter(pca_values[:, 0], pca_values[:, 1], 
                label=population_abb[population])

plt.legend()
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA Projection of X by Population')
plt.show()


# In[21]:


#1.3
sex_abb = {
    "F":[],
    "M":[]
}
sex_list = {
    "F":"Female",
    "M":"Male"
}


# In[25]:



def plot_pca(ax, data, labels, title, x_label, y_label):

    for key, value in data.items():
        value = np.array(value)
        ax.scatter(value[:, 0], value[:, 1], label=labels[key])
    ax.legend()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

pca = PCA(n_components=3)
X_PCA_3 = pca.fit_transform(X)

population_dict = {pop: [] for pop in set(genome[2] for genome in genomes)}
sex_dict = {sex: [] for sex in set(genome[1] for genome in genomes)}

for i, genome in enumerate(genomes):
    population_dict[genome[2]].append(X_PCA_3[i, [0, 2]])
    sex_dict[genome[1]].append(X_PCA_3[i, [0, 2]])

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
plot_pca(axs[0], population_dict, population_abb, 'PCA by Population', 'First Principal Component', 'Third Principal Component')
plot_pca(axs[1], sex_dict, sex_list, 'PCA by Sex', 'First Principal Component', 'Third Principal Component')

plt.tight_layout()
plt.show()


# In[27]:


#1.6
pca = PCA(n_components=3)
pca.fit(X)

ys = np.abs(pca.components_[2])

fig, ax = plt.subplots(figsize=(20, 5))
ax.plot(ys)
ax.set_xlabel('Nucleobase Index')
ax.set_ylabel('Absolute Value of the Third Principal Component')
ax.set_title('PCA Analysis of Nucleobase vs the Third Principal Component')
plt.show()

