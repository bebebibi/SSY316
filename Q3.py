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


# # Q3.3.8

# In[233]:


def load_vocab(filename):
    vocab = {}
    with open(filename, 'r') as file:
        for line in file:
            word, word_id = line.strip().split('\t')
            vocab[word] = int(word_id)
    return vocab

def load_corpus(filename, vocab):
    corpus = []
    with open(filename, 'r') as file:
        for line in file:
            doc = [vocab[word] for word in line.strip().split() if word in vocab]
            corpus.append(doc)
    return corpus



vocab = load_vocab('wiki_vocab.txt')
corpus = load_corpus('wiki_docs.txt', vocab)


# In[230]:


K = 10  
V = len(vocab)  
D = len(corpus) 


# Dirichlet prior parameter
alpha = np.ones(K) * 0.7 #tune

# Beta prior parmt 
a, b = 1,2  #tune

gamma = np.random.dirichlet(alpha, D)
mu = np.random.rand(D, max(len(doc) for doc in corpus), K)
eta = np.random.rand(D, max(len(doc) for doc in corpus))
lambda_param = np.random.rand(D, 2)  


beta = np.random.dirichlet(np.ones(V), K)
T = np.random.dirichlet(np.ones(K), K)


# In[148]:


def update_gamma(mu, alpha):

    gamma = np.zeros((D, K))
    for d in range(D):  
        for i, word in enumerate(corpus[d]):
            gamma[d] += mu[d, i]
        gamma[d] += alpha  
    return gamma

def update_mu(beta, gamma, T, eta, corpus):
   
    D = len(corpus)  
    K = beta.shape[0]  
    V = beta.shape[1] 
    mu = np.zeros((D, max(len(doc) for doc in corpus), K))
    
    for d in range(D):
        for i, word in enumerate(corpus[d]):
            for k in range(K):
                if i == 0:

                    mu[d, i, k] = np.exp(digamma(gamma[d, k])) * beta[k, word]
                else:
                    sum_transitions = 0
                    for prev_k in range(K):
                        #previous
                        transition_prob = T[prev_k, k]
                        sum_transitions += mu[d, i-1, prev_k] * transition_prob

                    mu[d, i, k] = eta[d, i] * np.exp(digamma(gamma[d, k])) * beta[k, word] +                                   (1 - eta[d, i]) * sum_transitions * beta[k, word]

            mu[d, i] /= np.sum(mu[d, i])  
    return mu






def update_eta(mu, lambda_param, T, corpus):

    D = len(corpus)  
    eta = np.zeros((D, max(len(doc) for doc in corpus)))
    for d in range(D):
        for i in range(1, len(corpus[d])):  

            sum_transitions = 0
            for k in range(K):
                for l in range(K):
                    sum_transitions += mu[d, i-1, k] * T[k, l] * mu[d, i, l]

            eta[d, i] = lambda_param[d, 0] * sum_transitions /                         (lambda_param[d, 0] * sum_transitions + lambda_param[d, 1])
    return eta



def update_lambda(eta, a, b):

    lambda_d = np.zeros((D, 2))
    for d in range(D):
        lambda_d[d, 0] = a + np.sum(eta[d])
        lambda_d[d, 1] = b + len(corpus[d]) - np.sum(eta[d])
    return lambda_d

def variational_e_step():
    global mu, eta, gamma, lambda_param
    gamma = update_gamma(mu, alpha)
    mu = update_mu(beta, gamma, T, eta, corpus)  
    eta = update_eta(mu, lambda_param, T, corpus)
    lambda_param = update_lambda(eta, a, b)



# In[150]:


def update_beta(corpus, mu):

    beta = np.zeros((K, V))
    for d in range(D):
        for i, word in enumerate(corpus[d]):
            beta[:, word] += mu[d, i]

    beta /= np.sum(beta, axis=1, keepdims=True)
    return beta


def update_transition_matrix(mu, eta):

    T = np.zeros((K, K))
    for d in range(D):
        for i in range(1, len(corpus[d])):
            for k in range(K):
                for l in range(K):
                    T[k, l] += mu[d, i-1, k] * mu[d, i, l] * (1 - eta[d, i])

    T /= np.sum(T, axis=1, keepdims=True)
    return T


def variational_m_step():

    global beta, T
    beta = update_beta(corpus, mu)
    T = update_transition_matrix(mu, eta)


# In[231]:


max_iter = 100
for iteration in range(max_iter):
    variational_e_step()
    variational_m_step()


reverse_vocab = {idx: word for word, idx in vocab.items()}


def print_top_words(beta, reverse_vocab, n_top_words=10):
    for i in range(len(beta)):
        print(f"Topic #{i}:")
        top_words_idx = beta[i].argsort()[:-n_top_words-1:-1]
        top_words = [reverse_vocab[idx] for idx in top_words_idx]
        print(" ".join(top_words))

print_top_words(beta, reverse_vocab)


# # 3.3.9
# 

# In[227]:


reverse_vocab = {idx: word for word, idx in vocab.items()}


def print_top_words_per_topic(beta, reverse_vocab, n_top_words=10):
    for i in range(len(beta)):
        print(f"Topic #{i + 1}:")
        top_words_idx = beta[i].argsort()[-n_top_words:][::-1]
        top_words = [reverse_vocab[idx] for idx in top_words_idx]
        print(" ".join(top_words))
        
        
        
print_top_words_per_topic(beta, reverse_vocab)


# In[215]:


def compute_elbo(beta, T, gamma, mu, eta, lambda_param, alpha, a, b, corpus):

    elbo = 0

    for d in range(D):  
        doc = corpus[d]
        for i in range(len(doc)):  
            word = doc[i]
            for k in range(K):  
                
                if i == 0:  
                    elbo += mu[d, i, k] * (np.log(beta[k, word]) + digamma(gamma[d, k]) - digamma(np.sum(gamma[d])))
                else:
                    #topic transit
                    for l in range(K):
                        elbo += mu[d, i-1, l] * mu[d, i, k] * (np.log(T[l, k]) + np.log(beta[k, word]))
                        elbo += mu[d, i, k] * (eta[d, i] * (digamma(gamma[d, k]) - digamma(np.sum(gamma[d]))))

    for d in range(D):
        # gamma
        elbo -= np.sum((gamma[d] - 1) * (digamma(gamma[d]) - digamma(np.sum(gamma[d]))))
        # mu
        elbo -= np.sum(mu[d, :, :] * np.log(mu[d, :, :] + 1e-10))
        # eta
        elbo -= np.sum(eta[d, 1:] * np.log(eta[d, 1:] + 1e-10) + (1 - eta[d, 1:]) * np.log(1 - eta[d, 1:] + 1e-10))

    # prior effects
    elbo += D * (gammaln(np.sum(alpha)) - np.sum(gammaln(alpha)))  
    elbo += D * (gammaln(a + b) - gammaln(a) - gammaln(b))  

    return elbo


# In[229]:


max_iter = 100
previous_elbo = float('-inf')


convergence_threshold = 0.2


for iteration in range(max_iter):
    variational_e_step()
    variational_m_step()

    current_elbo = compute_elbo(beta, T, gamma, mu, eta, lambda_param, alpha, a, b, corpus)
    print(f"Iteration {iteration + 1}: ELBO = {current_elbo}")

    # check
    if abs(current_elbo - previous_elbo) < convergence_threshold:
        print(f"Converged at iteration {iteration + 1}")
        break

    previous_elbo = current_elbo


# In[234]:


print(f"Final ELBO: {previous_elbo}")


# In[235]:


def print_top_words(beta, reverse_vocab, n_top_words=10):
    for i in range(len(beta)):
        print(f"Topic #{i + 1}:")
        top_words_idx = beta[i].argsort()[-n_top_words:][::-1]
        top_words = [reverse_vocab[idx] for idx in top_words_idx]
        print(" ".join(top_words))

print_top_words(beta, reverse_vocab)

