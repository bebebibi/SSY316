#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
import numpy as np
from scipy.stats import poisson
from scipy.stats import norm
from scipy.stats import gamma
import matplotlib.pyplot as plt


# In[33]:


df = pd.read_csv("premier_league_2013_2014.dat", header=None, dtype=int)


# In[34]:


print(df)


# In[35]:


def metropolis_hastings(outcome, sigma, t, iterations=5000):
    #initialization
    # (home, att0, def0, ..., att19, def19)
    theta = np.full(41, 0.1)
    # (mu_att, mu_def, tau_att, tau_def)
    eta = np.full(4, 0.1)
    
    theta_accepted_samples = [theta]
    eta_accepted_samples = [eta]
    
    #iterations
    accepted_count = 0
    for i in range(iterations):
        theta_change = np.random.normal(0, sigma, size=len(theta))
        theta_proposed = theta + theta_change
        eta_proposed = eta + np.random.normal(0, sigma, size=len(eta))
        eta_change = np.random.normal(0, sigma, size=len(eta))
        eta_proposed = eta + eta_change
        theta[1] = 0 #att0
        theta[2] = 0 #def0
        #print("iteration: {}\nhome\n".format(j, theta_proposed[0]))
        # acceptance rate
        log_accept_rate = cal_log_prob(outcome, theta_proposed, eta_proposed) - cal_log_prob(outcome, theta, eta)
        if np.log(np.random.rand()) < log_accept_rate:
            theta = theta_proposed
            eta = eta_proposed

    for i in range(iterations):
        for j in range(t):
            theta_change = np.random.normal(0, sigma, size=len(theta))
            theta_proposed = theta + theta_change
            eta_proposed = eta + np.random.normal(0, sigma, size=len(eta))
            eta_change = np.random.normal(0, sigma, size=len(eta))
            eta_proposed = eta + eta_change
            theta[1] = 0 #att0
            theta[2] = 0 #def0
            #print("iteration: {}\nhome\n".format(j, theta_proposed[0]))
            # acceptance rate
            log_accept_rate = cal_log_prob(outcome, theta_proposed, eta_proposed) - cal_log_prob(outcome, theta, eta)
            if np.log(np.random.rand()) < log_accept_rate:
                accepted_count += 1
                theta = theta_proposed
                eta = eta_proposed
                
        theta_accepted_samples.append(theta)
        eta_accepted_samples.append(eta)
        
        #print("iteration: {}\ntheta_accepted_samples:{}\neta_accepted_samples:{}".format(i, theta_accepted_samples, eta_accepted_samples))
        if i%100 == 0:
            print("iteration: {}".format(i))
    rejection_rate = 1 - accepted_count/(iterations*t)
    return theta_accepted_samples, eta_accepted_samples, rejection_rate


# In[36]:


def cal_log_prob(outcome, theta, eta):
    log_prob = 0
    for i in range(outcome.shape[0]):
        score_home = outcome.iloc[i, 0]
        score_away = outcome.iloc[i, 1]
        home_ind = outcome.iloc[i, 2]
        away_ind = outcome.iloc[i, 3]
        
        log_theta_i0 = theta[0] + theta[1+home_ind*2] - theta[2+away_ind*2]
        log_theta_i1 = theta[1+away_ind*2] - theta[2+home_ind*2]
        
        log_prob += np.log(poisson.pmf(score_home, np.exp(log_theta_i0))) + np.log(poisson.pmf(score_away, np.exp(log_theta_i1)))
        
    # print("log_prob1: {}".format(log_prob))
    tau_0 = 0.0001
    # home
    log_prob += np.log(norm.pdf(theta[0], 0, 1/tau_0))
    # att and def
    for t in range(1, len(theta), 2):
        # att
        log_prob += np.log(norm.pdf(theta[t], eta[0], 1/eta[2]))
#         print("theta[t], eta[0], 1/eta[2]: {}, {}, {}".format(theta[t], eta[0], 1/eta[2]))
#         print("aff:{}".format(np.log(norm.pdf(theta[t], eta[0], 1/eta[2]))))
        # def
        log_prob += np.log(norm.pdf(theta[t+1], eta[1], 1/eta[3]))
#         print("theta[t+1], eta[1], 1/eta[3]: {}, {}, {}".format(theta[t+1], eta[1], 1/eta[3]))
#         print("def:{}".format(np.log(norm.pdf(theta[t+1], eta[1], 1/eta[3]))))
    #print("log_prob2: {}".format(log_prob))
    # eta
    tau_1 = 0.0001
    alpha = 0.1
    beta = 0.1
    log_prob += np.log(norm.pdf(eta[0], 0, 1/tau_1))
    log_prob += np.log(norm.pdf(eta[1], 0, 1/tau_1))
    log_prob += np.log(gamma.pdf(eta[2], alpha, scale=1/beta))
    log_prob += np.log(gamma.pdf(eta[3], alpha, scale=1/beta))
    #print("log_prob3: {}".format(log_prob))
    return log_prob


# In[6]:


iterations = 5000
sigma_list = [0.005, 0.05, 0.5]
t_list = [1, 5, 20, 50]
for sigma in sigma_list:
    for t in t_list:
        print("\n--------------------------------\n")
        print("setting: sigma={}, t={}".format(sigma, t))
        theta_accepted_samples, eta_accepted_samples, rejection_rate = metropolis_hastings(df, sigma, t, iterations=iterations)
        fig = plt.figure()
        plt.plot(range(iterations+1), [row[0] for row in theta_accepted_samples])
        plt.title(f'Trace Plot of variable home with sigma: {sigma}, time step: {t}')
        
        print("rejection_rate: {}".format(rejection_rate))


# In[37]:


best_sigma = 0.05
best_t = 5

theta_accepted_samples, eta_accepted_samples, rejection_rate = metropolis_hastings(df, best_sigma, best_t, iterations=iterations)


# In[48]:


fig = plt.figure()
plt.hist([row[0] for row in theta_accepted_samples], bins=50, color='green', alpha=0.5)
plt.title("The posterior histogram of ariable home from the MCMC samples")


# In[67]:


teams_dict = {
    0: 'Arsenal', 1: 'Aston Villa', 2: 'Cardiff City', 3: 'Chelsea', 4: 'Crystal Palace',
    5: 'Everton', 6: 'Fulham', 7: 'Hull City', 8: 'Liverpool', 9: 'Manchester City',
    10: 'Manchester United', 11: 'Newcastle United', 12: 'Norwich City', 13: 'Southampton', 14: 'Stoke City',
    15: 'Sunderland', 16: 'Swansea City', 17: 'Tottenham Hotspur', 18: 'West Bromwich Albion', 19: 'West Ham United'
}

theta_sample_array = np.array(theta_accepted_samples)
theta_column_means = np.mean(theta_sample_array, axis=0)
print(theta_column_means)
# attack
for i in range(20):
    att_ind = 1+i*2
    def_ind = 2+i*2
    plt.scatter(theta_column_means[att_ind], theta_column_means[def_ind], marker='o', label=teams_dict[i])
    plt.annotate(teams_dict[i], 
                 (theta_column_means[att_ind], theta_column_means[def_ind]),
                 fontsize=7  # Adjust the font size as needed
                 )
plt.xlim(-0.6, 0.6)
plt.ylim(-0.6, 0.2)
plt.xlabel("Estimated Attacking Strength")
plt.ylabel("Estimated Defending Strength")
plt.title("Estimated Attacking and Defending Strength for Each Team")
plt.show()


# In[ ]:




