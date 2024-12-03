#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import math
import numpy as np
import time
from matplotlib import pyplot as plt
from matplotlib import animation
import copy
import random


data = []


# plan is an array of 40 floating point numbers
def sim(plan):
    #data = []
    for i in range(0, len(plan)):
        if plan[i] > 1:
            plan[i] = 1.0
        elif plan[i] < -1:
            plan[i] = -1.0

    dt = 0.1
    friction = 1.0
    gravity = 0.1
    mass = [30, 10, 5, 10, 5, 10]
    edgel = [0.5, 0.5, 0.5, 0.5, 0.9]
    edgesp = [160.0, 180.0, 160.0, 180.0, 160.0]
    edgef = [8.0, 8.0, 8.0, 8.0, 8.0]
    anglessp = [20.0, 20.0, 10.0, 10.0]
    anglesf = [8.0, 8.0, 4.0, 4.0]

    edge = [(0, 1), (1, 2), (0, 3), (3, 4), (0, 5)]
    angles = [(4, 0), (4, 2), (0, 1), (2, 3)]

    # vel and pos of the body parts, 0 is hip, 5 is head, others are joints
    v = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    p = [[0, 0, -0.25, 0.25, 0.25, 0.15], [1, 0.5, 0, 0.5, 0, 1.9]]

    spin = 0.0
    maxspin = 0.0
    lastang = 0.0

    for j in range(20):
        for k in range(10):
            lamb = 0.05 + 0.1 * k
            t0 = 0.5
            if j > 0:
                t0 = plan[2 * j - 2]
            t0 *= 1 - lamb
            t0 += plan[2 * j] * lamb

            t1 = 0.0
            if j > 0:
                t1 = plan[2 * j - 1]
            t1 *= 1 - lamb
            t1 += plan[2 * j + 1] * lamb

            contact = [False, False, False, False, False, False]
            for z in range(6):
                if p[1][z] <= 0:
                    contact[z] = True
                    spin = 0
                    p[1][z] = 0

            anglesl = [-(2.8 + t0), -(2.8 - t0), -(1 - t1) * 0.9, -(1 + t1) * 0.9]

            disp = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
            dist = [0, 0, 0, 0, 0]
            dispn = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
            for z in range(5):
                disp[0][z] = p[0][edge[z][1]] - p[0][edge[z][0]]
                disp[1][z] = p[1][edge[z][1]] - p[1][edge[z][0]]
                dist[z] = (
                    math.sqrt(disp[0][z] * disp[0][z] + disp[1][z] * disp[1][z]) + 0.01
                )
                inv = 1.0 / dist[z]
                dispn[0][z] = disp[0][z] * inv
                dispn[1][z] = disp[1][z] * inv

            dispv = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
            distv = [0, 0, 0, 0, 0]
            for z in range(5):
                dispv[0][z] = v[0][edge[z][1]] - v[0][edge[z][0]]
                dispv[1][z] = v[1][edge[z][1]] - v[1][edge[z][0]]
                distv[z] = 2 * (disp[0][z] * dispv[0][z] + disp[1][z] * dispv[1][z])

            forceedge = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
            for z in range(5):
                c = (edgel[z] - dist[z]) * edgesp[z] - distv[z] * edgef[z]
                forceedge[0][z] = c * dispn[0][z]
                forceedge[1][z] = c * dispn[1][z]

            edgeang = [0, 0, 0, 0, 0]
            edgeangv = [0, 0, 0, 0, 0]
            for z in range(5):
                edgeang[z] = math.atan2(disp[1][z], disp[0][z])
                edgeangv[z] = (dispv[0][z] * disp[1][z] - dispv[1][z] * disp[0][z]) / (
                    dist[z] * dist[z]
                )

            inc = edgeang[4] - lastang
            if inc < -math.pi:
                inc += 2.0 * math.pi
            elif inc > math.pi:
                inc -= 2.0 * math.pi
            spin += inc
            spinc = spin - 0.005 * (k + 10 * j)
            if spinc > maxspin:
                maxspin = spinc
                lastang = edgeang[4]

            angv = [0, 0, 0, 0]
            for z in range(4):
                angv[z] = edgeangv[angles[z][1]] - edgeangv[angles[z][0]]

            angf = [0, 0, 0, 0]
            for z in range(4):
                ang = edgeang[angles[z][1]] - edgeang[angles[z][0]] - anglesl[z]
                if ang > math.pi:
                    ang -= 2 * math.pi
                elif ang < -math.pi:
                    ang += 2 * math.pi
                m0 = dist[angles[z][0]] / edgel[angles[z][0]]
                m1 = dist[angles[z][1]] / edgel[angles[z][1]]
                angf[z] = ang * anglessp[z] - angv[z] * anglesf[z] * min(m0, m1)

            edgetorque = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
            for z in range(5):
                inv = 1.0 / (dist[z] * dist[z])
                edgetorque[0][z] = -disp[1][z] * inv
                edgetorque[1][z] = disp[0][z] * inv

            for z in range(4):
                i0 = angles[z][0]
                i1 = angles[z][1]
                forceedge[0][i0] += angf[z] * edgetorque[0][i0]
                forceedge[1][i0] += angf[z] * edgetorque[1][i0]
                forceedge[0][i1] -= angf[z] * edgetorque[0][i1]
                forceedge[1][i1] -= angf[z] * edgetorque[1][i1]

            f = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
            for z in range(5):
                i0 = edge[z][0]
                i1 = edge[z][1]
                f[0][i0] -= forceedge[0][z]
                f[1][i0] -= forceedge[1][z]
                f[0][i1] += forceedge[0][z]
                f[1][i1] += forceedge[1][z]

            for z in range(6):
                f[1][z] -= gravity * mass[z]
                invm = 1.0 / mass[z]
                v[0][z] += f[0][z] * dt * invm
                v[1][z] += f[1][z] * dt * invm

                if contact[z]:
                    fric = 0.0
                    if v[1][z] < 0.0:
                        fric = -v[1][z]
                        v[1][z] = 0.0

                    s = np.sign(v[0][z])
                    if v[0][z] * s < fric * friction:
                        v[0][z] = 0
                    else:
                        v[0][z] -= fric * friction * s
                p[0][z] += v[0][z] * dt
                p[1][z] += v[1][z] * dt

            data.append(copy.deepcopy(p))

            if contact[0] or contact[5]:
                return p[0][5]
    return p[0][5]


###########
# The following code is given as an example to store a video of the run and to display
# the run in a graphics window. You will treat sim(plan) as a black box objective
# function and maximize it.
###########


def init():
    ax.add_patch(patch)
    ax.add_patch(head)
    return patch, head


def animate(j):
    first = []
    second = []
    for i in joints:
        first.append(data[j][0][i])
        second.append(data[j][1][i])
    a = np.array([first, second])
    a = np.transpose(a)
    patch.set_xy(a)
    head.center = (data[j][0][5], data[j][1][5])
    return patch, head


if __name__ == "__main__":
    plan = [random.uniform(-1, 1) for i in range(40)]

    data = []
    total_distance = sim(plan)
    print("Total Distance = ", total_distance)

    # draw the simulation
    fig = plt.figure()
    fig.set_dpi(100)
    fig.set_size_inches(12, 3)

    ax = plt.axes(xlim=(-1, 10), ylim=(0, 3))

    joints = [5, 0, 1, 2, 1, 0, 3, 4]
    patch = plt.Polygon([[0, 0], [0, 0]], closed=None, fill=None, edgecolor="k")
    head = plt.Circle((0, 0), radius=0.15, fc="k", ec="k")

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=len(data), interval=20, repeat=False
    )
    anim.save('animation.gif', fps=50)
    plt.show()


# In[3]:


# Metropoli Hastings Optimization

def metropoli_hastings_optimization(num_iterations, sigma):
    curr_plan = [random.uniform(-1, 1) for i in range(40)]
    data = []
    curr_dist = sim(curr_plan)
    
    best_plan = curr_plan
    best_dist = curr_dist
    
    for i in range(num_iterations):
        proposed_plan = curr_plan + np.random.normal(0, sigma, size=40)
        proposed_plan = np.clip(proposed_plan, -1, 1)
        
        data = []
        proposed_dist = sim(proposed_plan)
        accepted_rate = min(1, np.exp(proposed_dist)/np.exp(curr_dist))
        
        if np.random.rand() < accepted_rate:
            curr_plan = proposed_plan
            curr_dist = proposed_dist
            
        if i%1000 == 0:
            print("Iteration: {}, curr_dist: {}".format(i, curr_dist))
            
        if curr_dist > best_dist:
            best_plan = curr_plan
            best_dist = curr_dist
    
    return best_plan, best_dist


# In[4]:


# run optimization
num_iter = 20000
sigma = 0.1
best_plan, best_dist = metropoli_hastings_optimization(num_iter, sigma)
data = []
best_dist = sim(best_plan)
print("Metropoli Hastings Optimization, Best Distance = ", best_dist)

# draw the simulation
fig = plt.figure()
fig.set_dpi(100)
fig.set_size_inches(12, 3)

ax = plt.axes(xlim=(-1, 10), ylim=(0, 3))

joints = [5, 0, 1, 2, 1, 0, 3, 4]
patch = plt.Polygon([[0, 0], [0, 0]], closed=None, fill=None, edgecolor="k")
head = plt.Circle((0, 0), radius=0.15, fc="k", ec="k")

anim = animation.FuncAnimation(
    fig, animate, init_func=init, frames=len(data), interval=20, repeat=False
)
anim.save('animation.gif', fps=50)
plt.show()


# In[5]:


# Gradient Descent

def compute_gradient(plan, epsilon=1e-6):
    data = []
    ori_dist = sim(plan)
    grads = np.zeros_like(plan)
    for i in range(len(plan)):
        new_plan = plan.copy()
        new_plan[i] += epsilon
        new_dist = sim(new_plan)
        # print("new_dist: {}".format(new_dist))
        grads[i] = epsilon * (new_dist - ori_dist)
        
    return grads

def gradient_descent(num_iterations=5000, learning_rate=0.01, epsilon=1e-6):
    curr_plan = [random.uniform(-1, 1) for i in range(40)]
    curr_dist = sim(curr_plan)
    
    best_plan = curr_plan
    best_dist = curr_dist
    for i in range(num_iterations):
        grads = compute_gradient(curr_plan, epsilon)
        new_plan = curr_plan + learning_rate * grads
        new_dist = sim(new_plan)
        curr_plan = new_plan
        curr_dist = new_dist
        
        if i%1000 == 0:
            print("Iteration: {}, curr_dist: {}".format(i, curr_dist))
        
        if curr_dist > best_dist:
            best_plan = curr_plan
            best_dist = curr_dist
        
    return best_plan, best_dist
        


# In[6]:


# run optimization
num_iter = 10000
best_plan, best_dist = gradient_descent(num_iterations=num_iter, learning_rate=0.1, epsilon=0.5)
data = []
best_dist = sim(best_plan)
print("Gradient Descent Optimization, Best Distance = ", best_dist)

# draw the simulation
fig = plt.figure()
fig.set_dpi(100)
fig.set_size_inches(12, 3)

ax = plt.axes(xlim=(-1, 10), ylim=(0, 3))

joints = [5, 0, 1, 2, 1, 0, 3, 4]
patch = plt.Polygon([[0, 0], [0, 0]], closed=None, fill=None, edgecolor="k")
head = plt.Circle((0, 0), radius=0.15, fc="k", ec="k")

anim = animation.FuncAnimation(
    fig, animate, init_func=init, frames=len(data), interval=20, repeat=False
)
anim.save('animation.gif', fps=50)
plt.show()


# In[15]:


def differential_evolution(bounds, population_size=10, crossover_rate=0.8, scaling_factor=0.8, max_iterations=100):
    dimension = len(bounds)
    population = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(population_size, dimension))
    
    for iteration in range(max_iterations):
        for i in range(population_size):
            # Select three distinct individuals (parents)
            indices = list(range(population_size))
            indices.remove(i)
            parents = population[np.random.choice(indices, 3, replace=False)]

            # Create a trial vector through mutation and crossover
            mutant = parents[0] + scaling_factor * (parents[1] - parents[2])
            crossover_mask = np.random.rand(dimension) < crossover_rate
            trial_vector = np.where(crossover_mask, mutant, population[i])

            # Ensure trial vector is within bounds
            trial_vector = np.clip(trial_vector, bounds[:, 0], bounds[:, 1])

            # Evaluate the objective function for the trial vector
            fitness_trial = sim(trial_vector)
            fitness_current = sim(population[i])

            # Update population if the trial vector is better
            if fitness_trial > fitness_current:
                population[i] = trial_vector

        # Print the best distance for monitoring
        best_distance = sim(population[np.argmax([sim(p) for p in population])])
        print(f"Iteration {iteration + 1}, Best Distance: {best_distance}")

    # Return the best solution found
    best_solution = population[np.argmax([sim(p) for p in population])]
    return best_solution


# In[17]:


# Set your bounds for each element in the plan
bounds = np.array([[-1, 1] for _ in range(40)])

# Set DE parameters
population_size = 20
crossover_rate = 0.7
scaling_factor = 0.5
max_iterations = 2000

optimized_plan = differential_evolution( bounds, population_size, crossover_rate, scaling_factor, max_iterations)

data = []
best_dist = sim(optimized_plan)
print("DIfferential Evolution, Best Distance = ", best_dist)

# draw the simulation
fig = plt.figure()
fig.set_dpi(100)
fig.set_size_inches(12, 3)

ax = plt.axes(xlim=(-1, 10), ylim=(0, 3))

joints = [5, 0, 1, 2, 1, 0, 3, 4]
patch = plt.Polygon([[0, 0], [0, 0]], closed=None, fill=None, edgecolor="k")
head = plt.Circle((0, 0), radius=0.15, fc="k", ec="k")

anim = animation.FuncAnimation(
    fig, animate, init_func=init, frames=len(data), interval=20, repeat=False
)
anim.save('animation.gif', fps=50)
plt.show()


# In[18]:


print(optimized_plan)


# In[ ]:




