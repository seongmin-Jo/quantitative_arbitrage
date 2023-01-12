import numpy as np
from typing import Optional
from dataclasses import dataclass

"""Plotting
import seaborn as sns
import matplotlib.pyplot as plt 
plt.style.use('dark_background')
"""


# brownian motion
# is continuous process such that its increments (small change in value) for any time scale are drawn from a normal distribution
# A.K.A for all n (n<t), if every W(t-n) - w(t-n-1) (= increments) are independent and normally distributed (markove property)
# If the increments are defined on a unit of time, then the distribution is the standard normal, zero mean, unit variance.

def get_dW(
    T : int,
    # Optional is used to show the parameter type which is allowed to use None
    random_state : Optional[int] = None
) -> np.ndarray:
    
    """
    Tp simulate dW(discrete increments of Sample T times from a normal distribution) of a Brownian Motion.
    Random state is optional to reproduce results.
    """
    np.random.seed(random_state)
    return np.random.normal(0.0, 1.0, T)


def get_W(
    T : int,
    random_state : Optional[int] = None
) -> np.ndarray:
    
    """
    To simulate a brownian motion 
    """
    
    dW = get_dW(T, random_state)
    # to cumulate sum and then make the fist index 0 and delete last element.
    dW_cs = dW.cumsum()
    dW_cs = np.insert(dW_cs, 0, 0)[:-1]
    return dW_cs


"""Example
dW = get_dW(T=1_000)
W = get_W(T=1_000)

#----------------------------------------------------------------
# plot

import matplotlib.pyplot as plt 
import seaborn as sns

fig = plt.figure(figsize=(15, 5))

title = "Brownian motion increments"
plt.subplot(1, 2, 1)
plt.plot(dW)
plt.gca().set_title(title, fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

title = "Brownian motion path"
plt.subplot(1, 2, 2)
plt.plot(W, 'r-')
plt.gca().set_title(title, fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
"""

# brownian motion can be correlated to another brownian motion.
# Let W_1 is brownian motion and correlated with W_3 which correlation is rho and W_2 is another independent brownian motion
# Then, dW_3t = rho * dW_1t + sqrt(1 - rho^2) * dW_2t

def _get_correlated_dW(
    dW : np.ndarray,
    rho : float,
    random_state : Optional[int] = None,
) -> np.ndarray:
    """
    Sample new brownian increments which is correlated with given increments dW and rho
    """
    
    # generate brownian increments
    dW2 = get_dW(len(dW),random_state)
    
    if np.array_equal(dW2, dW):
        # dW cannot be equal to dW2
        raise ValueError(
            "Brownian Increment error, try choosing different random state."
        )
        
    dW3 = rho * dW + np.sqrt(1 - rho ** 2) * dW2
    return dW3
    
    
# To generate many correlated brownian motion (=N-dimentional Wiener process) which corr is rho

def get_corr_dW_matrix(
    T : int, # the number of samples of each process
    n_process : int, # the numper of process 
    rho : Optional[float] = None,
    random_state : Optional[int] = None,
) -> np.ndarray : 
    """
    The correlation constant rho is used to generate a new process,
    which has rho correlation to a random process already generated,
    hence rho is only an approximation to the pairwise correlation.
    
    The resulting shape of the array is (T, n_procs).
    """
    # generate random value
    rng = np.random.default_rng(random_state)
    dWs : list[np.ndarray] = []
        
    for i in range(n_process):
        # to produce diffirent brownian motion
        random_state_i = _get_random_state_i(random_state, i)
        if i == 0 or rho is None:
            dW_i = get_dW(T,random_state=random_state_i)
        else:
            # get andom process in dWs
            dW_corr_ref = _get_corr_ref_dW(dWs, i, rng)
            dW_i = _get_correlated_dW(dW_corr_ref, rho, random_state_i)
        dWs.append(dW_i)
    return np.asarray(dWs).T

    
def _get_random_state_i(
    random_state : Optional[int],
    i : int
) -> Optional[int]:
    """Add i to random_state is is int, else return None."""
    return random_state if random_state is None else random_state + i


def _get_corr_ref_dW(
    # dWs : list[np.ndarray], 
    dWs : list,
    i : int, 
    rng : np.random.Generator
) -> np.ndarray:
    """
    Choose randomly a process (dW) from the
    already generated processes (dWs).
    """
    random_proc_idx = rng.choice(i)
    return dWs[random_proc_idx]

"""Example2
T = 1_000
n_procs = 53
rho = 0.9

corr_dWs = get_corr_dW_matrix(T, n_procs, rho)

#----------------------------------------------------------------
# plot

import matplotlib.pyplot as plt 
import seaborn as sns

fig = plt.figure(figsize=(15, 5))

# paths
title = "Correlated Brownian motion paths"
plt.subplot(1, 2, 1)
plt.plot(np.cumsum(corr_dWs, axis=0))
plt.gca().set_title(title, fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# correlation
title = "Correlation matrix heatmap"
plt.subplot(1, 2, 2)
sns.heatmap(np.corrcoef(corr_dWs, rowvar=False), cmap="viridis")
plt.gca().set_title(title, fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
"""
