import brownian_motion

import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt 
plt.style.use('dark_background')
import seaborn as sns

from sklearn.linear_model import LinearRegression

# Ornstein-Uhlenbeck process is not just stationary but also normally distributed
# ou parameter : dXt = mu(theta - Xt)dt + sigma * dBt
#      - Bt : Brownian Motion
#      - mu : deterministic part & the drift of the process 
#      - sigma : control the random process 
#      - If sigma is large enough, then mu become unsignificant for the process

@dataclass
class OUParams:
    mu: float  # mean reversion parameter
    theta: float  # asymptotic mean
    sigma: float  # Brownian motion scale (standard deviation)
        

def get_OU_process(
    T : int,
    OU_params : OUParams,
    # X_0 is initial value for the process.
    X_0 : Optional[float] = None,
    random_state : Optional[int] = None
) -> np.ndarray : 
    """Solution of the SDE
    X_t = X_0 * e^(-mu * t) 
          + theta(1-e^(-mu * t)) 
          + sigma * e^(-mu * t) * integral 0 ~ t e^(-mu * t)*dWs
    """
    t = np.arange(T,dtype='float64') # float to avoid np.exp overflow
    # part of the SDE
    exp_mu_t = np.exp(-OU_params.mu * t)
    dB = brownian_motion.get_dW(T, random_state)
    integral_B = _get_integral_B(t, dB, OU_params)
    _X_0 = _select_X_0(X_0, OU_params)
    
    return (
        _X_0 * exp_mu_t 
        + OU_params.theta * (1-exp_mu_t)
        + OU_params.sigma * exp_mu_t * integral_B
    )
    
    
def _select_X_0(
    X_0_in : Optional[float],
    OU_params : OUParams
) -> float:
    """
    X_0 is initial value for the process.
    if None, the X_0 is taken to be theta (asymptotic mean)
    """
    if X_0_in is not None:
        return X_0_in
    return OU_params.theta
    
    
def _get_integral_B(
    t : np.ndarray,
    dW : np.ndarray,
    OU_params : OUParams,
) -> np.ndarray :
    """Integral with respect to Brownian Motion (W), ∫...dW."""
    exp_mu_s = np.exp(OU_params.mu * t)
    integral_B = np.cumsum(exp_mu_s * dW)
    return np.insert(integral_B, 0, 0)[:-1]


"""Example
OU_params = OUParams(mu=0.07, theta=0.0, sigma=0.001)
OU_proc = get_OU_process(1000, OU_params)

#----------------------------------------------------
# plot
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(15, 7))

title = "Ornstein-Uhlenbeck process, "
title += r"$\mu=0.07$, $\theta = 0$, $\sigma = 0.001$"
plt.plot(OU_proc)
plt.gca().set_title(title, fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
"""

# Estimating OU parameters from data -> OLS regression 

# Estimating OU parameters from data -> OLS regression 

def estimate_OU_params(
    X_t : np.ndarray
) -> OUParams:
    """
    Estimate OU params from OLS regression.
    - X_t is a 1D array.
    Returns instance of OUParams.
    """
    y = np.diff(X_t)
    X = X_t[:-1].reshape(-1, 1)
    reg = LinearRegression(fit_intercept=True)
    reg.fit(X, y)
    # regression coeficient and constant
    mu = -reg.coef_[0]
    theta = reg.intercept_ / mu
    # residuals and their standard deviation
    y_hat = reg.predict(X)
    sigma = np.std(y - y_hat)
    
    """Fit AR(1) process from time series of price
    ts_y = ts.values[1:].reshape(-1, 1)
    ts_x = np.append(np.ones(len(ts_y)), ts.values[:-1]).reshape(2,-1).T
    
    phi = np.linalg.inv(ts_x.T @ ts_x) @ ts_x.T @ ts_y
    sigma = np.sqrt(np.sum((ts_y - ts_x @ phi) ** 2) / (len(ts_y)))
    phi = phi.reshape(-1)
    
    theta = phi[0] / (1-phi[1])
    mu = (1-phi[1]) / dt
    sigma = sigma / np.sqrt(dt)
    """
    
    """
    # X = alpha * asset1 - beta * asset2
   
    N = X.size
    Xx  = np.sum(X[0:-1])
    Xy  = np.sum(X[1:])
    Xxx = np.sum(X[0:-1]**2)
    Xxy = np.sum(X[0:-1] * X[1:])
    Xyy = np.sum(X[1:]**2)

    theta = (Xy * Xxx - Xx * Xxy) /  (N * (Xxx - Xxy) - (Xx**2 - Xx * Xy) ) 
    mu = -(1 / dt) * np.log(Xxy - theta * Xx - theta * Xy + N * theta**2) / (Xxx - 2 * theta * Xx + N * theta**2)

    pref = 2 * mu / (N * (1 - np.exp(-(2 * mu * dt))))
    term = Xyy -  2 * np.exp(-mu * dt) * Xxy + np.exp(-2 * mu * dt) * Xxx \
        - 2 * theta * (1 - np.exp(- mu * dt)) * (Xy - np.exp(-mu * dt) * Xx)\
        + N * theta**2 * (1 - np.exp(- mu * dt))**2

    sigma = np.sqrt(pref * term)
    """
    return OUParams(mu, theta, sigma)


"""Example2
# generate process with random_state to reproduce results
OU_params = OUParams(mu=0.07, theta=0.0, sigma=0.001)
OU_proc = get_OU_process(1000, OU_params, random_state=7)
OU_params_hat = estimate_OU_params(OU_proc)
"""
