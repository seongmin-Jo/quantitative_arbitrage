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
#      - Bt : standard Brownian motion under the probability measure P
#      - mu : deterministic part & the drift of the process, mean-reversion rate
#      - sigma : control the random process 
#      - θ : long-term mean, θ ∈ R
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
    return OUParams(mu, theta, sigma)


# 위의 것과 값이 똑같음. 왜일까 어이가 없다 그냥
def derive_ou_params(X: pd.Series,
                     dt: float = 1)  -> (np.float64, np.float64, np.float64):
    """
    Derived parametes for OU process from estimated parameters of AR(1) process
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
    return OUParams(mu, theta, sigma)


"""Example2
# generate process with random_state to reproduce results
OU_params = OUParams(mu=0.07, theta=0.0, sigma=0.001)
OU_proc = get_OU_process(1000, OU_params, random_state=7)
OU_params_hat = estimate_OU_params(OU_proc)
"""




"""
def compute_log_likelihood(
    X : np.array, 
    OU_params : OUParams, 
    increment_time : float
) -> int : 
    
    n = len(X)
    tilda_sigma_square = np.square(OU_params.sigma) * ( 1- np.exp(-2 * OU_params.mu * increment_time)) / (2 * OU_params.mu)
    tilda_sigma = tilda_sigma_square ** (1/2)
    
    # X[1:] means Xi and X[:-1] means Xi-1                                          
    summation = np.sum(np.square(X[1:] - (X[:-1] * np.exp(-OU_params.mu * increment_time))
                                 - (OU_params.theta * (1-np.exp(-OU_params.mu * increment_time)))))
    
    log_likelihood = (-np.log(2 * np.pi) / 2) - np.log(tilda_sigma) - (summation / (2*n*tilda_sigma_square))
    # since we want to maximize this total log likelihood, we need to minimize the
    #   negation of the this value (scipy doesn't support maximize)
    return -log_likelihood

def get_portfolio_table(
    asset1 : np.array,
    asset2 : np.array,
) -> pd.DataFrame :

    OU_param_df = pd.DataFrame(columns=['theta', 'mu', 
                                        'sigma', 'MLE', 'B'])
    
    if len(asset1) == len(asset2):
        increment_time = 1/(pd.to_datetime(asset2.index[-1]) - pd.to_datetime(asset2.index[0])).days
    else:
        raise ValueError(
            "Length of two dataset is not same"
        )
    # B 는 0부터 1사이의 비율 (0.01 단위로 discrete 하게 본다.)
    B_ratio_list = np.linspace(0, 1, 101)

    # Initial Value of A is 1
    # α = A/S0(1)
    alpha = 1 / asset1[0]

    for B in B_ratio_list:

        # tunning beta according to B
        # β = B / S0(2)
        beta = B / asset2[0]

        # Rebalancing 하기 위해 필요한 데이터가 1년 단위이므로 첫째날 값을 기준으로 계속 들고 있는다고 가정??
        X = np.array( alpha * asset1 - beta * asset2 )
        OU_params = estimate_OU_params(X)

        # OU parameter 를 기반으로 log likelihood 를 구함
        log_likelihood = compute_log_likelihood(X, OU_params, increment_time)
        OU_param_df = OU_param_df.append({
                                        'theta' : OU_params.theta,
                                        'mu' : OU_params.mu,
                                        'sigma' : OU_params.sigma,
                                        'MLE' : log_likelihood,
                                        'B' : B
                                        }, ignore_index=True)
    return OU_param_df
"""















import math
from math import sqrt, exp, log  # exp(n) == e^n, log(n) == ln(n)

def compute_log_likelihood(
    OU_params : tuple,
    *args : tuple
) -> int : 
    mu, theta, sigma = OU_params
    OU_params = OUParams(mu, theta, sigma)
    X, increment_time = args
    
    n = len(X)
    tilda_sigma_square = np.square(OU_params.sigma) * ( 1- np.exp(-2 * OU_params.mu * increment_time)) / (2 * OU_params.mu)
    tilda_sigma = tilda_sigma_square ** (1/2)
    
    # X[1:] means Xi and X[:-1] means Xi-1                                          
    summation = np.sum(np.square(X[1:] - (X[:-1] * np.exp(-OU_params.mu * increment_time))
                                 - (OU_params.theta * (1-np.exp(-OU_params.mu * increment_time)))))
    
    log_likelihood = (-np.log(2 * np.pi) / 2) - np.log(tilda_sigma) - (summation / (2*n*tilda_sigma_square))
    # since we want to maximize this total log likelihood, we need to minimize the
    #   negation of the this value (scipy doesn't support maximize)
    return log_likelihood

def __compute_log_likelihood(params, *args):
    '''
    Compute the average Log Likelihood, this function will by minimized by scipy.
    Find in (2.2) in linked paper

    returns: the average log likelihood from given parameters
    '''
    # functions passed into scipy's minimize() needs accept one parameter, a tuple of
    #   of values that we adjust to minimize the value we return.
    #   optionally, *args can be passed, which are values we don't change, but still want
    #   to use in our function (e.g. the measured heights in our sample or the value Pi)

    theta, mu, sigma = params
    X, dt = args
    n = len(X)

    sigma_tilde_squared = sigma ** 2 * (1 - exp(-2 * mu * dt)) / 2 * mu

    summation_term = 0

    for i in range(1, len(X)):
        summation_term += (X[i] - X[i - 1] * exp(-mu * dt) - theta * (1 - exp(-mu * dt))) ** 2

    summation_term = -summation_term / (2 * n * sigma_tilde_squared)

    log_likelihood = (-log(2 * math.pi) / 2) + (-log(sqrt(sigma_tilde_squared))) + summation_term

    return -log_likelihood
    # since we want to maximize this total log likelihood, we need to minimize the
    #   negation of the this value (scipy doesn't support maximize)

def maximized_avergae_log_likelihood(
    X : np.array, 
    dt : int
    ):
    '''
    Estimates Ornstein-Uhlenbeck coefficients (θ, µ, σ) of the given array
    using the Maximum Likelihood Estimation method

    input: X - array-like data to be fit as an OU process
    returns: θ, µ, σ, Total Log Likelihood
    '''
    sigma_inf_limit = 10e-4
    mu_inf_limit = 10e-4
    
    minimizer = minimize(
        fun = __compute_log_likelihood,
        x0 = (np.mean(X), 1, 1),# init value
        args = (X, dt),
        bounds = ((mu_inf_limit, None), (None, None), (sigma_inf_limit, None))
    )

    theta, mu, sigma = minimizer.x
    max_log_likelihood = -minimizer.fun  # undo negation from __compute_log_likelihood
    return mu, theta, sigma, max_log_likelihood


























