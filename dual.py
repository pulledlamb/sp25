import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from datetime import datetime
import itertools as it
import warnings
warnings.filterwarnings("ignore")


t0 = time()
# Set random seed for reproducibility
np.random.seed(42)

# simulation parameters
runs = 5
write = True
M = [10, 20] # number of time steps
I1 = np.array([4, 6]) * 4096  # itration for regression
I2 = np.array([1, 2]) * 1024  # iteration for valuation
J = [50, 75] # number of iterations for nested MCS
reg = [5, 9] # regression degree
AP = [True, False] # antithetic paths for variance reduction, negate random shocks so that cov is negative in varY1 + varY2 - covY1Y2
MM = [True, False] # moment matching of risk-neutral measure
ITM = [True, False] # in-the-money paths

results = pd.DataFrame()


def generate_random_numbers(N, antithetic = False, moment_match = False):
    if antithetic:
        Z1 = np.random.normal(0, 1, N // 2)
        Z2 = -Z1
        Z = np.concatenate((Z1, Z2))
    else:
        Z = np.random.normal(0, 1, N)

    if moment_match:
        Z = (Z - np.mean(Z)) / np.std(Z)
    
    return Z

def simulate_stock_paths(S0, r, sigma, T, N, M, antithetic = False, moment_match = False):
    dt = T / M
    paths = np.zeros((N, M + 1))
    paths[:, 0] = S0
    Z = generate_random_numbers(N, antithetic, moment_match)
    
    for t in range(1, M + 1):
        paths[:, t] = paths[:, t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    
    return paths


def regression_coeffs(X, Y, degree):
    return np.polyfit(X, Y, deg = degree)

def continuation_value(coeffs, X):
    return np.polyval(coeffs, X)

def exercise_decision(payoff, cont_value):
    exercise = payoff > cont_value
    return exercise

def nested_monte_carlo(St, J, r, sigma, T, M, antithetic = False, moment_match = False):
    dt = T / M
    Z = generate_random_numbers(J, antithetic, moment_match)    
    paths = St * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    return paths

def lsm_backward_induction(paths, payoff_func, df, reg_degree, only_itm):
    M = paths.shape[1] - 1
    V = np.zeros((paths.shape[0], M + 1))
    V[:, -1] = payoff_func(paths[:, -1])  # payoff at maturity for each simulated path
    coeffs_list = np.zeros((M + 1, reg_degree + 1))  # regression coefficients for each time step

    for t in range(M - 1, 0, -1):
        X = paths[:, t]
        Y = V[:, t + 1] * df  # discounted future values

        if only_itm:
            X_itm = np.compress(payoff_func(paths[:, t]) > 0, X)
            Y_itm = np.compress(payoff_func(paths[:, t]) > 0, Y)
            if len(X_itm) == 0:
                coeffs = np.zeros(reg_degree + 1) # no in-the-money paths, set regression coefficients to zero
            else:
                coeffs = regression_coeffs(X_itm, Y_itm, reg_degree)
        else:
            coeffs = regression_coeffs(X, Y, reg_degree)

        coeffs_list[t, :] = coeffs  # store regression coefficients

        cont_value = continuation_value(coeffs, X)
        payoff = payoff_func(X)
        exercise = exercise_decision(payoff, cont_value)

        V[exercise, t] = payoff_func(paths[exercise, t])
        V[~exercise, t] = cont_value[~exercise]

    return V, coeffs_list


def primal_valuation(paths, payoff_func, coeffs_list, df):
    M = paths.shape[1] - 1
    V = np.zeros((paths.shape[0], M + 1))
    V[:, -1] = payoff_func(paths[:, -1])  # payoff at maturity for each simulated path

    for t in range(M - 1, 0, -1):
        X = paths[:, t]
        coeffs = coeffs_list[t]
        cont_value = continuation_value(coeffs, X)

        payoff = payoff_func(X)
        exercise = exercise_decision(payoff, cont_value)

        V[exercise, t] = payoff_func(paths[exercise, t])
        V[~exercise, t] = cont_value[~exercise]

    return np.mean(V[:, 1]) * df

def dual_valuation(paths, payoff_func, coeffs_list, df, J, r, sigma, T, M, antithetic, moment_match):
    N, M1 = paths.shape
    Q = np.zeros((M1, N), dtype=float)
    U = np.zeros((M1, N), dtype=float)
    
    for t in range(1, M1):
        for j in range(N):
            coeffs = coeffs_list[t]
            Vt = max(payoff_func(paths[j, t]), continuation_value(coeffs, paths[j, t]))
            St = nested_monte_carlo(paths[j, t], J, r, sigma, T, M, antithetic, moment_match)
            Ct = continuation_value(coeffs, St)
            ht = payoff_func(St)
            VtJ = np.sum(np.where(ht > Ct, ht, Ct)) / len(St)
            Q[t, j] = Q[t - 1, j] / df + (Vt - VtJ)
            U[t, j] = max(U[t - 1, j] / df, paths[j, t] - Q[t, j])

            if t == M:
                V0 = np.maximum(U[t - 1, j] / df, np.mean(ht) - Q[t, j])
    return Q, U

def nested_mcs_american_option(S0, K, T, r, sigma, params, runs=5):
    t0 = time()
    payoff_func = lambda S: np.maximum(K - S, 0)
    for p in params:
        M_, I1_, I2_, J_, reg_, AP_, MM_, ITM_ = p
        for i in range(runs):
            dt = T / M_
            df = np.exp(-r * dt)
            # Regression phase
            paths_reg = simulate_stock_paths(S0, r, sigma, T, I1_, M_, AP_, MM_)
            V, coeffs_list = lsm_backward_induction(paths_reg, payoff_func, df, reg_, ITM_)
            # Primal valuation
            paths_val = simulate_stock_paths(S0, r, sigma, T, I2_, M_, AP_, MM_)
            V0_primal = primal_valuation(paths_val, payoff_func, coeffs_list, df)
            # Dual valuation
            Q, U = dual_valuation(paths_val, payoff_func, coeffs_list, df, J_, r, sigma, T, M_, AP_, MM_)
            U0 = np.sum(U[-1]) / I2_ * df ** M_
            AV = (V0_primal + U0) / 2
            print((i, (time() - t0) / 60, p), V0_primal, U0, AV)

if __name__ == "__main__":
    # Parameter grids
    M = [10, 20]
    I1 = [4 * 4096, 6 * 4096]
    I2 = [1 * 1024, 2 * 1024]
    J = [50, 75]
    reg = [5, 9]
    AP = [True, False]
    MM = [True, False]
    ITM = [True, False]
    params = list(it.product(M, I1, I2, J, reg, AP, MM, ITM))

    S0 = 36
    K = 40
    T = 1.0
    r = 0.06
    sigma = 0.2

    nested_mcs_american_option(S0, K, T, r, sigma, params)