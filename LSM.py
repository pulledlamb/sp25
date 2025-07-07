# using an ordinary least squares regression to approximate the continuation value of an American option
# this provides a lower bound on the option price since exercise decision is suboptimal

# Monte Carlo with LSM
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

def simulate_stock_paths(S0, r, sigma, T, N, M):
    """Simulate N stock price paths with M time steps."""
    dt = T / M
    paths = np.zeros((N, M + 1))
    paths[:, 0] = S0
    for t in range(1, M + 1):
        Z = np.random.normal(0, 1, N)
        paths[:, t] = paths[:, t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    return paths

def plot_stock_paths(paths, n_paths=10):
    """Plot the first n_paths simulated stock price paths."""
    plt.figure(figsize=(10, 6))
    for i in range(min(n_paths, paths.shape[0])):
        plt.plot(paths[i], lw=1)
    plt.title('Simulated Stock Price Paths')
    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    plt.grid()
    plt.show()

def regression_continuation(X, Y, degree=2):
    """Fit polynomial regression and return continuation values."""
    coeffs = np.polyfit(X, Y, deg=degree)
    return np.polyval(coeffs, X), coeffs

def lsm_american_option(S0, K, T, r, sigma, N, M, payoff_func):
    """Price an American option using Least Squares Monte Carlo."""
    paths = simulate_stock_paths(S0, r, sigma, T, N, M)
    dt = T / M
    df = np.exp(-r * dt)
    V = np.zeros((N, M + 1))
    V[:, -1] = payoff_func(paths[:, -1])
    for t in range(M - 1, 0, -1):
        X = paths[:, t]
        Y = V[:, t + 1] * df
        continuation_value, _ = regression_continuation(X, Y)
        exercise = payoff_func(X) > continuation_value
        V[exercise, t] = payoff_func(X[exercise])
        V[~exercise, t] = continuation_value[~exercise]
    return np.mean(V[:, 1]) * df

if __name__ == "__main__":
    # Model parameters
    S0 = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.2
    N = 50000
    M = 100

    # Payoff function for a call option
    call_payoff = lambda S: np.maximum(S - K, 0)

    # Simulate and plot
    paths = simulate_stock_paths(S0, r, sigma, T, N, M)
    plot_stock_paths(paths)

    # Price the American option
    option_price = lsm_american_option(S0, K, T, r, sigma, N, M, call_payoff)
    print(f"American Option Price: {option_price:.2f}")