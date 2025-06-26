import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Seed for reproducibility
np.random.seed(42)

# --- User Input ---
principal = float(input("Enter your principal amount: "))
tickers = input("Enter ETF tickers separated by space: ").split()

# --- Data Collection ---
end_date = datetime(2025, 6, 24)
start_date = end_date - timedelta(days=5*365)  # 5 years back
data = yf.download(tickers, start=start_date, end=end_date)['Close']

# --- Returns & Risk Metrics ---
returns = data.pct_change().dropna()
mean_returns = returns.mean() * 252  # Annualized return
cov_matrix = returns.cov() * 252     # Annualized covariance
num_assets = len(tickers)

# --- Portfolio Performance Function ---
def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate=0.02):
    portfolio_return = np.sum(mean_returns * weights) * 100  # %
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * 100  # %
    sharpe_ratio = (portfolio_return - risk_free_rate * 100) / portfolio_volatility
    return portfolio_return, portfolio_volatility, sharpe_ratio

# --- Optimization Objective with Penalty ---
def neg_sharpe_ratio_with_penalty(weights, mean_returns, cov_matrix, risk_free_rate=0.02, penalty_factor=50):
    _, _, sharpe = portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)
    weight_variance = np.var(weights)
    target_weight = 1.0 / num_assets
    deviation_penalty = np.sum((weights - target_weight) ** 2)
    return -sharpe + penalty_factor * (weight_variance + deviation_penalty)

# --- Constraints & Bounds ---
min_weight = 0.01
constraints = (
    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
    {'type': 'ineq', 'fun': lambda x: x - min_weight}
)
bounds = tuple((min_weight, 1) for _ in range(num_assets))
initial_guess = np.array([1. / num_assets] * num_assets)

# --- Optimization ---
opt_result = minimize(neg_sharpe_ratio_with_penalty, initial_guess, args=(mean_returns, cov_matrix),
                      method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 2000})
opt_weights = opt_result.x
opt_return, opt_volatility, opt_sharpe = portfolio_performance(opt_weights, mean_returns, cov_matrix)

# --- VaR & CVaR ---
daily_portfolio_returns = returns @ opt_weights
portfolio_mean = daily_portfolio_returns.mean()
portfolio_std = daily_portfolio_returns.std()
confidence_level = 0.05
z_score = -1.645

parametric_var = -(portfolio_mean + z_score * portfolio_std) * principal
historical_var = -np.percentile(daily_portfolio_returns, confidence_level * 100) * principal
tail_losses = daily_portfolio_returns[daily_portfolio_returns <= np.percentile(daily_portfolio_returns, confidence_level * 100)]
cvar = -tail_losses.mean() * principal

# --- Marginal VaR ---
def marginal_var(weights, cov_matrix, portfolio_std, confidence_level=0.05):
    z_score = -1.645
    marginal = z_score * (cov_matrix @ weights) / portfolio_std
    return marginal * principal

marginal_vars = marginal_var(opt_weights, cov_matrix / 252, portfolio_std)

# --- Incremental VaR ---
incremental_vars = []
perturbation = 0.01
for i in range(num_assets):
    new_weights = opt_weights.copy()
    new_weights[i] += perturbation
    new_weights = new_weights / np.sum(new_weights)
    new_returns = (returns @ new_weights)
    new_var = -np.percentile(new_returns, confidence_level * 100) * principal
    incremental_vars.append((new_var - historical_var) / perturbation)

# --- Cumulative Growth Chart ---
cumulative_returns = (returns @ opt_weights).cumsum() * principal + principal
plt.figure(figsize=(10, 6))
plt.plot(cumulative_returns.index, cumulative_returns, label='Portfolio Value')
plt.title('Portfolio Growth Over Time (5 Years)')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.grid(True)
plt.savefig('portfolio_growth.png')
plt.close()

# --- Portfolio Weights Chart ---
plt.figure(figsize=(8, 6))
plt.bar(tickers, opt_weights)
plt.title('Optimal Portfolio Weights')
plt.xlabel('Security')
plt.ylabel('Weight')
plt.savefig('portfolio_weights.png')
plt.close()

# --- Correlation Heatmap ---
plt.figure(figsize=(8, 6))
sns.heatmap(returns.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap of Securities (5 Years)')
plt.savefig('correlation_heatmap.png')
plt.close()

# --- Final Output ---
print(f"\n✅ Optimal Weights: {dict(zip(tickers, opt_weights))}")
print(f"📈 Expected Annual Return: {opt_return:.2f}%")
print(f"📉 Annual Volatility: {opt_volatility:.2f}%")
print(f"📊 Sharpe Ratio: {opt_sharpe:.2f}")
print(f"\n🔻 Parametric VaR (95%, 1-day): ${parametric_var:,.2f}")
print(f"🔻 Historical VaR (95%, 1-day): ${historical_var:,.2f}")
print(f"🔻 CVaR (95%, 1-day): ${cvar:,.2f}")
print(f"\n📌 Marginal VaR:")
for t, mv in zip(tickers, marginal_vars):
    print(f"  - {t}: ${mv:,.2f}")
print(f"\n📌 Incremental VaR:")
for t, iv in zip(tickers, incremental_vars):
    print(f"  - {t}: ${iv:,.2f}")
