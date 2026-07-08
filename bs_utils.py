import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

# SIMPLIFICATION: Flat term structure
RISK_FREE_RATE = 0.045

# SIMPLIFICATION: Constant S&P 500 blended dividend yield (ignores discrete ex-div effects)
DIVIDEND_YIELD = 0.013

def _d1_d2(S, K, T, r, sigma, q):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2

def black_scholes_price(S, K, T, r, sigma, option_type, q=DIVIDEND_YIELD):
    """Calculates the standard European Black-Scholes price with a continuous dividend yield."""
    if T <= 0 or sigma <= 0:
        # Return intrinsic value at expiration or zero-volatility limit
        return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0) if option_type == 'call' else max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)
        
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    
    if option_type == 'call':
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

def bs_delta(S, K, T, r, sigma, option_type, q=DIVIDEND_YIELD):
    if T <= 0 or sigma <= 0: 
        return np.exp(-q * T) if (option_type == 'call' and S > K) else (-np.exp(-q * T) if (option_type == 'put' and S < K) else 0.0)
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    if option_type == 'call':
        return np.exp(-q * T) * norm.cdf(d1)
    elif option_type == 'put':
        return np.exp(-q * T) * (norm.cdf(d1) - 1.0)

def bs_gamma(S, K, T, r, sigma, q=DIVIDEND_YIELD):
    if T <= 0 or sigma <= 0: return 0.0
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))

def bs_vega(S, K, T, r, sigma, q=DIVIDEND_YIELD):
    """
    Calculates Vega per 1.0 (100%) change in implied volatility.
    Note: This raw scaling is strictly required for the Newton-Raphson derivative step. 
    Do not rescale to 1% here.
    """
    if T <= 0 or sigma <= 0: return 0.0
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

def bs_theta(S, K, T, r, sigma, option_type, q=DIVIDEND_YIELD):
    if T <= 0 or sigma <= 0: return 0.0
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    term1 = -(S * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    if option_type == 'call':
        return term1 + q * S * np.exp(-q * T) * norm.cdf(d1) - r * K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        return term1 - q * S * np.exp(-q * T) * norm.cdf(-d1) + r * K * np.exp(-r * T) * norm.cdf(-d2)

def bs_rho(S, K, T, r, sigma, option_type, q=DIVIDEND_YIELD):
    if T <= 0 or sigma <= 0: return 0.0
    _, d2 = _d1_d2(S, K, T, r, sigma, q)
    if option_type == 'call':
        return K * T * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        return -K * T * np.exp(-r * T) * norm.cdf(-d2)

def implied_volatility(market_price, S, K, T, r, option_type, q=DIVIDEND_YIELD):
    """
    Hybrid solver for Implied Volatility.
    
    Design Justification:
    Attempts Newton-Raphson first for extreme speed via closed-form vega derivatives. 
    If vega < 1e-8 or NR fails to converge, falls back to robust Brentq bounded search.
    
    Note on Validation:
    Since SPX is European-style, this solver should converge to a much tighter match 
    against yfinance's reported IV than an American-style SPY attempt would. This validates
    our model-instrument fit, not just mathematical solver correctness.
    """
    # Arbitrage / intrinsic value bounds check (European lower bound)
    intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0) if option_type == 'call' else max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)
    if market_price < intrinsic or market_price <= 0 or T <= 0:
        return np.nan

    # 1) Newton-Raphson Method
    MAX_ITER = 50
    TOL = 1e-6
    sigma = 0.3  # Initial guess (30% IV)
    
    for _ in range(MAX_ITER):
        price = black_scholes_price(S, K, T, r, sigma, option_type, q)
        diff = price - market_price
        if abs(diff) < TOL:
            return sigma
            
        vega = bs_vega(S, K, T, r, sigma, q)
        # Derivative is functionally zero, NR denominator explodes
        if vega < 1e-8:
            break
            
        sigma = sigma - (diff / vega)
        # Bounds safety check
        if sigma < 0.001 or sigma > 5.0:
            break

    # 2) Brentq Bounded Search (Fallback)
    def objective(sig):
        return black_scholes_price(S, K, T, r, sig, option_type, q) - market_price
        
    try:
        # Bound search tightly between 0.1% and 500% IV
        res = brentq(objective, 0.001, 5.0, xtol=1e-6, maxiter=100)
        return res
    except (ValueError, RuntimeError):
        return np.nan
