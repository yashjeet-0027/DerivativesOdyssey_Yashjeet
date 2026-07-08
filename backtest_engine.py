import pandas as pd
import numpy as np
from bs_utils import black_scholes_price, RISK_FREE_RATE, DIVIDEND_YIELD

class VolatilityBacktester:
    def __init__(self, data_path='data/backtest_data.csv', initial_capital=100000):
        self.df = pd.read_csv(data_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.initial_capital = initial_capital

    def run_backtest(self, strategy='ML_Directional', trade_size=10):
        results = []
        capital = self.initial_capital
        equity_curve = [capital]
        dates = [self.df['Date'].iloc[0]]

        # We need the exact Close price 21 days in the future to calculate our option payoff
        self.df['Close_21d_ahead'] = self.df['Close'].shift(-21)
        trade_df = self.df.dropna(subset=['Close_21d_ahead']).copy()

        for _, row in trade_df.iterrows():
            S = row['Close']
            K = S # At-The-Money Strike
            T = 21 / 365.0
            sigma_priced = row['hv_21'] # Using historical vol as the market IV proxy
            
            # 1. Price the Straddle using your BS Engine
            call_price = black_scholes_price(S, K, T, RISK_FREE_RATE, sigma_priced, 'call', q=DIVIDEND_YIELD)
            put_price = black_scholes_price(S, K, T, RISK_FREE_RATE, sigma_priced, 'put', q=DIVIDEND_YIELD)
            straddle_cost = (call_price + put_price) * 100 * trade_size # 100 shares per contract
            
            # 2. Calculate Realized Payoff 21 days later
            future_S = row['Close_21d_ahead']
            payoff = (max(future_S - K, 0) + max(K - future_S, 0)) * 100 * trade_size

            # 3. Strategy Logic (ML vs Naive)
            if strategy == 'Always_Long':
                pnl = payoff - straddle_cost
                signal = 'Long'
            elif strategy == 'Always_Short':
                pnl = straddle_cost - payoff
                signal = 'Short'
            elif strategy == 'ML_Directional':
                # THE ALPHA: If our XGBoost model predicts higher vol than the market is pricing, BUY. Else, SELL.
                if row['Pred_XGB'] > sigma_priced:
                    pnl = payoff - straddle_cost
                    signal = 'Long'
                else:
                    pnl = straddle_cost - payoff
                    signal = 'Short'
            
            capital += pnl
            results.append({
                'Date': row['Date'],
                'Market_IV': round(sigma_priced, 4),
                'XGB_Prediction': round(row['Pred_XGB'], 4),
                'Signal': signal,
                'Straddle_Cost': round(straddle_cost, 2),
                'Payoff': round(payoff, 2),
                'PnL': round(pnl, 2)
            })
            equity_curve.append(capital)
            dates.append(row['Date'])
            
        res_df = pd.DataFrame(results)
        
        # Calculate standard Quant Metrics
        returns = res_df['PnL'] / self.initial_capital
        win_rate = (res_df['PnL'] > 0).mean()
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        peak = pd.Series(equity_curve).cummax()
        max_dd = ((peak - pd.Series(equity_curve)) / peak).max()
        
        metrics = {
            'Total Return (%)': ((capital - self.initial_capital) / self.initial_capital) * 100,
            'Win Rate (%)': win_rate * 100,
            'Sharpe Ratio': sharpe,
            'Max Drawdown (%)': max_dd * 100
        }
        
        return res_df, pd.DataFrame({'Date': dates, 'Equity': equity_curve}), metrics