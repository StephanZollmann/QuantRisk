# Risk measures
# Author: Stephan Zollmann

# import packages
import pandas as pd
pd.options.mode.chained_assignment = None
import datetime as dt
from arch import arch_model

# import own functions
from quant_risk_util import *


#%% Query Data

symbol = "^GDAXI"
start = dt.datetime(2012,12,31)
end = dt.datetime(2022,12,31)

data = query_price_data(symbol, start, end)
#%% Global params

backtesting_start = '2020'
backtesting_end = '2022'
investment = 1e6
alpha = 0.1
#%% Create VaR & CVaR based on assumptions/computations of the loss distribution

start_date = "2020"
end_date = data[-1:].index.item()

dax_risk_dist_2020 = StaticRiskMeasures(data, 0.01, start_date, end_date)  
dax_risk_dist_2020.compute_all()

print("Show VaR & CVaR for the DAX for a fixed window: ")
print(dax_risk_dist_2020.measures)
print("\n")
#%% Backtest the distribution based VaR by comparing the number of violations and computing a loss function

rolling_window = 252

dax_risk_dist_backtesting = BacktestingStatic(data, backtesting_start, backtesting_end, investment, alpha, rolling_window)
dax_risk_dist_backtesting.compare_vars()

dax_risk_dist_backtesting_eval = BacktestingEval(dax_risk_dist_backtesting.compare_df_val, dax_risk_dist_backtesting.alpha)

dax_risk_dist_backtesting_eval.plot_vars()
dax_risk_dist_backtesting_eval.compute_violations()
dax_risk_dist_backtesting_eval.loss_function()

print("Show distribution based VaR backtesting results:")
print("Actual VaR violations vs expectation given alpha")
print(dax_risk_dist_backtesting_eval.violation_result)
print("\n")
print("Q-measure results (smaller Q indicates better fit)")
print(dax_risk_dist_backtesting_eval.q_measure)
print("\n")
#%% Estimate simple garch models to forecast volatility


#always leave rescale at True otherwise convergence of optimization may fail
basic_gm = arch_model(data['Return'], p = 1, q = 1,
                      mean = 'constant', vol = 'GARCH', dist = 't', rescale = True)

exp_gm = arch_model(data['Return'], p = 1, q = 1, o = 1,
                      mean = 'constant', vol = 'EGARCH', dist = 't', rescale = True)

# zip chosen models with a name in tuple and put in a list
models = [("GARCH_1_1", basic_gm), ("EGARCH_1_1_1", exp_gm)]


model_fitting_start = "2018" # implies 2 years or 2 * 252 days to estimate each model

modelling_results = garch_modelling(models, backtesting_start, model_fitting_start)
#%% Backtest a dynamic VaR forecast (trough volatility prediction)

# Instantiate Backtesting of garch results
dax_risk_garch_backtesting = BacktestingGarch(data, backtesting_start, backtesting_end, investment, alpha, models, modelling_results)
dax_risk_garch_backtesting.compare_vars()

dax_risk_garch_backtesting_eval = BacktestingEval(dax_risk_garch_backtesting.compare_df_val, dax_risk_garch_backtesting.alpha)

dax_risk_garch_backtesting_eval.plot_vars()
dax_risk_garch_backtesting_eval.compute_violations()
dax_risk_garch_backtesting_eval.loss_function()

print("Show garch based VaR backtesting results:")
print("Actual VaR violations vs expectation given alpha")
print(dax_risk_garch_backtesting_eval.violation_result)
print("\n")
print("Q-measure results (smaller Q indicates better fit)")
print(dax_risk_garch_backtesting_eval.q_measure)
print("\n")





