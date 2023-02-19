# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 15:15:33 2023

@author: szollmann
"""

import pandas as pd
pd.options.mode.chained_assignment = None
import datetime as dt
from arch import arch_model
from start_func import *


#%% Query Data

data = query_price_data(symbol = "^GDAXI",
                        start = dt.datetime(2012,12,31),
                        end = dt.datetime(2022,12,31))


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

print(dax_risk_dist_2020.measures)
        

#%% Backtest the distribution based VaR by comparing the number of violations and computing a loss function

rolling_window = 252

dax_risk_dist_backtesting = BacktestingStatic(data, backtesting_start, backtesting_end, investment, alpha, rolling_window)
dax_risk_dist_backtesting.compare_vars()

dax_risk_dist_backtesting_eval = BacktestingEval(dax_risk_dist_backtesting.compare_df_val, dax_risk_dist_backtesting.alpha)

dax_risk_dist_backtesting_eval.plot_vars()
dax_risk_dist_backtesting_eval.compute_violations()
dax_risk_dist_backtesting_eval.loss_function()

print(dax_risk_dist_backtesting_eval.violation_result)
print(dax_risk_dist_backtesting_eval.q_measure)

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

# Instaniate Backtesting of garch results
dax_risk_garch_backtesting = BacktestingGarch(data, backtesting_start, backtesting_end, investment, alpha, models, modelling_results)
dax_risk_garch_backtesting.compare_vars()

dax_risk_garch_backtesting_eval = BacktestingEval(dax_risk_garch_backtesting.compare_df_val, dax_risk_garch_backtesting.alpha)

dax_risk_garch_backtesting_eval.plot_vars()
dax_risk_garch_backtesting_eval.compute_violations()
dax_risk_garch_backtesting_eval.loss_function()

print(dax_risk_garch_backtesting_eval.violation_result)
print(dax_risk_garch_backtesting_eval.q_measure)


