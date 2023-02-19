# Risk measures util
# Author: Stephan Zollmann

import pandas as pd
pd.options.mode.chained_assignment = None
import pandas_datareader.data as web
import yfinance as yf
yf.pdr_override()
import numpy as np
from functools import wraps
import time 
from scipy.stats import norm, t#, genextreme
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

#import datetime as dt
#import scipy.stats as sps
#from scipy.integrate import quad
#from sklearn.neighbors import KernelDensity
#from arch import arch_model
#from matplotlib.gridspec import GridSpec
#import matplotlib.dates as mdates


def timing_decorator(func):  
    "Small decorator to time functions and print them when they are called."
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        
        print("Running function {} ...".format(func.__name__))
        
        start_time = time.time()
        
        result = func(*args, **kwargs)
        
        print("Time elapsed: {} seconds".format(round(time.time() - start_time)))
        print("\n")
        return result
    return wrapper

class InvalidIndexException(Exception):
    """
    Raised when the queried data does not have a datetime index.
    """
    pass

@timing_decorator
def query_price_data(symbol, start, end):
    """
    
    Parameters
    ----------
    symbol : str
        Symbol (yahoo finance) of the asset to be queried.
    start : datetime
        Start date of data to query.
    end : datetime
        End date of data to query.

    Raises
    ------
    InvalidIndexException
        Raised when the queried data does not have a datetime index.

    Returns
    -------
    asset_data : df
        Returns a dataframe with the price history of the specified asset.

    """
    
    data = web.DataReader(symbol, start, end)
    
    if data.index.inferred_type != "datetime64":
        raise InvalidIndexException
            
    asset_prices = data["Close"].copy()
    asset_prices.sort_index(inplace = True)
    
    asset_data = asset_prices.to_frame(name = "Price")
    asset_data["Return"] = asset_data.Price.pct_change()
    
    asset_data = asset_data.iloc[1:]
    
    return asset_data 
    

class RiskMeasures:
    
    """
    A class used to represent general Risk measures such as VaR and CVar.
    
    --------------------------------------------------------------------

    Attributes
    ----------
    data : DataFrame
        Dataframe of the price history of an asset
    losses : Series
        Series of the losses of the asset.
    alpha : int/float
        alpha for confidence calculations.

    """
    
    
    def __init__(self, data, alpha):
        
        self.data = data
        self.losses = - data.Return
        self.losses.rename("Losses", inplace = True)
        self.alpha = alpha

class StaticRiskMeasures(RiskMeasures):
    
    """
    A subclass of RiskMeasures to represent distribution based Risk measures.

    ------------------------------------------------------------------------

    Attributes
    ----------
    data : DataFrame
        Dataframe of the price history of an asset.
    losses : Series
        Series of the losses of the asset.
    alpha : int/float
        alpha for confidence calculations.
    start_date : Str/Datetime
        Specify start of estimation window.
    end_date : Str/Datetime
        Specify end of estimation window.

    Methods
    -------
    compute_measures_normal
        Computes VaR & CVaR by assuming a normal loss distribution.
    compute_measures_empirical
        Computes VaR & CVaR by fitting an empirical loss distribution.
    compute_measures_fitted_t
        Computes VaR & CVaR by fitting a t-distribution to the losses.
    compute_all
        calls all computation functions.
    """
    
    def __init__(self, data, alpha, start_date, end_date):
        super(StaticRiskMeasures, self).__init__(data, alpha)
                
        self.start_date = start_date
        self.end_date = end_date
        
        self.losses_subset = self.losses.loc[start_date:end_date]
        
        self.measures = pd.DataFrame(index = ["VaR", "CVaR"])
    
    def compute_measures_normal(self):
        
        mean = self.losses_subset.mean()
        std = self.losses_subset.std()
        
        VaR = norm.ppf(1 - self.alpha, loc = mean, scale = std)
        
        tail_loss = norm.expect(lambda x: x, loc = mean, scale = std, lb = VaR)
        CVaR = (1 / self.alpha) * tail_loss
        
        self.measures["Normal"] = [VaR, CVaR]
        
        #return [VaR, CVaR]
        
    def compute_measures_empirical(self):
        
        VaR = np.quantile(self.losses_subset, 1 - self.alpha)
        
        #TODO: Investigate why numerical integration produces implausible values
        # =============================================================================
        # kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(np.reshape(losses_subset.to_frame(), (-1,1)))
        # pdf = lambda x : np.exp(kde.score_samples([[x]]))[0]
        # tail_loss = quad(lambda x: x * pdf(x), a=VaR, b=np.inf)[0]
        # CVaR = (1 / self.alpha) * tail_loss
        # =============================================================================
        
        # try with losses_subset.to_numpy instead of reshape etc.
        
        # Rather naive CVaR right now.
        CVaR = self.losses_subset[self.losses_subset > VaR].mean()
        
        self.measures["Empirical"] = [VaR, CVaR]
        
        #return [VaR, CVaR]
        
        
    def compute_measures_fitted_t(self):
        
        params = t.fit(self.losses_subset)
        
        VaR = t.ppf(1- self.alpha, *params)
        
        tail_loss = t.expect(lambda x: x, args = (params[0],), loc = params[1], scale = params[2], lb = VaR)
        CVaR = (1 / self.alpha) * tail_loss
        
        self.measures["Fitted_t"] = [VaR, CVaR]
        
        #return [VaR, CVaR]
    
    def compute_all(self):
        
        self.compute_measures_normal()
        self.compute_measures_empirical()
        self.compute_measures_fitted_t()
                
    
    def compare_measures_visually(self):
        pass
        
        
class InvalidSamplingException(Exception):
    """
    Raised when the chosen params (backtesting_start & rolling_window) result
    in invalid indexing i.e. the full dataset is not sufficient to perform the test.
    Choose a later backtesting_start or a smaller rolling window.
    """
    pass


class Backtesting:
    
    """
    A class used to represent general backtesting.

    ---------------------------------------------

    Attributes
    ----------
    data : DataFrame
        Dataframe of the price history of an asset
    backtesting_start : Str/Datetime
        Start of backtesting window.
    backtesting_end : Str/Datetime
        End of backtesting window.
    investment : Int/float
        Investment at start of backtesting window.
    alpha : int/float
        alpha for confidence calculations.

    """
    
    def __init__(self, data, backtesting_start, backtesting_end, investment, alpha):       
        
        backtesting_data = data.loc[backtesting_start:backtesting_end]
        backtesting_data.loc[:,"Losses_perc"] = - backtesting_data["Return"]
        backtesting_data.loc[:,"Value"] = (backtesting_data["Return"] + 1).cumprod() * investment
        backtesting_data.loc[:, "Losses_val"] = backtesting_data["Value"] - backtesting_data["Value"].shift(1)
        
        self.backtesting_data = backtesting_data
        self.alpha = alpha
        self.rolled_VaRs = {}


class BacktestingStatic(Backtesting):
    
    """
    A class used to represent backtesting of distribution based VaRs.
    
    ----------------------------------------------------------------

    Attributes
    ----------
    data : DataFrame
        Dataframe of the price history of an asset
    backtesting_start : Str/Datetime
        Start of backtesting window.
    backtesting_end : Str/Datetime
        End of backtesting window.
    investment : Int/float
        Investment at start of backtesting window.
    alpha : int/float
        alpha for confidence calculations.
    rolling_window: int
        Number of days/obs used as estimation window.
        
    Methods
    -------
    normal_dist
        Computes rolled VaR by assuming a normal loss distribution.
    empirical_dist
        Computes rolled VaR by fitting an empirical loss distribution.
    fitted_t_dist
        Computes rolled VaR by fitting a t-distribution to the losses.
    compute_all
        calls all computation functions.
    compare_vars
        compares value at risk with actual losses (percentage and value based)
    """

    
    def __init__(self, data, backtesting_start, backtesting_end, investment, alpha, rolling_window):
        super(BacktestingStatic, self).__init__(data, backtesting_start, backtesting_end, investment, alpha)
        
        exact_date_start = data[backtesting_start:].head(1).index.item()
        start_iloc = data.index.get_loc(exact_date_start)

        exact_date_end = data[backtesting_end:].tail(1).index.item()
        end_iloc = data.index.get_loc(exact_date_end)
        
        start_sample = start_iloc - rolling_window + 1
        
        if start_sample < 0:
            raise InvalidSamplingException
        
        end_sample = end_iloc + 1
        
        self.sample_losses = - data["Return"].iloc[start_sample:end_sample]
        self.rolling_window = rolling_window
        
    def normal_dist(self):
        
        sample = self.sample_losses
        window = self.rolling_window
           
        mu = sample.rolling(window).mean().dropna()
        sigma = sample.rolling(window).std().dropna()
        
        VaR = pd.Series(norm.ppf(1 - self.alpha, loc = mu, scale = sigma), index = self.backtesting_data.index)
        self.rolled_VaRs["Norm"] = VaR
        
        
    def empirical_dist(self):
        
        VaR = self.sample_losses.rolling(self.rolling_window).quantile(1-self.alpha).dropna()
        self.rolled_VaRs["Empirical"] = VaR
    
    def fitted_t_dist(self):
        
        sample = self.sample_losses
        window = self.rolling_window
        
        params = (sample.rolling(window).apply(lambda x:t.fit(x)[0]).dropna(),
                  sample.rolling(window).apply(lambda x:t.fit(x)[1]).dropna(),
                  sample.rolling(window).apply(lambda x:t.fit(x)[2]).dropna())
                  

        VaR = pd.Series(t.ppf(1-self.alpha, *params), index = self.backtesting_data.index)
        self.rolled_VaRs["Fitted_t"] = VaR
    
    def compute_all(self):
        
        self.normal_dist()
        self.empirical_dist()
        self.fitted_t_dist()
        
    def compare_vars(self):
        
        if not bool(self.rolled_VaRs):
            self.compute_all()
        
        VaRs_df = pd.concat(self.rolled_VaRs, axis = 1)
        self.compare_df_perc = pd.concat([self.backtesting_data["Losses_perc"], VaRs_df], axis = 1)
        
        VaRs_val_df = VaRs_df.mul(self.backtesting_data["Value"], axis = 0)
        self.compare_df_val = pd.concat([self.backtesting_data["Losses_val"], VaRs_val_df], axis = 1)
              
            

class BacktestingEval:
    
    """
    A class used to represent evaluation of all backtesting processes.

    ----------------------------------------------------------------

    Attributes
    ----------
    compare_df_val : DataFrame
        Dataframe of the price history of an asset
    alpha : int/float
        alpha for confidence calculations.
    
    Methods
    -------
    plot_vars
        Plots the VaR forecasts vs the actual losses.
    compute_violations
        Computes the number of violations of the VaR forecasts and compares
        to the expected number of violations based on the chosen alpha.
    loss_function
        Computes the Q-measure which is a goodness of fit measure for the VaR forecasts.
    """
    
    def __init__(self, compare_df_val, alpha):
        
        self.compare_df_val = compare_df_val
        self.pnl = compare_df_val.Losses_val
        self.pnl[0] = 0
        self.VaR_measures = compare_df_val.drop("Losses_val", axis = 1)
        self.alpha = alpha
        
    def plot_vars(self):
        
        index = self.compare_df_val.index.values
   
        fig, ax = plt.subplots(figsize=(10, 6))

        fig.suptitle('Vergleich der VaR-Metriken für alpha = {a}%'.format(a = int(self.alpha * 100)), fontsize = 16, y=0.95)

        for col in self.VaR_measures.columns:
            sns.lineplot(x = index, y = self.VaR_measures[col], label = "VaR_" + col,  ax = ax)
            
        sns.scatterplot(x = index, y = self.pnl, label = "PnL", ax = ax, color = "red")

        ax.set_ylabel("Verluste mit anfänglichem Investment von 1 Mio. EUR")

        ax.legend(loc = "lower right")
        ax.grid()
        ax.set_xlabel("Zeitraum der Rückrechnung")
        
        ax.yaxis.set_major_formatter('{x:1.2f}€')

        # =============================================================================
        # fmt = mdates.DateFormatter('%Y')
        # ax.xaxis.set_major_formatter(fmt)
        # =============================================================================

        ax.spines["top"].set_alpha(0)
        ax.spines["bottom"].set_alpha(.3)
        ax.spines["right"].set_alpha(0)
        ax.spines["left"].set_alpha(.3)
        
        return fig
        
        
    def compute_violations(self):
        
        self._violation_result = {}
        
        for col in self.VaR_measures.columns:
            
            self._violation_result[col] = np.sum(self.VaR_measures[col] < self.pnl)
        
        self._violation_result["Expected"] = self.alpha * self.VaR_measures.shape[0]
        self.violation_result = pd.DataFrame.from_dict(self._violation_result, orient = "index", columns = ["Violations"])
                        
    def loss_function(self):
        """
        asymmetric loss function after Gonzales, Lee & Mishra (2004)
        """
        self.detailed_tables = {}
        self._q_measure = {}
        
        for col in self.VaR_measures.columns:
                       
            violation_indicator = (self.VaR_measures[col] < self.pnl).astype(int)
            deviation = self.VaR_measures[col] - self.pnl
            weight = self.alpha - violation_indicator
            weighted_deviation = deviation * weight
            
            self._q_measure[col] = np.round(np.mean(weighted_deviation), 3)
            self.q_measure = pd.DataFrame.from_dict(self._q_measure, orient = "index", columns = ["Q-Measure"])

            table = pd.concat([self.VaR_measures[col], self.pnl, deviation, violation_indicator, weight, weighted_deviation], axis = 1)
            table.columns = ["VaR", "PnL", "Deviation", "violation_indicator", "weight", "Weighted_deviation"]
            
            self.detailed_tables[col] = table
            
@timing_decorator           
def garch_modelling(models, backtesting_start, estimation_start):
    """
    

    Parameters
    ----------
    models : list
        list of garch_models and labels.
    backtesting_start : Str/Datetime
        Start of backtesting window.
    backtesting_end : Str/Datetime
        End of backtesting window.

    Returns
    -------
    model_dict : dict
        dictionary with results of the model estimation.

    """
    
    model_eval = {}
    predictions = {}
    params = {}
    err_term_df = {}

    for name, model in models:
        
        model_result = model.fit(first_obs = estimation_start)
        err_term_df[name] = model_result.params.nu
        model_forecast = model_result.forecast(start = backtesting_start, horizon = 1, #forecast one day ahead
                                               reindex=False) #silence legacy behavior warning
        model_var = model_forecast.variance.dropna() / np.power(model_result.scale, 2)
        model_mean = model_forecast.mean.dropna()
        
        model_params = pd.concat([model_mean, model_var], axis = 1)
        model_params.columns = ["mean", "variance"]
        
        params[name] = model_params
        
        predictions[name] = model_var.squeeze()
        
        vola_proxy = model._y_original[backtesting_start:] ** 2
        
        mse = mean_squared_error(vola_proxy, model_var)
        aic = model_result.aic
        
        model_eval[name] = {"MSE":mse,
                            "AIC":np.round(aic,2)}        
    
    vola_comparison = pd.concat(predictions, axis = 1)
    vola_comparison["vola_proxy"] = vola_proxy
    
    model_dict = {"eval":model_eval,
                  "vola":vola_comparison,
                  "params":params,
                  "err_term_degrees": err_term_df}
    
    return model_dict


class BacktestingGarch(Backtesting):
    
    """
    A class used to represent backtesting of garch based VaRs.
    
    ----------------------------------------------------------------

    Attributes
    ----------
    data : DataFrame
        Dataframe of the price history of an asset
    backtesting_start : Str/Datetime
        Start of backtesting window.
    backtesting_end : Str/Datetime
        End of backtesting window.
    investment : Int/float
        Investment at start of backtesting window.
    models : list
        list of garch_models and labels.
    modelling_results : dict
        dictionary with results of the model estimation.
        
    Methods
    -------
    garch_var
        Computes rolled VaR with the respective garch models.
    compare_vars
        compares value at risk with actual losses (percentage and value based).
    """
    
    def __init__(self, data, backtesting_start, backtesting_end, investment, alpha, models, modelling_results):
        super(BacktestingGarch, self).__init__(data, backtesting_start, backtesting_end, investment, alpha)
    
        self.models = models
        self.modelling_results = modelling_results
    
    def garch_var(self):
        
        for name, model in self.models:
            
            mean = self.modelling_results["params"][name]["mean"]
            var = self.modelling_results["params"][name]["variance"]
            
            nu = self.modelling_results["err_term_degrees"][name]
            q_parametric = - model.distribution.ppf(self.alpha, nu)
            
            VaR_parametric = mean + np.sqrt(var).values * q_parametric            
            VaR_parametric = pd.Series(VaR_parametric, index = mean.index)
            
            self.rolled_VaRs[name] = VaR_parametric
            
    def compare_vars(self):
        
        if not bool(self.rolled_VaRs):
            self.garch_var()
        
        VaRs_df = pd.concat(self.rolled_VaRs, axis = 1)
        self.compare_df_perc = pd.concat([self.backtesting_data["Losses_perc"], VaRs_df], axis = 1)
        
        VaRs_val_df = VaRs_df.mul(self.backtesting_data["Value"], axis = 0)
        self.compare_df_val = pd.concat([self.backtesting_data["Losses_val"], VaRs_val_df], axis = 1)