''' This module keeps all the basic functions for the machine learning modeling 
such as calculating error metrics, plotting predicted vs original time series, etc. 
So that we can use the same functions seamlessly across different models and this will also 
allow us to compare different models using the same metrics.
'''

# Core
import numpy as np
import pandas as pd

# Visualization
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display, HTML
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Preprocessing
from sklearn.preprocessing import StandardScaler

# Machine Learning
from sklearn import linear_model
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

# Metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Time series split for cross validation 
from sklearn.model_selection import TimeSeriesSplit

# Optimization
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Statistics
from statsmodels.tsa.stattools import adfuller


def centered(text:str)->None:
    '''Centralize text prints

    :param text: Text to be centralized on the screen
    '''
    display(HTML(f"<div style='text-align:center'>{text}</div>"))


def data_check(df:pd.DataFrame)->None:
    '''Display the dataset report with data type of each feature, number of "nan" values, duplicated rows, 
    data description with statistical information and etc.

    :param df: DataFrame of the dataset
    '''
    print("="*149)
    centered('Data Information')
    display(df.info())

    print("="*149)
    centered('Check for nan values and duplicated rows')
    print(f'Number of nan values: {df.isna().sum().sum()}')
    print(f'Number of duplicated rows: {df.duplicated().sum()}')

    print("="*149)
    centered('Data Description')
    display(df.describe().T)

    print("="*149)
    centered('Data Head')
    display(df.head())

    print("="*149)
    centered('Data Shape')
    print(f'Data Shape: {df.shape}')

    print("="*149)
    centered('Analysis period')
    print(f'From: {df.index.min()}\nTo: {df.index.max()}')
    print(f'Total days: {df.index.max() - df.index.min()}')


def plot_predvstrue_reg(pred, truth, model_name=None):
    '''Draw the observed versus the predicted Solar Irradiance (GHI)

    :param pred: _description_
    :param truth: _description_
    :param model_name: _description_, defaults to None
    '''
    fig, ax = plt.subplots(1,1, figsize=(8,8))
    ax.scatter(truth, pred, edgecolors='white', alpha=0.5) 
    _ = plt.xlabel(f"Observed Solar Irradiance ({truth.name}) in $KW-hr/m^2/day$")
    _ = plt.ylabel(f"Predicted Solar Irradiance ({truth.name}) in $KW-hr/m^2/day$")
    _ = plt.title(f"Observed vs Predicted Solar Irradiance ({truth.name}) using {model_name}")
    # plt.xlim(-100, 1000)
    # plt.ylim(-200, 1300)
    #plotting 45 deg line to see how the prediction differs from the observed values
    x = np.linspace(*ax.get_xlim())
    ax.plot(x, x, color='red')


def coef_plot(X, title:str, model:linear_model):
    # Plotting the coefficients to check the importance of each coefficient 
    _ = plt.figure(figsize = (16, 6))
    _ = plt.plot(range(len(X.columns)), model.coef_)
    _ = plt.xticks(range(len(X.columns)), X.columns.values, rotation = 90)
    _ = plt.margins(0.02)
    _ = plt.axhline(0, linewidth = 0.5, color = 'r')
    _ = plt.title(title)
    _ = plt.ylabel('coeff')
    _ = plt.xlabel('Features')


def residuals_plot(X, y, model:linear_model, hist=False):
    # PLotting the residuals
    residuals = (y - model.predict(X))
    _ = plt.figure(figsize=(5,5))
    _ = plt.scatter(model.predict(X) , residuals, edgecolors='white', alpha = 0.5) 
    _ = plt.xlabel(f"Model predicted solar irradiance ({y.name}) values")
    _ = plt.ylabel("Residuals")
    model_text= str(model)[:-2]
    if model_text=='LinearRegression':
        _ = plt.title(f"Fitted values versus Residuals for {model_text}")
    else:
        _ = plt.title(f"Fitted values versus Residuals for {model_text} Regression")
    if hist:
        _ = plt.figure(figsize=(8,5))
        _ = sns.histplot(residuals);
        _ = plt.xlabel('Residuals');
        _ = plt.title('Residuals');


def train_test(
    data:pd.DataFrame, 
    target:str, 
    test_size:float = 0.15, 
    scale:bool = False, 
    cols_to_transform:list=None, 
    include_test_scale:bool=False, 
    plot=True):
    """
    
        Perform train-test split with respect to time series structure
        
        - df: dataframe with variables X_n to train on and the dependent output y which is the column 'target' in this notebook
        - test_size: size of test set
        - scale: if True, then the columns in the -'cols_to_transform'- list will be scaled using StandardScaler
        - include_test_scale: If True, the StandardScaler fits the data on the training as well as the test set; if False, then
        the StandardScaler fits only on the training set.
        
    """
    df = data.copy()
    # get the index after which test set starts
    test_index = int(len(df)*(1-test_size))
    
    # StandardScaler fit on the entire dataset
    if scale and include_test_scale:
        scaler = StandardScaler()
        df[cols_to_transform] = scaler.fit_transform(df[cols_to_transform])
        
    X_train = df.drop(target, axis = 1).iloc[:test_index]
    y_train = df[target].iloc[:test_index]
    X_test = df.drop(target, axis = 1).iloc[test_index:]
    y_test = df[target].iloc[test_index:]

    if plot:
        _ = plt.figure(figsize=(16,6))
        _ = plt.title(f'Solar Irradiance ({target}) train and test sets', size=20)
        _ = plt.plot(y_train, label='Training set')
        _ = plt.plot(y_test, label='Test set', color='orange')
        _ = plt.legend()
        plt.show();

    
    # StandardScaler fit only on the training set
    if scale and not include_test_scale:
        scaler = StandardScaler()
        X_train[cols_to_transform] = scaler.fit_transform(X_train[cols_to_transform])
        X_test[cols_to_transform] = scaler.transform(X_test[cols_to_transform])
    
    return X_train, X_test, y_train, y_test


def error_metrics(y_pred, y_truth, model_name, test, dict_error:dict):
    """
    Printing error metrics like RMSE (root mean square error), R2 score, 
    MAE (mean absolute error), MAPE (mean absolute % error). 
    
    y_pred: predicted values of y using the model model_name
    y_truth: observed values of y
    model_name: name of the model used for predictions
    test: if validating on test set, True; otherwise False for training set validation
    
    The function will print the RMSE, R2, MAE and MAPE error metrics for the model_name and also store the results along with 
    model_name in the dictionary dict_error so that we can compare all the models at the end.
    """
    if isinstance(y_pred, np.ndarray):
        y_pred = y_pred
    else:
        y_pred = y_pred.to_numpy()
        
    if isinstance(y_truth, np.ndarray):
        y_truth = y_truth
    else:
        y_truth = y_truth.to_numpy()
        
    print('\nError metrics for model {}'.format(model_name))
    
    RMSE = np.sqrt(mean_squared_error(y_truth, y_pred))
    print("RMSE or Root mean squared error: %.2f" % RMSE)
    
    # Explained variance score: 1 is perfect prediction

    R2 = r2_score(y_truth, y_pred)
    print('R\u00b2 score: %.2f' % R2 )

    MAE = mean_absolute_error(y_truth, y_pred)
    print('MAE or Mean Absolute Error: %.2f' % MAE)

    # MAPE = (np.mean(np.abs((y_truth - y_pred) / y_truth)) * 100)
    # print('Mean Absolute Percentage Error: %.2f %%' % MAPE)
    
    # Appending the error values along with the model_name to the dict
    if test:
        train_test = 'test'
    else:
        train_test = 'train'
    
    #df = pd.DataFrame({'model': model_name, 'RMSE':RMSE, 'R2':R2, 'MAE':MAE, 'MAPE':MAPE}, index=[0])
    name_error = ['model', 'train_test', 'RMSE', 'R\u00b2', 'MAE']#, 'MAPE']
    value_error = [model_name, train_test, RMSE, R2, MAE]#, MAPE]
    list_error = list(zip(name_error, value_error))
    
    for error in list_error:
        if error[0] in dict_error:
            # append the new number to the existing array at this slot
            dict_error[error[0]].append(error[1])
        else:
            # create a new array in this slot
            dict_error[error[0]] = [error[1]]
    return dict_error

    
def plot_timeseries(ts, title = 'og', opacity = 1):
    """
    Plot plotly time series of any given timeseries ts
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(x = ts.index, y = ts.values, name = "observed",
                        line_color = 'lightslategrey', opacity = opacity))

    fig.update_layout(title_text = title,
                xaxis_rangeslider_visible = True)
    fig.show()


def plot_ts_pred(og_ts, pred_ts, model_name=None, og_ts_opacity = 0.5, pred_ts_opacity = 0.5):
    """
    Plot plotly time series of the original (og_ts) and predicted (pred_ts) 
    time series values to check how our model performs.
    model_name: name of the model used for predictions
    og_ts_opacity: opacity of the original time series
    pred_ts_opacity: opacity of the predicted time series
    """
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(x = og_ts.index, y = np.array(og_ts.values), name = "Observed",
                        line_color = 'deepskyblue', opacity = og_ts_opacity))

    try:
        fig.add_trace(go.Scatter(x = pred_ts.index, y = pred_ts, name = model_name,
                        line_color = 'lightslategrey', opacity = pred_ts_opacity))
    except: #if predicted values are a numpy array they won't have an index
        fig.add_trace(go.Scatter(x = og_ts.index, y = pred_ts, name = model_name,
                        line_color = 'lightslategrey', opacity = pred_ts_opacity))


    #fig.add_trace(go)
    fig.update_layout(title_text = f'Observed test set vs predicted Solar Irradiance ({og_ts.name}) KW-hr/m2/day using {model_name}',
                xaxis_rangeslider_visible = True)
    fig.show()


# Trying elastic net regression
def trend_model(data, target, cols_to_transform, l1_space, alpha_space, cols_use = 0, scale = True, test_size = 0.15, 
                include_test_scale=False, plot=False):
    """
    Tuning, fitting and predicting with an Elastic net regression model.
    data: time series dataframe including X and y variables
    col_use: columns including the y variable to be used from the data
    cols_to_transform: columns to be scaled using StandardScaler if scale = True
    l1_space: potential values to try for the l1_ratio parameter of the elastic net regression
    include_test_scale: If True then the StandardScaler will be fit on the entire dataset instead of just the training set
    
    A note about l1_ratio: The ElasticNet mixing parameter, with 0 <= l1_ratio <= 1. 
    For l1_ratio = 0 the penalty is an L2 penalty. For l1_ratio = 1 it is an L1 penalty. 
    For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.
    """
    
    # Creating the train test split
    if cols_use != 0:
        df = data[cols_use]
    else:
        df = data
    
    X_train, X_test, y_train, y_test = train_test(df, target=target, test_size=test_size, 
                                              scale=scale, cols_to_transform=cols_to_transform, 
                                              include_test_scale=include_test_scale, plot=plot)

    
    # Create the hyperparameter grid
    #l1_space = np.linspace(0, 1, 50)
    param_grid = {'l1_ratio': l1_space, 'alpha': alpha_space}

    # Instantiate the ElasticNet regressor: elastic_net
    elastic_net = ElasticNet()

    # for time-series cross-validation set 5 folds
    tscv = TimeSeriesSplit(n_splits=5)

    # Setup the GridSearchCV object: gm_cv ...trying 5 fold cross validation 
    gm_cv = GridSearchCV(elastic_net, param_grid, cv = tscv)

    # Fit it to the training data
    gm_cv.fit(X_train, y_train)

    # Predict on the test set and compute metrics
    y_pred = gm_cv.predict(X_test)
    r2 = gm_cv.score(X_test, y_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Tuned ElasticNet l1 ratio: {}".format(gm_cv.best_params_))
    print("Tuned ElasticNet R\u00b2: {}".format(r2))
    print("Tuned ElasticNet RMSE: {}".format(np.sqrt(mse)))
    # fitting the elastic net again using the best model from above

    elastic_net_opt = ElasticNet(l1_ratio = gm_cv.best_params_['l1_ratio']) 
    elastic_net_opt.fit(X_train, y_train)
    
    # Plot the coefficients
    _ = plt.figure(figsize = (15, 7))
    _ = plt.plot(range(len(X_train.columns)), elastic_net_opt.coef_)
    _ = plt.xticks(range(len(X_train.columns)), X_train.columns.values, rotation = 90)
    _ = plt.margins(0.02)
    _ = plt.axhline(0, linewidth = 0.5, color = 'r')
    _ = plt.title('Significance of features as per Elastic regularization')
    _ = plt.ylabel('Elastic net coeff')
    _ = plt.xlabel('Features')
    
    # Plotting y_true vs predicted
    _ = plt.figure(figsize = (5,5))
    plot_predvstrue_reg(
        pred=elastic_net_opt.predict(X_test), 
        truth=y_test, 
        model_name='Elastic net optimal linear regression')
    
    # returns the train and test X and y sets and also the optimal model
    return X_train, X_test, y_train, y_test, elastic_net_opt


def run_adfuller(ts):
    result = adfuller(ts)
    # Print test statistic
    print("t-stat", result[0])
    # Print p-value
    print("p-value", result[1])
    # Print #lags used
    print("#lags used", result[2])
    # Print critical values
    print("critical values", result[4]) 


def plot_ewma(ts, alpha):
    expw_ma = ts.ewm(alpha=alpha).mean()
    
    plot_ts_pred(
        og_ts=ts, 
        pred_ts=expw_ma, 
        model_name=f'Exponentially smoothed data (\u03B1 = {alpha}')


def acf_pacf_plots(ts, lags):
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (16,10))
    
    # Plot the ACF of ts
    _ = plot_acf(ts, lags = lags, zero = False, ax = ax1, alpha = 0.05)

    # Plot the PACF of ts
    _ = plot_pacf(ts, lags = lags, method = "ols", zero = False, ax = ax2, alpha = 0.05)


def add_fourier_terms(df, year_k, day_k):
    """
    df: dataframe to add the fourier terms to 
    year_k: the number of Fourier terms the year period should have. Thus the model will be fit on 2*year_k terms (1 term for 
    sine and 1 for cosine)
    day_k:same as year_k but for daily periods
    """
    
    for k in range(1, year_k+1):
        # year has a period of 365.25 including the leap year
        df['year_sin'+str(k)] = np.sin(2 *k* np.pi * df.index.dayofyear/365.25) 
        df['year_cos'+str(k)] = np.cos(2 *k* np.pi * df.index.dayofyear/365.25)

    for k in range(1, day_k+1):
        
        # day has period of 24
        df['hour_sin'+str(k)] = np.sin(2 *k* np.pi * df.index.hour/24)
        df['hour_cos'+str(k)] = np.cos(2 *k* np.pi * df.index.hour/24) 