""" Keep all the basic functions for the exploratory data analysis."""
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
# Core
import pandas as pd
import plotly.graph_objects as go
# Visualization
import seaborn as sns
from IPython.display import HTML, display
# Statistics
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

cmap = 'RdBu'


def centered(text: str) -> None:
    """Centralize text prints.

    :param text: Text to be centralized on the screen
    """
    display(HTML(f"<div style='text-align:center'>{text}</div>"))


def data_check(df: pd.DataFrame) -> None:
    """Display the dataset report with data type of each feature,
    number of "nan" values, duplicated rows,
    data description with statistical information and etc.

    :param df: DataFrame of the dataset
    """
    print('='*149)
    centered('Data Information')
    display(df.info())

    print('='*149)
    centered('Check for nan values and duplicated rows')
    print(f'Number of nan values: {df.isna().sum().sum()}')
    print(f'Number of duplicated rows: {df.duplicated().sum()}')

    print('='*149)
    centered('Data Description')
    display(df.describe().T)

    print('='*149)
    centered('Data Head')
    display(df.head())

    print('='*149)
    centered('Data Shape')
    print(f'Data Shape: {df.shape}')

    print('='*149)
    centered('Analysis period')
    print(f'From: {df.index.min()}\nTo: {df.index.max()}')
    print(f'Total days: {df.index.max() - df.index.min()}')


def count_values(df: pd.DataFrame, values: list) -> None:
    """Display the percentage of a value or a list of values on each feature.

    :param df: DataFrame of the dataset
    :param args: Value or list of values to be checked on each feature
    """
    for col in list(df):
        total = df[col].isin(values).sum()
        print(f'{col}: {total:-<25}> {(total/df[col].count()*100):.2f}%')


def hist_box(df: pd.DataFrame, feature: str, avoid_zeros: bool = False) -> None:
    """Draw the histogram and boxplot distributions of a feature

    :param df: DataFrame of the dataset
    :param feature: Feature to be analyzed
    :param avoid_zeros: Use of zeros in the distribution, defaults to False
    """
    if avoid_zeros:
        df2 = df.mask(df == 0)
    else:
        df2 = df.copy()

    fig, (ax_box, ax_hist) = plt.subplots(
        nrows=2,
        sharex=True,
        gridspec_kw={'height_ratios': (0.25, 0.75)},
        figsize=(12, 7)
    )

    # boxplot with a star indicating the mean value of the feature
    sns.boxplot(data=df2, x=feature, ax=ax_box,
                showmeans=True)  # , color = "pink")
    sns.histplot(data=df2, x=feature, ax=ax_hist)

    # Add mean and median to histogram
    ax_hist.axvline(df2[feature].mean(), color='green')  # mean
    ax_hist.axvline(df2[feature].median(), color='orange')  # median

    fig.suptitle('Distribution of ' + feature, fontsize=16)


def heatmap_diag(df: pd.DataFrame, annot: bool = False, fontsize: int = 16) -> None:
    """Draw the diagonal heatmap of the dataset.

    :param df: DataFrame of the dataset, defaults to pd.DataFrame
    :param annot: Display the numbers on each cell of the heatmap, defaults to False
    """
    plt.figure(figsize=(16, 10))
    # Compute the correlation matrix
    corr = df.corr()
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(data=corr, mask=mask, vmin=-1, vmax=1, annot=annot, fmt='.2f',
                cmap='RdBu', annot_kws={'size': fontsize})  # cmap = "Spectral");
    plt.show()


def pairplot_reg(df: pd.DataFrame, features: list, only_day: bool = False) -> None:
    """Draw diagonal pairwise relationships from the dataset with the scatterplot,
    linear regression and Pearson's coefficient of each combination of two features
    from a list of features.

    :param df: DataFrame of the dataset
    :param features: List of features to be analyzed
    :param only_day: Daytime use only, defaults to False
    """
    if only_day:
        df2 = df[df.GHI != 0]  # Select the day time based on GHI = 0
    else:
        df2 = df.copy()

    # sns.reset_defaults()
    def corrfunc(x, y, **kws):
        r = stats.pearsonr(x, y)[0]
        ax = plt.gca()
        ax.annotate(
            f'r = {r:.2f}',
            xy=(.1, .5), xycoords=ax.transAxes,
            weight='bold', fontsize=20, color='k')
        ax.grid(False)

    g = sns.pairplot(
        data=df2, vars=features,
        kind='reg', diag_kind='kde',
        plot_kws={'line_kws': {'color': 'red'},
                  'scatter_kws': {'s': 1, 'alpha': 0.1}},
        corner=True)
    g = g.map_lower(corrfunc)

    g.tight_layout()
    plt.show()


def boxplots(df: pd.DataFrame, x: str = 'Month', y: list = ['GHI', 'DNI', 'DHI']) -> None:
    """Draw boxplot with nested grouping by two variables.

    :param df: DataFrame of the dataset
    :param x: Variable of the axis x, defaults to 'Month'
    :param y: Variables of the axis y, defaults to ['GHI', 'DNI', 'DHI']
    """
    fig, axs = plt.subplots(nrows=len(y), ncols=1, figsize=(
        16, 16), linewidth=0.5, alpha=0.4)

    for i, ax in enumerate(axs):
        ax.set_title(y[i])
        ax.set_ylabel(y[i])
        sns.boxplot(data=df, x=x, y=y[i], ax=ax)
    plt.show()


def plot_timeseries(ts, title='og', line_color='lightslategrey'):
    """Plot plotly time series of any given timeseries ts."""
    fig = go.Figure()

    _ = fig.add_trace(go.Scatter(
        x=ts.index,
        y=ts.values,
        name='observed',
        opacity=1, line_color=line_color))

    _ = fig.update_layout(title_text=title,
                          xaxis_rangeslider_visible=True)
    fig.show()


def run_adfuller(ts):
    """_summary_.

    :param ts: _description_
    """
    result = adfuller(ts)
    # Print test statistic
    print('t-stat', result[0])
    # Print p-value
    print('p-value', result[1])
    # Print #lags used
    print('#lags used', result[2])
    # Print critical values
    print('critical values', result[4])


def acf_pacf_plots(ts, lags):
    """_summary_.

    :param ts: _description_
    :param lags: _description_
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

    # Plot the ACF of ts
    _ = plot_acf(ts, lags=lags, zero=False, ax=ax1, alpha=0.05)

    # Plot the PACF of ts
    _ = plot_pacf(ts, lags=lags, zero=False, ax=ax2, alpha=0.05)

# def scatter(df:pd.DataFrame, x:str='Month', y:list=['GHI', 'DNI', 'DHI'])->None:
#     """Draw scatterplot of two variables

#     :param df: DataFrame of dataset
#     :param x: Variable of the axis x, defaults to 'Month'
#     :param y: Variable of the axis y, defaults to ['GHI', 'DNI', 'DHI']
#     """
#     df[y].plot(subplots=True, figsize=(16, 8))
#     [ax.legend(loc = 1) for ax in plt.gcf().axes]
#     plt.suptitle('Irradiances in $KW-hr/m^2/day$ of GHI, DNI and DHI')
#     plt.tight_layout()
#     plt.subplots_adjust(top = 0.95)
#     plt.show();


def add_fourier_terms(df, year_k, day_k):
    """
    df: dataframe to add the fourier terms to.
    year_k: the number of Fourier terms the year period should have. Thus the model will be fit on 2*year_k terms (1 term for
    sine and 1 for cosine)
    day_k:same as year_k but for daily periods
    """
    for k in range(1, year_k+1):
        # year has a period of 365.25 including the leap year
        df['year_sin'+str(k)] = np.sin(2 * k * np.pi *
                                       df.index.dayofyear/365.25)
        df['year_cos'+str(k)] = np.cos(2 * k * np.pi *
                                       df.index.dayofyear/365.25)

    for k in range(1, day_k+1):

        # day has period of 24
        df['hour_sin'+str(k)] = np.sin(2 * k * np.pi * df.index.hour/24)
        df['hour_cos'+str(k)] = np.cos(2 * k * np.pi * df.index.hour/24)


def hourly_irrad(df: pd.DataFrame, y: list = ['GHI', 'DNI', 'DHI']) -> None:
    """Draw average hourly observed feature over the entire period.

    :param df: DataFrame of the dataset
    :param y: Variable of the axis y, defaults to ['GHI', 'DNI', 'DHI']
    """
    plt.figure(figsize=(16, 6))
    df.groupby('Hour')[y].mean().plot()
    plt.ylabel('Irradiance in $KW-hr/m^2/day$')
    plt.ylim([0, df.groupby('Hour')[y].mean().max().max() + 200])
    plt.xticks(df['Hour'].unique())
    plt.title(
        'Hourly Solar Irradiance consumption in $KW-hr/m^2/day$ averaged over 21 years (2001-2021)')
    plt.show()


def daily_irrad(df: pd.DataFrame, y: list = ['GHI', 'DNI', 'DHI']) -> None:
    """Draw average daily observed feature over the entire period.

    :param df: DataFrame of the dataset
    :param y: Variable of the axis y, defaults to ['GHI', 'DNI', 'DHI']
    """
    plt.figure(figsize=(16, 6))
    df.groupby('Day')[y].mean().plot()
    plt.ylabel('Irradiance in $KW-hr/m^2/day$')
    plt.ylim([0, df.groupby('Day')[y].mean().max().max() + 200])
    plt.xticks(df['Day'].unique())
    plt.title(
        'Daily Solar Irradiance consumption in $KW-hr/m^2/day$ averaged over 12 months (Jan-Dez)')
    plt.show()


def monthly_irrad(df: pd.DataFrame, y: list = ['GHI', 'DNI', 'DHI']) -> None:
    """Draw average monthly observed feature over the entire period.

    :param df: DataFrame of the dataset
    :param y: Variable of the axis y, defaults to ['GHI', 'DNI', 'DHI']
    """
    plt.figure(figsize=(16, 6))
    df.groupby('Month')[y].mean().plot()
    plt.ylabel('Irradiance in $KW-hr/m^2/day$')
    plt.ylim([0, df.groupby('Month')[y].mean().max().max() + 200])
    plt.xticks(df['Month'].unique())
    plt.title(
        'Monthly Solar Irradiance consumption in $KW-hr/m^2/day$ averaged over 21 years (2001-2021)')
    plt.show()


def max_monthly(df: pd.DataFrame, feature: str) -> None:
    """Draw a lineplot with the max value of a feature for each month.

    :param df: DataFrame of the dataset
    :param feature: Feature to be analyzed
    """
    monthly_en = df.resample('M', label='left')[feature].max()
    plt.figure(figsize=(16, 6))
    # plotting the max monthly energy consumption
    plt.plot(monthly_en)
    # ensuring the limits on x axis to be between the dataframe's datetime limits
    plt.xlim(monthly_en.index.min(), monthly_en.index.max())
    # Using matplotlib MonthLocator to be used in the xticks to mark individual months
    locator = mdates.MonthLocator(bymonthday=1, interval=6)  # every 6 months
    # fmt = mdates.DateFormatter('%m-%y')  # xticks to be displayed as 01-14 (i.e. Jan'14) and so on
    # xticks to be displayed as 01-14 (i.e. Jan'14) and so on
    fmt = mdates.DateFormatter('%m-%y')
    X = plt.gca().xaxis
    # Setting the locator
    X.set_major_locator(locator)
    # Specify formatter
    X.set_major_formatter(fmt)
    plt.xticks(rotation=60, fontsize=8)
    plt.ylabel('Max Solar Irradiance in $KW-hr/m^2/day$')
    plt.xlabel('Date')
    plt.show()
    # plt.figure(figsize=(15,6))
    # df['GHI'].rolling(24*30).max().plot()


def detect_trend(df: pd.DataFrame, text: str, feature: str) -> None:
    """Draw a lineplot with the feature average over the day and linear regression to check the trend.

    :param df: DataFrame of the dataset
    :param text: Text applied for title
    :param feature: Feature to be analyzed
    """
    plt.figure(figsize=(16, 6))
    coefficients, residuals, _, _, _ = np.polyfit(range(len(df)),
                                                  df,
                                                  1,
                                                  full=True)

    mse = residuals[0]/(len(df))
    nrmse = np.sqrt(mse)/(df.max() - df.min())

    print('Slope ' + str(coefficients[0]))
    print('NRMSE: ' + str(nrmse))

    plt.xticks(rotation=90)
    plt.plot(df,
             marker='.',
             linestyle='-',
             linewidth=0.5,
             color='blue',
             label='Original')

    plt.plot([coefficients[0]*x + coefficients[1] for x in range(len(df))],
             marker='o',
             markersize=8,
             linestyle='-',
             linewidth=0.5,
             color='orange',
             label='Regression line')

    plt.title(f'({text}) - {feature}')
    plt.xlabel('Month')
    plt.ylabel('Solar Irradiance (kW-hr/m^2/day)')
    plt.legend()
    plt.show()


def trend(df: pd.DataFrame, feature: str, year: str) -> None:
    """Draw a lineplot with the feature average over the day and linear regression to check the trend.

    :param df: DataFrame of the dataset
    :param feature: Feature to be analyzed
    :param year: Specific year or 'all' available years to be analyzed
    """
    if year == 'all':
        detect_trend(df=df[feature].resample('M').sum().values,
                     text='2001 - 2021', feature=feature)
    else:
        detect_trend(df=df[feature].resample('M').sum(
        ).loc[year].values, text=year, feature=feature)


def split_TS(data: pd.DataFrame, n: int) -> None:
    """Split the data into n equal parts and display the mean and variance of each part.

    :param data: Data to be analyzed
    :param n: Number of parts to split the data
    """
    X = np.array_split(data.values, n)
    display(pd.DataFrame({
        'mean': list(map(lambda x: round(np.mean(x), 2), X)),
        'variance': list(map(lambda x: round(np.var(x), 2), X))
    })
    )


def ADF_test(df: pd.DataFrame, feature: str) -> None:
    """Augmented Dickey-Fuller test.

    :param df: DataFrame of the dataset
    :param feature: Feature to be analyzed
    """
    for year in np.unique(df.Year):
        # detect_trend(data_grouped.loc[str(year)]['GHI'].values)

        X = df.loc[str(year)][feature].values

        result = adfuller(X)

        print('='*30)
        print(f'For the year : {year}')
        print(f'ADF Statistic: {result[0]:.6f}')
        print(f'p-value: {result[1]:.6f}')
        print('Critical Values:')
        for key, value in result[4].items():
            print('\t%5s: %8.3f' % (key, value))

    X = df[feature].values

    result = adfuller(X)

    print('='*30)
    print('For the CONSOLIDATED 21 years (2001 - 2021)')
    print(f'ADF Statistic: {result[0]:.6f}')
    print(f'p-value: {result[1]:.6f}')
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%5s: %8.3f' % (key, value))


def test_stationarity(data: pd.DataFrame, reduced_data: pd.DataFrame = None, smoothing_type: str = 'R') -> None:
    """Test the stationarity of the time series data.

    :param data: Data to be analyzed
    :param reduced_data: Reduced data to be analyzed, defaults to None
    :param smoothing_type: Smoothing type of the reduced data that can be "MA" for mean avarage,\
         "R" for rolling MA or "E" for exponential weighted MA, defaults to 'R'
    """
    plt.figure(figsize=(16, 6))
    orig = plt.plot(data,
                    marker='.',
                    linestyle='-',
                    linewidth=0.5,
                    color='blue',
                    label='Original')

    mean = plt.plot(reduced_data,
                    marker='.',
                    linestyle='-',
                    linewidth=0.5,
                    color='red',
                    label='Reduced')

    if smoothing_type == 'R':
        title = 'Smoothing by Rolling MA'
    elif smoothing_type == 'E':
        title = 'Smoothing by Exponential weighted MA'
    else:
        title = 'Default MA'

    plt.legend(loc='best')
    plt.title(title)
    plt.show(block=False)

    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(reduced_data, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=[
                         'Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])

    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value

    print(dfoutput)
