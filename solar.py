import numpy as np
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import datetime as dt

class Solar:
    """
    This class enables data loading, plotting and statistical analysis of a given feature,
     upon initialization load a sample of data to check if feature exists. 
        
    """

    def __init__(self, symbol="GHI", unit='KW-hr/m^2/day'):        
        self.symbol = symbol
        self.unit = unit

    @st.cache(show_spinner=False) #Using st.cache allows st to load the data once and cache it. 
    def load_full_data(self):
        full_data = pd.read_csv('data/NASA_Dataset_Cleaned.csv', index_col = 'DATETIME', parse_dates=['DATETIME', 'DATETIME'])        
        return full_data


    def create_features(self, df):
        data = self.load_full_data()
        target_map = data[self.symbol].to_dict()
        df['lag0'] = (df.index - pd.Timedelta('5 hours')).map(target_map)
        df['lag1'] = (df.index - pd.Timedelta('7 days')).map(target_map)
        df['lag2'] = (df.index - pd.Timedelta('8 days')).map(target_map)
        df['lag3'] = (df.index - pd.Timedelta('9 days')).map(target_map)
        df['lag4'] = (df.index - pd.Timedelta('10 days')).map(target_map)
        df['lag5'] = (df.index - pd.Timedelta('11 days')).map(target_map)
        df['lag6'] = (df.index - pd.Timedelta('12 days')).map(target_map)
        df['lag7'] = (df.index - pd.Timedelta('13 days')).map(target_map)
        df['lag8'] = (df.index - pd.Timedelta('364 days')).map(target_map)
        df['lag9'] = (df.index - pd.Timedelta('728 days')).map(target_map)
        df['lag10'] = (df.index - pd.Timedelta('1092 days')).map(target_map)  

        df['date'] = df.index
        df['hour'] = df['date'].dt.hour
        df['dayofweek'] = df['date'].dt.dayofweek
        df['quarter'] = df['date'].dt.quarter
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['dayofyear'] = df['date'].dt.dayofyear
        df['dayofmonth'] = df['date'].dt.day
        df['weekofyear'] = df['date'].dt.weekofyear
        df.set_index('date', inplace=True)
        return df

    def load_data(self, start, end, scaler=False):
        """
        takes a start and end dates, download data do some processing and returns dataframe
        """

        full_data = self.load_full_data()
        self.start = start
        self.end = end

        index = pd.date_range(start=full_data.index[-1] + dt.timedelta(hours=1), end=self.end, freq='H')
        columns = full_data.columns
        df = pd.DataFrame(index=index, columns=columns)
            
        full_data = pd.concat([full_data, df])
        full_data = self.create_features(full_data)
        if scaler:
            # not adding year variable here because the model will use the most recent lag energy consumption values
            cols_to_transform = ['T2MDEW', 'DIFFUSE_ILLUMINANCE', 'DIRECT_ILLUMINANCE',
            'GLOBAL_ILLUMINANCE', 'RH2M', 'QV2M', 'PS', 'T2M', 'SZA', 'WS2M'] # other columns are binary values
            scaler = StandardScaler()
            full_data[cols_to_transform] = scaler.fit_transform(full_data[cols_to_transform])

        data = full_data[start:end]
        # data['date'] = data.index

        # Check if there is data
        try:
            assert len(data) > 0
        except AssertionError:
            print("Cannot fetch data, check spelling or time window")
        data.reset_index(inplace=True)        
        self.data = data
        # return self.data


    # @st.cache(show_spinner=False) #Using st.cache allows st to load the data once and cache it. 
    def model(self):
        xgbtunedreg = xgb.Booster()
        xgbtunedreg.load_model(f"models/xgb_model_{self.symbol}.json")
        best_ntree = xgbtunedreg.best_ntree_limit

        # Scale
        data_copy = self.data.copy()
        data_copy.set_index('date', inplace=True)
        vars_to_drop = ['DNI', 'DHI', 'GHI']
        data_copy.drop(labels=vars_to_drop, axis=1, inplace=True)

        # # Prepare data as DMatrix objects
        test = xgb.DMatrix(data_copy)
        self.preds = xgbtunedreg.predict(test, ntree_limit=best_ntree)
        self.data['preds'] = self.preds

    def plot_raw_data(self, type:str='chart', forecast=False):
        """
        Plot plotly time series chart of a selected feature on a given plotly.graph_objects.Figure object
        """
        
        fig = go.Figure()

        if type=='chart':

            fig = fig.add_trace(
                go.Scatter(
                    x=self.data['date'], 
                    y=self.data[self.symbol], 
                    name=f"Observed {self.symbol}",
                    # line_color='deepskyblue', 
                    opacity=0.6))

            if forecast:
                self.model()

                fig = fig.add_trace(
                    go.Scatter(
                        x=self.data['date'], 
                        y=self.data['preds'], 
                        name=f"Predicted {self.symbol}",
                        # line_color='orange', 
                        opacity=0.6))


            fig = fig.update_layout( 
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(step="all"),
                            dict(count=1,
                                label="1y",
                                step="year",
                                stepmode="backward"),
                            dict(count=6,
                                label="6m",
                                step="month",
                                stepmode="backward"),
                            dict(count=1,
                                label="1m",
                                step="month",
                                stepmode="backward"),
                            dict(count=7,
                                label="7d",
                                step="day",
                                stepmode="backward"),
                        ])
                    ),
                    type="date",
                    rangeslider=dict(visible=True),
                ),
                xaxis_title="Date",
                yaxis_title=f"{self.symbol} ({self.unit})"
            )
        elif type=='hist':
            fig = fig.add_trace(
                go.Histogram(
                    x=self.data[self.symbol],
                    name=self.symbol,
                    opacity = 0.5))
            fig = fig.update_layout(
                xaxis_title=f"{self.unit}",
                yaxis_title="count"
            )
        else:
            print('Invalid plot type!\nChoose one of the folowing plot types: chart or hist')
            return None


        fig = fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0, pad=0),
            legend=dict(
                x=0.005,
                y=0.99,
                traceorder="normal",
                font=dict(size=12),
            ),
            showlegend=True,
            autosize=False,
            template="plotly",
            )

        return fig
