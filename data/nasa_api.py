"""
*Version: 2.0 Published: 2021/03/09* Source: [NASA POWER](https://power.larc.nasa.gov/).

POWER API Multi-Point Download
This is an overview of the process to request data from multiple data points from the POWER API.
"""
import multiprocessing
import os
import shutil
import sys
import time
from datetime import datetime
from glob import glob
from math import ceil

import numpy as np
import pandas as pd
import requests
import urllib3
from dateutil.relativedelta import relativedelta

urllib3.disable_warnings()


def download_function(collection):
    """_summary_.

    :param collection: _description_
    """
    request, filepath = collection
    response = requests.get(url=request, verify=False, timeout=300.00).json()

    df = pd.DataFrame.from_dict(response['properties']['parameter'])

    if not os.path.exists('temp'):
        os.makedirs('temp')

    if not os.path.exists(os.sep.join(['temp', filepath])):
        df.to_csv(os.sep.join(['temp', filepath]))


class Process():
    """_summary_."""

    def __init__(self):
        """_summary_."""
        # Please do not go more than five concurrent requests.
        self.processes = 5
        self.request_template = \
            'https://power.larc.nasa.gov/api/temporal/hourly/point?'\
            'parameters={parameters}&'\
            'community=RE&longitude={longitude}&latitude={latitude}&site-elevation={elevation}&'\
            'start={start}&end={end}&'\
            'format=JSON'
        self.filename_template = '{seq}_{year}.csv'
        self.messages = []
        self.times = {}

        arr = [
            'AOD_55', 'AOD_84', 'ALLSKY_KT', 'ALLSKY_NKT', 'ALLSKY_SRF_ALB', 'ALLSKY_SFC_LW_DWN',
            'ALLSKY_SFC_PAR_TOT', 'ALLSKY_SFC_SW_DIFF', 'ALLSKY_SFC_SW_DNI', 'ALLSKY_SFC_SW_DWN',
            'ALLSKY_SFC_UVA', 'ALLSKY_SFC_UVB', 'ALLSKY_SFC_UV_INDEX', 'CLRSKY_KT', 'CLRSKY_SRF_ALB',
            'CLRSKY_SFC_LW_DWN', 'CLRSKY_SFC_PAR_TOT', 'CLRSKY_SFC_SW_DIFF', 'CLRSKY_SFC_SW_DNI',
            'CLRSKY_SFC_SW_DWN', 'PSC', 'CLOUD_AMT', 'CLOUD_OD', 'T2MDEW', 'DIFFUSE_ILLUMINANCE',
            'DIRECT_ILLUMINANCE', 'TS', 'GLOBAL_ILLUMINANCE', 'PW', 'RH2M', 'PRECSNOLAND', 'QV10M',
            'QV2M', 'PS', 'T2M', 'TOA_SW_DNI', 'TOA_SW_DWN', 'T2MWET', 'SZA', 'WS2M', 'WS10M', 'WS50M',
            'WD2M', 'WD10M', 'WD50M', 'PRECTOTCORR'
        ]
        self.sublist_size = ceil(len(arr)/20)
        # split the parameters list into sublists in order to have maximum 15 elements per run (hourly data)
        self.full_parameters = np.array_split(arr, self.sublist_size)

    def execute(self):
        """_summary_."""
        Start_Time = time.time()

        locations = [(49.5, -111.34, 800)]  # latitude, longitude, elevation

        start_date = datetime(2001, 1, 1).date()
        end_date = datetime(2022, 4, 1).date()
        # end_date = datetime.now().date()

        start_list = []
        end_list = []
        while start_date <= end_date:
            start_list.append(start_date.strftime('%Y%m%d'))
            if start_date.year != end_date.year:
                end_list.append(f'{start_date.year}1231')
            else:
                end_list.append(end_date.strftime('%Y%m%d'))
            start_date += relativedelta(years=1)

        requests = []
        for latitude, longitude, elevation in locations:
            for start, end in zip(start_list, end_list):
                for i, parameters in enumerate(self.full_parameters, 1):
                    request = self.request_template.format(
                        latitude=latitude, longitude=longitude, elevation=elevation,
                        parameters=','.join(parameters),
                        start=start, end=end)
                    filename = self.filename_template.format(
                        seq=i, year=start[:4])
                    requests.append((request, filename))

        requests_total = len(requests)

        pool = multiprocessing.Pool(self.processes)
        x = pool.imap_unordered(download_function, requests)

        for i, df in enumerate(x, 1):
            sys.stderr.write('\rExporting {0:%}'.format(i/requests_total))

        self.times['Total Script'] = round((time.time() - Start_Time), 2)

        print('\n')
        print('Total Script Time:', self.times['Total Script'])

    def join_files(self):
        """Join all csv files into one."""
        files = sorted(glob('temp/*.csv'))  # Sort the csv files
        # Convert each csv file to DataFrame
        files = np.array(list(map(pd.read_csv, files)), dtype=object)
        # Split the files for concatenation
        files = np.array_split(files, self.sublist_size)
        # Concatenate the files vertically adding up each year
        files = list(map(pd.concat, files))
        # Rename the first column to datetime
        files = [df.rename(columns={'Unnamed: 0': 'datetime'}) for df in files]
        # Set the first datetime column to be the index
        files = [df.set_index('datetime') for df in files]
        # Concatenate the files horizontally adding up each feature
        file = pd.concat(files, axis=1)
        # Format the datetime column
        file.index = pd.to_datetime(file.index, format='%Y%m%d%H')
        # Delete duplicated columns
        file = file.loc[:, ~file.columns.duplicated()].copy()
        file.to_csv('data/Solar_NASA_BRD1.csv')
        shutil.rmtree(r'temp')


if __name__ == '__main__':
    Process().execute()
    Process().join_files()
