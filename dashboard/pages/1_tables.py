import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

@st.cache
def get_nasa_data():
    solar = pd.read_csv('data/NASA_Dataset_Cleaned.csv', index_col = 'DATETIME', parse_dates=['DATETIME', 'DATETIME'])
    return solar

@st.cache
def get_solar_map():
     solar_map = pd.read_excel("data/Solar Coordinates.xlsx")
     return solar_map


solar = get_nasa_data()
solar_map = get_solar_map()

st.title("Clear Sky Solar Irradiance")
st.markdown("Welcome to the Clear Sky Solar Irradiance Forecast of the Burdett (BRD1) Solar Farm!\n For this project, we'll use the [Nasa Power Project Dataset](https://power.larc.nasa.gov/) that has the hourly solar and meteorological data from NASA")
st.header("NASA Power Project: data at a glance")
st.markdown("The first five records of the NASA data after being cleaned.")
st.dataframe(solar.head())

st.header("Solar Farm location")
st.markdown("Below in the map of Alberta we can check all the 21 solar farms that will be explored.\nThe mark size shows the maximum capacity of each solar farm.")
solar_map.rename(columns={"Latitude": "lat", "Longitude": "lon"}, inplace=True)
# st.map(solar_map[["lat", "lon"]])



color_scale = [(0, 'orange'), (1,'red')]
fig = px.scatter_mapbox(solar_map, 
                        lat="lat", 
                        lon="lon", 
                        hover_name="Project", 
                        hover_data=["Project", "Max Capacity"],
                        color="Max Capacity",
                        color_continuous_scale=color_scale,
                        size="Max Capacity",
                        size_max=50,
                        zoom=7, 
                        height=600,
                        width=1050)

fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
st.write(fig)


st.subheader("Selecting a subset of columns")
st.write(f"Out of the {solar.shape[1]} columns, you might want to view only a subset.")
defaultcols = list(solar.columns)
cols = st.multiselect("Columns", solar.columns.tolist(), default=defaultcols)
st.dataframe(solar[cols].head(10))