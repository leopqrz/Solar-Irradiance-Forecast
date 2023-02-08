# Core
import streamlit as st
import datetime as dt
import numpy as np
from solar import Solar
import urllib.request


# Visualization
from PIL import Image

st.set_page_config(
  layout="wide", 
  initial_sidebar_state="expanded",
  page_title="Intro",
  page_icon="https://s3.dualstack.us-east-2.amazonaws.com/pythondotorg-assets/media/community/logos/python-logo-only.png",
)

image_url = "https://www.python.org/static/community_logos/python-logo.png"
filename = "python-logo.png"
urllib.request.urlretrieve(image_url, filename)
image = Image.open(filename)
st.image(image, caption='', width=300)

# st.write("# Solar Power Forecast")
# st.sidebar.success("Select a demo above.")

st.title('Independent variables')

st.header("Line Plot")
st.markdown(
"""
In the sidebar select the time interval and the independed variable (feature) to be visualized.
"""
)

# --------------------------- Layout setting ---------------------------
window_selection_c = st.sidebar.container() # Create an empty container in the sidebar
window_selection_c.markdown("## Insights") # Add a title to the sidebar container
sub_columns = window_selection_c.columns(2) # Split the container into two columns for start and end date
bar = st.progress(0)

# --------------------------- Time window selection ---------------------------
LAST_DATE = dt.datetime(2021,12,31) #dt.datetime.now().date()
DEFAULT_START_DATE = dt.datetime(2001,1,1) #LAST_DATE - dt.timedelta(days=365)
bar.progress(10)
START_DATE = sub_columns[0].date_input("From", value=DEFAULT_START_DATE, max_value=LAST_DATE)
END_DATE = sub_columns[1].date_input("To", value=LAST_DATE, min_value=START_DATE)

LAST_TIME = dt.time(16,0,0,0) #dt.datetime.now().time()
DEFAULT_START_TIME = dt.time(0,0,0,0) #LAST_TIME
START_TIME = sub_columns[0].time_input("From", value=DEFAULT_START_TIME, label_visibility="collapsed")
END_TIME = sub_columns[1].time_input("To", value=LAST_TIME, label_visibility="collapsed")

START = dt.datetime.combine(START_DATE, START_TIME)
END = dt.datetime.combine(END_DATE, END_TIME)
bar.progress(30)
# --------------------------- Feature selection ---------------------------
SOLAR = np.array(['CLRSKY_SRF_ALB', 'T2MDEW', 'DIFFUSE_ILLUMINANCE','DIRECT_ILLUMINANCE','GLOBAL_ILLUMINANCE',\
    'RH2M', 'QV2M', 'PS', 'T2M', 'SZA', 'WS2M'])#, 'DHI', 'DNI', 'GHI', 'Year', 'Month', 'Day', 'Hour', 'season'])
UNIT = ['dimensionless', '°C', 'lux', 'lux', 'lux', '%', 'g/Kg', 'KPa', '°C', 'Not Available', 'm/s']
SYMB = window_selection_c.selectbox("select the feature", SOLAR)

solar = Solar(symbol=SYMB, unit=UNIT[np.where(SOLAR == SYMB)[0][0]])
solar.load_data(START, END)

# --------------------------- Plot feature linechart ---------------------------
chart_width = st.expander(label="chart width").slider("", 1000, 3000, 1500)

fig1 = solar.plot_raw_data('chart')

# --------------------------- Styling for plotly ---------------------------
fig1.update_layout(width=chart_width,)
st.write(fig1)
bar.progress(70)
# --------------------------- Plot feature histogram ---------------------------
st.header("Histogram Plot")
nbins = st.expander(label="number of bins").slider("", 5, 200, 20)

fig2 = solar.plot_raw_data('hist')

# --------------------------- Styling for plotly ---------------------------
fig2.update_traces(nbinsx=nbins,)
st.write(fig2)
bar.progress(100)
