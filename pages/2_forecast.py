import streamlit as st
import datetime as dt
import numpy as np
from solar import Solar

st.set_page_config(layout="wide", initial_sidebar_state="expanded")
st.title('Clear Sky Solar Irradiance Forecast')

st.header("Line Plot")
st.markdown(
"""
In the sidebar select the time interval and the irradiance (target) to be visualized.
In case you select a time range outside the stored data you are going to visualize 
its forecasted results together with the prediction intervals.
"""
)

# --------------------------- Layout setting ---------------------------
window_selection_c = st.sidebar.container() # Create an empty container in the sidebar
window_selection_c.markdown("## Forecasts") # Add a title to the sidebar container
sub_columns = window_selection_c.columns(2) # Split the container into two columns for start and end date
bar = st.progress(0)

# --------------------------- Time window selection ---------------------------
DEFAULT_START_DATE = dt.datetime(2021,12,28) 
LAST_DATE = DEFAULT_START_DATE + dt.timedelta(days=7) #dt.datetime.now().date()
bar.progress(10)
START_DATE = sub_columns[0].date_input("From", value=DEFAULT_START_DATE, max_value=LAST_DATE)
END_DATE = sub_columns[1].date_input("To", value=LAST_DATE, min_value=START_DATE)

DEFAULT_START_TIME = dt.time(16,0,0,0) 
LAST_TIME = DEFAULT_START_TIME #dt.datetime.now().time()
START_TIME = sub_columns[0].time_input("From", value=DEFAULT_START_TIME, label_visibility="collapsed")
END_TIME = sub_columns[1].time_input("To", value=LAST_TIME, label_visibility="collapsed")

START = dt.datetime.combine(START_DATE, START_TIME)
END = dt.datetime.combine(END_DATE, END_TIME)
bar.progress(70)
# --------------------------- Feature selection ---------------------------
SOLAR = np.array(['GHI', 'DHI', 'DNI'])
UNIT = ['KW-hr/m^2/day', 'KW-hr/m^2/day', 'KW-hr/m^2/day']
SYMB = window_selection_c.selectbox("select the feature", SOLAR)

solar = Solar(symbol=SYMB, unit=UNIT[np.where(SOLAR == SYMB)[0][0]])
solar.load_data(START, END, scaler=True)     

# --------------------------- Plot feature linechart ---------------------------
chart_width = st.expander(label="chart width").slider("", 1000, 3000, 1500)

fig = solar.plot_raw_data('chart', forecast=True)

# --------------------------- Styling for plotly ---------------------------
fig.update_layout(width=chart_width,)
st.write(fig)
bar.progress(80)


solar_table = solar.data[['date', 'preds']]
solar_table.rename(columns={'preds': SYMB}, inplace=True)
solar_table.set_index('date', inplace=True)
st.dataframe(solar_table)
bar.progress(100)
