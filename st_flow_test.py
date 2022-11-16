# import libs
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import streamlit as st
from scipy import stats


#######################################
# Load the data
#######################################
df_flow = (pd
 .read_csv('raw_data/d12_hr_MLHVflow_201902.csv', parse_dates=['timestamp'])
 .assign(timestamp=lambda df_: df_['timestamp'].dt.tz_localize(None))
 .assign(year=lambda df_: df_['timestamp'].dt.year,
         month=lambda df_: df_['timestamp'].dt.month,
         day=lambda df_: df_['timestamp'].dt.day,
         hour=lambda df_: df_['timestamp'].dt.hour)
)


#######################################
# Define functions
#######################################
@st.cache
def route_df(route_num):
    route_raw = df_flow[(df_flow.route==route_num)]
    pivot_raw = pd.pivot_table(route_raw, values='total_flow', index=['station_id', 'abs_pm'], columns=['hour'], aggfunc=np.mean).sort_values('abs_pm')
    route_raw_new = pivot_raw[(np.abs(stats.zscore(pivot_raw)) < 3)].dropna().reset_index()
    route_flow_df = route_raw[route_raw['station_id'].isin(route_raw_new['station_id'])]
    route_flow_df = route_flow_df.assign(day_of_week=lambda x: (x['timestamp'].dt.dayofweek))
    route_flow_df['weekday'] = route_flow_df['day_of_week'].apply(lambda x: 'weekdays' if x < 6 else 'weekends')
    return route_flow_df

@st.cache
def weekday_pivot_df(route_df):
    direction_a = route_df['direction'].unique()[0]
    direction_b = route_df['direction'].unique()[1]
    route_df_direction_a_pivot = pd.pivot_table(route_df[route_df['direction']==direction_a], values='total_flow', index=['hour'], columns=['weekday'], aggfunc=np.mean).reset_index()
    route_df_direction_b_pivot = pd.pivot_table(route_df[route_df['direction']==direction_b], values='total_flow', index=['hour'], columns=['weekday'], aggfunc=np.mean).reset_index()
    route_df_pivot = route_df_direction_a_pivot.add(route_df_direction_b_pivot, fill_value=0)
    route_df_pivot['hour'] = route_df_pivot['hour']/2
    route_df_pivot = route_df_pivot.round(0)
    return route_df_pivot



#######################################
#Start building Streamlit App
#######################################


routes_number_list = df_flow['route'].unique()
routes_number_list = list(routes_number_list)
# routes_number_list.insert(0, 'Select the route.')


# streamlit sidebar
route_number = st.sidebar.selectbox("Choose the route", (routes_number_list))

route_num_df = route_df(route_number)
route_flow_df = weekday_pivot_df(route_num_df)

st.write(f"{route_number} is selected!")
st.dataframe(route_flow_df.style.format("{:.0f}"))

fig = go.Figure()
fig.add_trace(go.Scatter(x=route_flow_df['hour'], y=route_flow_df['weekdays'], name='weekday flow', line=dict(color='red', width=4)))
fig.add_trace(go.Scatter(x=route_flow_df['hour'], y=route_flow_df['weekends'], name='weekend flow', line=dict(color='royalblue', width=4)))
# fig = px.line(route_flow_df, x='hour', y=route_flow_df.columns, range_x=(-0.5, 23.5), title=f"Weekday vs. Weekend Avg. Hourly Flow for Route {route_number}", markers=True)
fig.update_yaxes(minor_ticks="inside")
fig.update_layout(title=f"Weekday vs. Weekend Avg. Hourly Flow for Route {route_number}",
                   xaxis_title='Hour',
                   yaxis_title='Avg. Flow')
st.plotly_chart(fig, use_container_width=True)
