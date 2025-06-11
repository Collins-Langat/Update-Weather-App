#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('pip', 'install pandas requests dash dash-bootstrap-components plotly scikit-learn xgboost joblib jupyter-dash')


# In[3]:


# ------------------- IMPORTS -------------------
import requests
import pandas as pd
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import plotly.express as px
from dash.dependencies import Input, Output, State
import joblib
from datetime import datetime


# ------------------- LOAD MODEL -------------------
xgb_model = joblib.load(r'C:\Users\colli\Desktop\stuff\rain_predictor_xgb.pkl')
if xgb_model is None:
    raise ValueError("Failed to load the XGBoost model. Check the file path.")

# ------------------- FETCH WEATHER DATA -------------------
def fetch_weather_data(latitude, longitude):
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": [
            "daylight_duration", "temperature_2m_mean", "cape_mean", "cape_max", "cape_min",
            "cloud_cover_mean", "cloud_cover_max", "cloud_cover_min",
            "dew_point_2m_mean", "relative_humidity_2m_mean", "relative_humidity_2m_max", "relative_humidity_2m_min",
            "surface_pressure_mean", "surface_pressure_max",
            "wind_gusts_10m_mean", "winddirection_10m_dominant", "wind_speed_10m_mean",
            "visibility_mean", "visibility_min", "visibility_max",
            "precipitation_probability_mean", "updraft_max", "rain_sum"
        ],
        "current": "rain",
        "timezone": "America/New_York"
    }
    response = requests.get("https://api.open-meteo.com/v1/forecast", params=params)
    data = response.json()
    if 'daily' not in data or not data['daily']:
        raise ValueError("No daily data returned from API.")
    df = pd.DataFrame(data['daily'])
    if 'time' in df.columns:
        df['date'] = pd.to_datetime(df['time'])
        df.drop(columns=['time'], inplace=True)
    else:
        raise ValueError("Missing 'time' column in response.")
    return df

# ------------------- INITIAL SETUP -------------------
initial_lat = 38.2542
initial_lon = -85.7594
df = fetch_weather_data(initial_lat, initial_lon)

# ------------------- DASH APP -------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = "Rainfall Prediction Dashboard"

app.layout = dbc.Container([

    dbc.Row([
        dbc.Col(html.Img(src='/assets/logo.jpeg', style={
            'height': '80px',
            'margin': '0 auto',
            'display': 'block',
            'paddingBottom': '10px'
        }))
    ]),

    dbc.Row([dbc.Col(html.H1("Rainfall Prediction Dashboard", style={'textAlign': 'center', 'color': '#FFFFFF'}))]),

    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("📍 Location Selector", style={'fontWeight': 'bold'}),
            dbc.CardBody([
                html.Label("Latitude:", style={'color': 'white'}),
                dcc.Input(id='input-lat', type='number', value=initial_lat, step=0.01),
                html.Label("Longitude:", style={'color': 'white', 'marginLeft': '20px'}),
                dcc.Input(id='input-lon', type='number', value=initial_lon, step=0.01),
                html.Br(), html.Br(),
                html.Button('🔄 Update Location', id='submit-coordinates', n_clicks=0, className='btn btn-info')
            ])
        ], style={'backgroundColor': '#2b2b2b'}, inverse=True))
    ], style={'marginBottom': '20px'}),

    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("📊 Variable to Plot", style={'fontWeight': 'bold'}),
            dbc.CardBody([
                dcc.Dropdown(
                    id='variable-dropdown',
                    options=[
                        {'label': '🌧️ Rain Sum', 'value': 'rain_sum'},
                        {'label': '🌡️ Temperature', 'value': 'temperature_2m_mean'},
                        {'label': '💧 Humidity (max)', 'value': 'relative_humidity_2m_max'},
                        {'label': '🌬️ Wind Speed', 'value': 'wind_speed_10m_mean'},
                        {'label': '☁️ Cloud Cover (mean)', 'value': 'cloud_cover_mean'},
                        {'label': '🔮 Predicted Rain Probability (%)', 'value': 'Rain_Probability'}
                    ],
                    value='rain_sum',
                    clearable=False,
                    style={
                        'color': 'black',
                        'backgroundColor': '#e0e0e0',
                        'border': '1px solid #ccc',
                        'borderRadius': '4px',
                        'padding': '6px'
                    }
                )
            ])
        ], style={'backgroundColor': '#2b2b2b'}, inverse=True))
    ], style={'marginBottom': '20px'}),

    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("📅 Days to Display"),
            dbc.CardBody([
                dcc.Slider(
                    id='day-slider',
                    min=1, max=len(df), step=1, value=len(df),
                    marks={i: str(i) for i in range(1, len(df) + 1)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ])
        ], style={'backgroundColor': '#2b2b2b'}, inverse=True))
    ], style={'marginBottom': '30px'}),

    html.H3("🗓️ Forecast Table", style={'color': 'white', 'marginTop': 30}),
    html.Div(id='forecast-table'),

    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("📊 Forecast Visualization", style={'fontWeight': 'bold'}),
            dbc.CardBody([dcc.Graph(id='main-graph', config={'displayModeBar': False})])
        ], color="secondary", inverse=False))
    ]),

    html.Footer("⚙️ Built with Dash • Powered by Open-Meteo API", 
                style={'textAlign': 'center', 'color': 'gray', 'marginTop': 40, 'fontSize': '0.9em'})

], fluid=True, style={'backgroundColor': '#1c1c1e', 'padding': '30px'})

# ------------------- CALLBACK -------------------
@app.callback(
    [Output('main-graph', 'figure'),
     Output('forecast-table', 'children'),
     Output('day-slider', 'max'),
     Output('day-slider', 'marks')],
    [Input('submit-coordinates', 'n_clicks'),
     Input('variable-dropdown', 'value'),
     Input('day-slider', 'value')],
    [State('input-lat', 'value'),
     State('input-lon', 'value')]
)
def update_dashboard(n_clicks, selected_variable, num_days, lat, lon):
    df_updated = fetch_weather_data(lat, lon)
    features_df = df_updated[['cloud_cover_min', 'dew_point_2m_mean', 'relative_humidity_2m_max', 'cape_mean']]
    rain_probs = xgb_model.predict_proba(features_df)[:, 1]
    rain_binary = (rain_probs > 0.5).astype(int)
    df_updated['Rain_Probability'] = (rain_probs * 100).round(2)
    df_updated['Rain_Predicted'] = rain_binary
    df_updated['Will It Rain Today?'] = df_updated['Rain_Predicted'].map({1: "Yes", 0: "No"})
    df_updated['date'] = df_updated['date'].dt.strftime('%b %d, %Y')

    filtered_df = df_updated.head(num_days)
    fig = px.line(filtered_df, x='date', y=selected_variable, title=f"{selected_variable} Over Time")
    fig.update_layout(plot_bgcolor='#2c2c2c', paper_bgcolor='#2c2c2c', font_color='white',
                      title_font_size=18, hovermode='x unified', margin=dict(l=30, r=30, t=60, b=30))

    table_df = df_updated[['date', 'rain_sum', 'Rain_Probability', 'Will It Rain Today?']].head(num_days)
    table = dbc.Table.from_dataframe(table_df, striped=True, bordered=True, hover=True, style={"color": "white"})

    new_max = len(df_updated)
    new_marks = {i: str(i) for i in range(1, new_max + 1)}

    return fig, table, new_max, new_marks

# ------------------- RUN APP -------------------
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)


