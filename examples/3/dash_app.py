import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import pandas as pd


app = dash.Dash(__name__)

datasets = None

def make_app(data):
    global datasets
    datasets = data
    df1 = datasets[0]
    app.layout = html.Div([
        html.Label('Select X-axis Column'),
        dcc.Dropdown(
            id='x-axis-dropdown',
            options=[{'label': col, 'value': col} for col in df1.columns],
            value=df1.columns[0]
        ),
        html.Label('Select Y-axis Column'),
        dcc.Dropdown(
            id='y-axis-dropdown',
            options=[{'label': col, 'value': col} for col in df1.columns],
            value=df1.columns[1]
        ),
        html.Label('Select Z-axis Column'),
        dcc.Dropdown(
            id='z-axis-dropdown',
            options=[{'label': col, 'value': col} for col in df1.columns],
            value=df1.columns[2]
        ),
        dcc.Graph(id='scatter-plot', style={'width': '90vh', 'height': '90vh', 'justify-content': 'center'})
    ])

    app.run_server(debug=True)

@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('x-axis-dropdown', 'value'),
     Input('y-axis-dropdown', 'value'),
     Input('z-axis-dropdown', 'value')]
)
def update_3d_scatter_plot(x_column, y_column, z_column):
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])

    for i, df in enumerate(datasets):
        trace = go.Scatter3d(
            x=df[x_column],
            y=df[y_column],
            z=df[z_column],
            mode='markers',
            marker=dict(size=6, color=i),  # Assign unique color based on dataset index
            name=f'Dataset {i+1}'
        )
        fig.add_trace(trace)

    fig.update_layout(scene=dict(aspectmode='cube'))
    return fig
