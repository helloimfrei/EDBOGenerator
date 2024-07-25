import plotly.graph_objects as go
import pandas as pd

def test_vis():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))
    fig.add_trace(go.Scatter(x=[20, 30, 40], y=[50, 60, 70]))
    fig.show()
