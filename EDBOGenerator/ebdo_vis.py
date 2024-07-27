import plotly.graph_objects as go
import os
import pandas as pd
from generator import EDBOGenerator

def test_vis():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))
    fig.add_trace(go.Scatter(x=[20, 30, 40], y=[50, 60, 70]))
    fig.show()

def vis_origin(generator:EDBOGenerator):
    """
    create source variables for all visualizations from round files. store in df
    """
    pass

def scatter(generator:EDBOGenerator,exclude_rounds:list = None):
    round_data = {}
    for file in os.listdir(generator.run_dir):
        if file.startswith('pred'):
            df = pd.read_csv(os.path.join(generator.run_dir,file))
            round_data[file]

