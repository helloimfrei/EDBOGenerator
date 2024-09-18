import plotly.graph_objects as go
import os
import pandas as pd

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from generator import EDBOGenerator

def test_vis():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))
    fig.add_trace(go.Scatter(x=[20, 30, 40], y=[50, 60, 70]))
    fig.show()

def collect_preds(generator:'EDBOGenerator'):
    """
    Collect prediction data from run dir, store in dict as df
    """
    preds = {}
    for file in os.listdir(generator.run_dir):
        if file.startswith('pred'):
            preds[file.replace('.csv','')] = pd.read_csv(file)
    return {k.replace(f'pred_{generator.run_name}_',''):v for k,v in preds.items()}


def scatter(generator:'EDBOGenerator'):

    pass

