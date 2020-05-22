import numpy as np
import plotly.graph_objects as go


class TrainScatterPlot:
    def __init__(self, xlabel, ylabel):        
        self.fig = go.FigureWidget(
            data=go.Scatter(
                x=[], y=[],
                text=[],
                mode='markers',
                marker=dict(
                    size=10,
                    showscale=True
                ),
            ),
            layout=go.Layout(
                xaxis=dict(
                    title=xlabel,
                    type='log',
                    tickformat='.0e',
                ),
                yaxis=dict(
                    tickformat='.2',
#                     type='log',                
                    title=ylabel
                ), 
            )
        )
    
    def insert(self, x, y, z=None):
        scatter = self.fig.data[0]
        scatter.x = np.append(scatter.x, x)
        scatter.y = np.append(scatter.y, y)
        
        if z is not None:
            scatter.text = np.append(scatter.text, z)
            scatter.marker.color = scatter.text


