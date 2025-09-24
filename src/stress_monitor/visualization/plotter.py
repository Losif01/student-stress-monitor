import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Optional
import numpy as np

class StressDataVisualizer:
    """Class for creating visualizations of stress data"""
    
    def __init__(self, style: str = "seaborn"):
        self.style = style
        self.set_style()
    
    def set_style(self):
        """Set plotting style"""
        if self.style == "seaborn":
            sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def create_histograms(self, df: pd.DataFrame, cols: Optional[List[str]] = None) -> go.Figure:
        """Create histogram grid using Plotly"""
        if cols is None:
            cols = df.columns.tolist()
        
        n_cols = 3
        n_rows = (len(cols) + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=cols
        )
        
        for i, col in enumerate(cols):
            row = i // n_cols + 1
            col_num = i % n_cols + 1
            
            fig.add_trace(
                go.Histogram(x=df[col], name=col, nbinsx=20),
                row=row, col=col_num
            )
        
        fig.update_layout(
            height=300 * n_rows,
            title_text="Feature Distributions",
            showlegend=False
        )
        
        return fig
    
    def create_correlation_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Create correlation heatmap using Plotly"""
        corr_matrix = df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu_r',
            zmin=-1,
            zmax=1,
            hoverongaps=False,
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title="Feature Correlation Heatmap",
            xaxis_title="Features",
            yaxis_title="Features",
            width=800,
            height=600
        )
        
        return fig
    
    def create_model_comparison_chart(self, results: dict) -> go.Figure:
        """Create bar chart comparing model performances"""
        models = list(results.keys())
        accuracies = [results[model] for model in models]
        
        fig = go.Figure(data=[
            go.Bar(x=models, y=accuracies, 
                  marker_color=px.colors.qualitative.Set3)
        ])
        
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Models",
            yaxis_title="Accuracy",
            yaxis=dict(range=[0, 1]),
            width=800,
            height=500
        )
        
        return fig