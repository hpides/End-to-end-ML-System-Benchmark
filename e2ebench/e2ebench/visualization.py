from math import floor
import pickle
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import cm
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

class HyperparemeterVisualizer:
    @staticmethod
    def plot_with_plotly(df_from_cli):
        for _, row in df_from_cli.iterrows():
            color_scale = px.colors.diverging.Tealrose
            data_dict = row['measurement_data']
            hyperparams = data_dict['hyperparameters']
            hyperparam_df = data_dict['df']
            target = data_dict['target']
            target_low_means_good = data_dict['low_means_good']

            if not target_low_means_good:
                color_scale = list(reversed(color_scale))

            fig = px.parallel_coordinates(hyperparam_df, 
                                        color=target,
                                        dimensions=hyperparams,
                                        color_continuous_scale=color_scale)
            fig.show()

    @staticmethod
    def plot_with_matplotlib(df_from_cli):
        """
        TODO
        """
        print("Hyperparameters cannot be visualized with matplotlib yet.", file=sys.stderr)

class ConfusionMatrixVisualizer:
    @staticmethod
    def plot_with_matplotlib(df_from_cli):
        for _, row in df_from_cli.iterrows():
            conf_mat_np = row['measurement_data']['matrix']
            labels = row['measurement_data']['labels']
            conf_mat_df = pd.DataFrame(conf_mat_np, 
                                       index=pd.Index(labels, name="predicted"),
                                       columns=pd.Index(labels, name="actual")
            )
            sns.heatmap(conf_mat_df, annot=True)
            plt.show()

    @staticmethod
    def plot_with_plotly(df_from_cli):
        for _, row in df_from_cli.iterrows():
            conf_mat_np = row['measurement_data']['matrix']
            labels = row['measurement_data']['labels']
            layout = {
                'title' : f"Confusion matrix for measurement {row['uuid']} \"{row['measurement_desc']}\"",
                'xaxis' : {'title' : 'actual'},
                'yaxis' : {'title' : 'predicted'}
            }
            fig = go.Figure(
                data = go.Heatmap(z=conf_mat_np, x=labels, y=labels),
                layout=layout
            )
            fig.show()

class TimedeltaMultiLineChartVisualizer:
    """
    Expects a dataframe as created by the CLI.
    Made for plotting multiple time-based graphs such as power and memory usage.
    Each 'measurement_value' entry of the dataframe needs to be a dict of the following structure:
    {'timestamps': [ts1, ts2, ...],
     'measurements' : [m1, m2, ...],
     other key-value pairs}
    Other key-value pairs are values that are general for the entire measurement series such as a sampling interval.
    """
    @staticmethod
    def plot_with_matplotlib(df_from_cli):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)

        for _, run_row in df_from_cli.iterrows():
            measurement_dict = run_row['measurement_data']
            timestamps = pd.to_datetime(measurement_dict.pop('timestamps'))
            relative_timestamps = timestamps - timestamps[0]
            measurements = measurement_dict.pop('measurements')
            ax.plot(relative_timestamps, measurements,
                    label=f"{run_row['uuid']}, \"{run_row['measurement_desc']}\"")
        
        ax.set_ylabel(df_from_cli.loc[0, 'measurement_unit'])
        ax.set_xlabel("Elapsed time in seconds")
        plt.title(df_from_cli.loc[0, 'measurement_type'])

        ax.yaxis.set_major_locator(ticker.LinearLocator(12))
        plt.legend()
        plt.show()

    @staticmethod
    def plot_with_plotly(df_from_cli):
        fig = go.Figure()
        
        for _, run_row in df_from_cli.iterrows():
            measurement_dict = run_row['measurement_data']
            timestamps = pd.to_datetime(measurement_dict.pop('timestamps'))
            relative_timestamps = timestamps - timestamps[0]
            measurements = measurement_dict.pop('measurements')
            fig.add_trace(go.Scatter(
                x=relative_timestamps, y=measurements,
                mode='lines+markers',
                name=f"{run_row['uuid']}, \"{run_row['measurement_desc']}\""
            ))

        fig.update_layout(
            title=df_from_cli.loc[0, 'measurement_type'],
            xaxis_title="Time elapsed since start of measurement",
            yaxis_title=df_from_cli.loc[0, 'measurement_unit']
        )
        fig.show()

class EpochMultiLineChartVisualizer:
    """
    Like TimedeltaMultiLineChartVisualizer, but uses epoch-id on the xaxis instead of elapsed time.
    Each 'measurement_value' entry of the dataframe needs to be a list of measurements:
    The epoch-ids are assumed from the index in this list.
    """

    @staticmethod
    def plot_with_matplotlib(df_from_cli):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)

        for _, run_row in df_from_cli.iterrows():
            measurements = run_row['measurement_data']
            epoch_ids = np.arange(1, len(measurements) + 1)
            plt.xticks()
            ax.plot(epoch_ids, measurements,
                    label=f"{run_row['uuid']}, \"{run_row['measurement_desc']}\" from {run_row['measurement_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        ax.set_ylabel(df_from_cli.loc[0, 'measurement_unit'])
        ax.set_xlabel("Epoch number")
        plt.title(df_from_cli.loc[0, 'measurement_type'])

        ax.yaxis.set_major_locator(ticker.LinearLocator(12))
        plt.legend()
        plt.show()

    @staticmethod
    def plot_with_plotly(df_from_cli):
        fig = go.Figure()
        
        for _, run_row in df_from_cli.iterrows():
            measurements = run_row['measurement_data']
            epoch_ids = np.arange(1, len(measurements) + 1)
            fig.add_trace(go.Scatter(
                x=epoch_ids, y=measurements,
                mode='lines+markers',
                name=f"{run_row['uuid']}, \"{run_row['measurement_desc']}\""
            ))

        fig.update_layout(
            title=df_from_cli.loc[0, 'measurement_type'],
            xaxis_title="Epochs elapsed",
            yaxis_title=df_from_cli.loc[0, 'measurement_unit']
        )
        fig.show()

class BarhVisualizer:
    @staticmethod
    def prepare_df(df_from_cli):
        df_from_cli['measurement_time_str'] = df_from_cli['measurement_time'].dt.strftime("%Y-%m-%d %H:%M:%S")
        df_from_cli['x_labels'] = df_from_cli['measurement_time_str'] + " \"" + df_from_cli['measurement_desc']
        df_from_cli.sort_values(by='measurement_time', inplace=True)

        return df_from_cli

    @staticmethod
    def plot_with_matplotlib(df_from_cli):
        prepared_df = BarhVisualizer.prepare_df(df_from_cli)
        
        ax = prepared_df.plot.barh(x='x_labels', y='measurement_data', stacked=False)
        plt.title(prepared_df.loc[0, 'measurement_type'])
        plt.xlabel(prepared_df.loc[0, 'measurement_unit'])

        x_offset = 0
        y_offset = 0.02
        for p in ax.patches:
            b = p.get_bbox()
            val = "{:.2f}".format(b.x1 - b.x0)
            ax.annotate(val, ((b.x0 + b.x1) / 2 + x_offset, b.y1 + y_offset))

        plt.show()

    @staticmethod
    def plot_with_plotly(df_from_cli):
        prepared_df = BarhVisualizer.prepare_df(df_from_cli)

        fig = px.bar(prepared_df, 
                    x='x_labels', y='measurement_data',
                    hover_data={'value' : prepared_df['measurement_data'].map(
                                    lambda val: f"{val} {prepared_df.loc[0, 'measurement_unit']}"),
                                'uuid': True,
                                'type': prepared_df['measurement_type'],
                                'description': prepared_df['measurement_desc'],
                                'meta description' : prepared_df['meta_desc'].replace('', 'None'),
                                'meta start time' : prepared_df['meta_start_time'].dt.strftime("%Y-%m-%d %H:%M:%S")},
                    color='measurement_data',
                    labels={'x_labels' : 'Measurement', 'measurement_data' : prepared_df.loc[0, 'measurement_unit']}
        )
        fig.update_layout(
            title=prepared_df.loc[0, 'measurement_type']
        )
        fig.show()

visualization_func_mapper = {
    "matplotlib" : {
        "throughput" : BarhVisualizer.plot_with_matplotlib,
        "latency" : BarhVisualizer.plot_with_matplotlib,
        "power" : TimedeltaMultiLineChartVisualizer.plot_with_matplotlib,
        "energy" : BarhVisualizer.plot_with_matplotlib,
        "memory" : TimedeltaMultiLineChartVisualizer.plot_with_matplotlib,
        "time" : BarhVisualizer.plot_with_matplotlib,
        "loss" : EpochMultiLineChartVisualizer.plot_with_matplotlib,
        "tta" : EpochMultiLineChartVisualizer.plot_with_matplotlib,
        "confusion-matrix" : ConfusionMatrixVisualizer.plot_with_matplotlib,
        "hyperparameters" : HyperparemeterVisualizer.plot_with_matplotlib
    },
    "plotly" : {
        "throughput" : BarhVisualizer.plot_with_plotly,
        "latency" : BarhVisualizer.plot_with_plotly,
        "power" : TimedeltaMultiLineChartVisualizer.plot_with_plotly,
        "energy" : BarhVisualizer.plot_with_plotly,
        "memory" : TimedeltaMultiLineChartVisualizer.plot_with_plotly,
        "time" : BarhVisualizer.plot_with_plotly,
        "loss" : EpochMultiLineChartVisualizer.plot_with_plotly,
        "tta" : EpochMultiLineChartVisualizer.plot_with_plotly,
        "confusion-matrix" : ConfusionMatrixVisualizer.plot_with_plotly,
        "hyperparameters" : HyperparemeterVisualizer.plot_with_plotly
    }
}