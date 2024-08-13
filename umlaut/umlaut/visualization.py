import logging
from math import floor
import pickle
import sys
import ast
import statistics

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import cm
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import json

class Visualizer:
    def __init__(self, df_from_cli, plotting_backend):
        self.df = df_from_cli
        self.plotting_backend = plotting_backend

    def plot(self):
        if self.plotting_backend == 'matplotlib':
            return self.plot_with_matplotlib()
        if self.plotting_backend == 'plotly':
            return self.plot_with_plotly()
        if self.plotting_backend == 'text':
            return self.plot_with_text()

class HyperparemeterVisualizer(Visualizer):
    def plot_with_plotly(self):
        figs = []
        for _, row in self.df.iterrows():
            # every row needs to be visualized individually since the hyperparameters do not necessarily line up
            color_scale = px.colors.diverging.Tealrose
            data_dict = row['measurement_data']
            hyperparams = data_dict['hyperparameters']
            measurement_df = data_dict['df']
            target = data_dict['target']
            target_low_means_good = data_dict['low_means_good']

            if not target_low_means_good:
                color_scale = list(reversed(color_scale))

            fig = px.parallel_coordinates(measurement_df, 
                                        color=target,
                                        dimensions=hyperparams,
                                        color_continuous_scale=color_scale)
            figs.append(fig)
        
        return figs

    def plot_with_matplotlib(self):
        """
        TODO
        """
        logging.error("Hyperparameters cannot be visualized with matplotlib yet.")

        # return a list with no figures since the CLI expects a figure or a list of figures
        return []

class ConfusionMatrixVisualizer(Visualizer):
    def plot_with_matplotlib(self):
        plt.rcParams.update({'font.size': 18})
        plt.tight_layout()
        figs = []
        for _, row in self.df.iterrows():
            # every row needs to be visualized individually since each row corresponds to one confusion matrix
            measurement_dict = json.loads(row['measurement_data'])
            conf_mat_np = json.loads(measurement_dict['matrix'].replace(" ", ","))
            labels = measurement_dict['labels']
            conf_mat_df = pd.DataFrame(conf_mat_np, 
                                       index=pd.Index(labels, name="predicted"),
                                       columns=pd.Index(labels, name="actual")
            )
            fig, ax = plt.subplots()
            sns.heatmap(conf_mat_df, annot=True, ax=ax)
            ax.set_title("Metric: Confusion Matrix")
            figs.append(fig)

        return figs

    def plot_with_plotly(self):
        figs = []
        for _, row in self.df.iterrows():
            # every row needs to be visualized individually since each row corresponds to one confusion matrix
            conf_mat_np = row['measurement_data']['matrix']
            labels = row['measurement_data']['labels']
            layout = {
                'title' : f"Confusion matrix for measurement {row['uuid']} \"{row['measurement_description']}\"",
                'xaxis' : {'title' : 'actual'},
                'yaxis' : {'title' : 'predicted'}
            }
            fig = go.Figure(
                data=go.Heatmap(z=conf_mat_np, x=labels, y=labels),
                layout=layout
            )
            figs.append(fig)

        return figs

class TimebasedMultiLineChartVisualizer(Visualizer):
    """
    Expects a dataframe as created by the CLI.
    Made for plotting multiple time-based graphs such as power and memory usage.
    Each 'measurement_value' entry of the dataframe needs to be a dict of the following structure:
    {'timestamps': [ts1, ts2, ...],
     'measurements' : [m1, m2, ...],
     other key-value pairs}
    Other key-value pairs are values that are general for the entire measurement series such as a sampling interval.
    """

    def __init__(self, df_from_cli, plotting_backend):
        super().__init__(df_from_cli, plotting_backend)
        self.timedelta_lists = []
        self.x_tick_vals = []
        self.x_tick_labels = []
        self.measurements_lists = []
        self.linelabels = []

        for _, row in self.df.iterrows():
            measurement_dict = row['measurement_data']
            measurement_dict = json.loads(measurement_dict)
            timestamps = pd.to_datetime(measurement_dict.pop('timestamps'))
            self.timedelta_lists.append(timestamps - timestamps[0])
            x_tick_idx = np.floor(np.linspace(0, len(self.timedelta_lists[-1])-1, 5)).astype(int)
            if len(self.x_tick_vals) == 0 or self.timedelta_lists[-1][x_tick_idx][4] > self.x_tick_vals[4]:
                self.x_tick_vals = self.timedelta_lists[-1][x_tick_idx]
                self.x_tick_labels = self.x_tick_vals.map(self._strfdelta)
            self.measurements_lists.append(measurement_dict.pop('measurements'))
            measurement_time = row['measurement_datetime'].strftime("%Y-%m-%d %H:%M:%S")
            self.linelabels.append("\"" + row['measured_method_name'] + "\"\nfrom\n" + measurement_time)


    def _strfdelta(self, td):
        hours, remainder = divmod(td.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"

    def plot_with_text(self):
        for i in range(len(self.measurements_lists)):
            print("Statistics for operation " + str(i+1) + ":\n")
            print("MAX: " + str(max(self.measurements_lists[i])))
            print("MIN: " + str(min(self.measurements_lists[i])))
            print("MEAN: " + str(statistics.mean(self.measurements_lists[i])))
            print("ST DEV: " + str(statistics.stdev(self.measurements_lists[i])))
            print("===================================" + "\n")

    def plot_with_matplotlib(self):
        plt.rcParams.update({'font.size': 18})
        fig, ax = plt.subplots()
        plt.tight_layout()

        for i in range(len(self.df)):
            ax.plot(self.timedelta_lists[i], self.measurements_lists[i], label=self.linelabels[i])
        
        ax.set_xlabel(self.xaxis_label)
        ax.set_ylabel(self.yaxis_label)
        ax.set_title(self.title)
        ax.set_xticks([td.value for td in self.x_tick_vals])
        ax.set_xticklabels(self.x_tick_labels)

        ax.yaxis.set_major_locator(ticker.LinearLocator(12))
        
        return [fig]

    def plot_with_plotly(self):
        fig = go.Figure()
        
        for i in range(len(self.df)):
            fig.add_trace(go.Scatter(
                x=self.timedelta_lists[i], y=self.measurements_lists[i],
                mode='lines+markers',
                name=self.linelabels[i]
            ))

        fig.update_layout(
            title=self.title,
            xaxis_title=self.xaxis_label,
            yaxis_title=self.yaxis_label,
            xaxis=dict(
                tickmode='array',
                tickvals=self.x_tick_vals,
                ticktext=self.x_tick_labels
            )
        )
        
        return [fig]

class EpochbasedMultiLineChartVisualizer(Visualizer):
    """
    Like TimedeltaMultiLineChartVisualizer, but uses epoch-id on the xaxis instead of elapsed time.
    Each 'measurement_value' entry of the dataframe needs to be a list of measurements.
    The epoch-ids are assumed from the index in this list.
    """

    def plot_with_matplotlib(self):
        plt.rcParams.update({'font.size': 18})
        fig, ax = plt.subplots()
        plt.tight_layout()
        for _, row in self.df.iterrows():
            measurements = row['measurement_data']
            epoch_ids = np.arange(1, len(measurements) + 1)
            measurement_time = row['measurement_datetime'].strftime("%Y-%m-%d %H:%M:%S")
            linelabel = "\"" + row['measured_method_name'] + "\"\nfrom\n" + measurement_time
            
            ax.plot(epoch_ids, measurements, label=linelabel)
        
        ax.set_xlabel(self.xaxis_label)
        ax.set_ylabel(self.yaxis_label)
        ax.set_title(self.title)
        ax.yaxis.set_major_locator(ticker.LinearLocator(12))
        
        return [fig]

    def plot_with_plotly(self):
        fig = go.Figure()
        
        for _, row in self.df.iterrows():
            measurements = row['measurement_data']
            epoch_ids = np.arange(1, len(measurements) + 1)
            measurement_time = row['measurement_datetime'].strftime("%Y-%m-%d %H:%M:%S")
            linelabel = "\"" + row['measurement_description'] + "\"\nfrom\n" + measurement_time

            fig.add_trace(go.Scatter(
                x=epoch_ids, y=measurements,
                mode='lines+markers',
                name=linelabel
            ))

        fig.update_layout(
            title=self.title,
            xaxis_title=self.xaxis_label,
            yaxis_title=self.yaxis_label
        )
        
        return [fig]

class BarVisualizer(Visualizer):
    def __init__(self, df_from_cli, plotting_backend):
        super().__init__(df_from_cli, plotting_backend)
        df_from_cli['measurement_time_str'] = df_from_cli['measurement_datetime'].dt.strftime("%Y-%m-%d %H:%M:%S")
        df_from_cli['x_labels'] = " \"" + df_from_cli['measured_method_name'] + "\"\nfrom\n" + df_from_cli['measurement_time_str']
        df_from_cli.sort_values(by='measurement_datetime', inplace=True)
        self.df = df_from_cli

    def plot_with_text(self):
        print("Time needed for each operation: \n")
        print(self.df["measurement_data"])


    def plot_with_matplotlib(self):
        plt.rcParams.update({'font.size': 18})
        fig, ax = plt.subplots()
        plt.tight_layout()
        self.df.plot.barh(x='x_labels', y='measurement_data', stacked=False, legend=False, ax=ax)
        
        plt.title(self.title)
        # weird because this is a horizontal bar chart
        plt.xlabel(self.yaxis_label)
        plt.ylabel(self.xaxis_label)

        # annotate bars with measurement value
        x_offset = 0
        y_offset = 0.02
        for p in ax.patches:
            b = p.get_bbox()
            val = "{:.2f}".format(b.x1 - b.x0)
            ax.annotate(val, ((b.x0 + b.x1) / 2 + x_offset, b.y1 + y_offset))

        return [fig]

    def plot_with_plotly(self):
        fig = px.bar(self.df, 
                    x='x_labels', y='measurement_data',
                    hover_data={'uuid': True,
                                'description': self.df['measurement_description'],
                                'meta description' : self.df['meta_description'].replace('', 'None'),
                                'meta start time' : self.df['meta_start_time'].dt.strftime("%Y-%m-%d %H:%M:%S")},
                    color='measurement_data',
        )
        fig.update_xaxes(type='category')
        fig.update_layout(
            title=self.title,
            xaxis_title=self.xaxis_label,
            yaxis_title=self.yaxis_label
        )
        
        return [fig]


class ThroughputVisualizer(BarVisualizer):
    title = "Metric: Throughput"
    xaxis_label = ""
    yaxis_label = "Throughput in entries/second"

class LatencyVisualizer(BarVisualizer):
    title = "Metric: Latency"
    xaxis_label = ""
    yaxis_label = "Latencies in Seconds/entry"

class PowerVisualizer(TimebasedMultiLineChartVisualizer):
    title = "Metric: Power"
    xaxis_label = "Time elapsed since start of pipeline run"
    yaxis_label = "Watt"

class GPUPowerVisualizer(TimebasedMultiLineChartVisualizer):
    title = "Metric: GPU Power"
    xaxis_label = "Time elapsed since start of pipeline run"
    yaxis_label = "Watt"

class EnergyVisualizer(BarVisualizer):
    title = "Metric: Power"
    xaxis_label = "Time elapsed since start of pipeline run"
    yaxis_label = "Energy usage in µJ"

class MemoryVisualizer(TimebasedMultiLineChartVisualizer):
    title = "Metric: Memory usage"
    xaxis_label = "Seconds elapsed since start of pipeline run"
    yaxis_label = "Memory usage in MB"

class GPUMemoryVisualizer(TimebasedMultiLineChartVisualizer):
    title = "Metric: GPU Memory usage"
    xaxis_label = "Seconds elapsed since start of pipeline run"
    yaxis_label = "GPU Memory usage in MB"

class TimeVisualizer(BarVisualizer):
    title = "Metric: Time"
    xaxis_label = ""
    yaxis_label = "Time taken in seconds"

class GPUTimeVisualizer(BarVisualizer):
    title = "Metric: Time"
    xaxis_label = ""
    yaxis_label = "Time taken in seconds"

class LossVisualizer(EpochbasedMultiLineChartVisualizer):
    title = "Metric: Loss over epochs"
    xaxis_label = "Epoch ID"
    yaxis_label = "Loss"

class TTAVisualizer(EpochbasedMultiLineChartVisualizer):
    title = "Metric: Time-to-accuracy"
    xaxis_label = "Epoch ID"
    yaxis_label = "Accuracy"

class CPUVisualizer(TimebasedMultiLineChartVisualizer):
    title = "Metric: CPU usage"
    xaxis_label = "Time elapsed since start of pipeline run"
    yaxis_label = "CPU usage in %"

class GPUVisualizer(TimebasedMultiLineChartVisualizer):
    title = "Metric: GPU usage"
    xaxis_label = "Time elapsed since start of pipeline run"
    yaxis_label = "GPU usage in %"

class TimedTTAVisualizer(BarVisualizer):
    title = "Metric: Time to target accuracy"
    xaxis_label = "Time elapsed since until target accuracy reached"
    yaxis_label = "Time taken in seconds"

type_to_visualizer_class_mapper = {
    "throughput" : ThroughputVisualizer,
    "latency" : LatencyVisualizer,
    "power" : PowerVisualizer,
    "gpupower": GPUPowerVisualizer,
    "energy" : EnergyVisualizer,
    "memory" : MemoryVisualizer,
    "gpumemory" : GPUMemoryVisualizer,
    "time" : TimeVisualizer,
    "gputime" : GPUTimeVisualizer,
    "loss" : LossVisualizer,
    "tta" : TTAVisualizer,
    "confusion-matrix" : ConfusionMatrixVisualizer,
    "hyperparameters" : HyperparemeterVisualizer,
    "cpu": CPUVisualizer,
    "gpu": GPUVisualizer,
    "timed tta": TimedTTAVisualizer
    }
