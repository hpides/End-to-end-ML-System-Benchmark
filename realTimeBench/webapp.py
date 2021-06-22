import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table
import multiprocessing
import time


class WebApp:
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

    def __init__(self, rtb):
        self.rtb = rtb
        self.app = dash.Dash(__name__, external_stylesheets=self.external_stylesheets)
        self.acc_counter = self.loss_counter = 0

    def run(self, metrics, debug=False):

        timestamps = self.rtb.getComparisonOptions()

        self.app.layout = html.Div(
            html.Div([
                html.H4('Real Time Benchmarker'),
                html.Div([html.H4("Live feed"), html.P(id='live-update-text')], style={'width': '49%', 'display': 'inline-block'}),
                html.Div([html.H4("Comparison with run from ---", id="comparison-title"),
                          html.Div([
                              dcc.Dropdown(
                                  id='comparison-dropdown',
                                  options=[{'label': time, 'value': time} for time in timestamps],
                                  value=timestamps,
                                  clearable=False
                              ),
                              html.Div(id='dd-output-container')
                          ]),
                          html.P(id="comparison-text")], style={"float": "right", 'width': '49%', 'display': 'inline-block'}),
                html.Div([html.H4("Epoch Summary"),
                          html.P(id='epoch-summary-text')], style={'width': '49%', 'display': 'inline-block'}),
                html.Div([html.H4("Warnings"),
                          html.Div(html.P(id='warnings-text', style={"height": 60}))],
                         style={'width': '49%', 'display': 'inline-block', "float": "right"}),
                html.Div([
                    html.H4("Overview of Epochs"),
                    dash_table.DataTable(
                        id='epoch-table',
                        columns=[{'id': 'epoch', 'name': 'Epoch'},
                                 {'id': 'loss', 'name': 'Loss'},
                                 {'id': 'trend_loss', 'name': 'Trend Loss'},
                                 {'id': 'acc', 'name': 'Accuracy'},
                                 {'id': 'trend_acc', 'name': 'Trend accuracy'},
                                 {'id': 'time', 'name': 'Time taken'},
                                 {'id': 'time_batch', 'name': 'Avg time per batch'}],
                        data=[],
                        style_table={
                            'height': 500,
                            'overflowY': 'scroll'
                        }
                    )
                ]),
                dcc.Interval(
                    id='interval-component',
                    interval=1 * 1000,  # in milliseconds
                    n_intervals=0
                ),
            ])
        )

        @self.app.callback([Output('live-update-text', 'children'),
                            Output('warnings-text', 'children')],
                           [Input('interval-component', 'n_intervals')],
                           [State('warnings-text', 'children')])
        def update_batch(n, warning):
            style = {'padding': '5px', 'fontSize': '16px'}
            live = metrics["live"]
            if live["epoch"] is None:
                return html.P(["There have been no metrics tracked so far."], style=style), html.P([""])
            loss_color = "green"
            acc_color = "green"
            if live["loss_trend"] > 0:
                loss_color = "red"
                warningtext = "Epoch: {} - Loss has not decreased since {} batches".format(str(live["epoch"]),
                                                                                           self.loss_counter)
                self.loss_counter += 1
                warning = html.P([warningtext, html.Br()])
            else:
                self.loss_counter = 0
                warning = html.P(html.Br())
            if live["acc_trend"] < 0:
                acc_color = "red"
                warningtext = "Epoch: {} - Accuracy has not increased since {} batches".format(str(live["epoch"]),
                                                                                               self.acc_counter)
                self.acc_counter += 1
                warning = html.P([warning, warningtext, html.Br()])
            else:
                self.acc_counter = 0

            time = str(int(live["time"] // 60)) + "m" + str(int(live["time"] % 60)) + "s"

            if live["phase"] == "train":

                eta_epoch = str(int(live["eta_epoch"]//60)) + "m" + str(int(live["eta_epoch"] % 60)) + "s"
                eta_train = str(int(live["eta_train"]//60)) + "m" + str(int(live["eta_train"] % 60)) + "s"

                return [
                    html.P(
                        ["Phase: Training", html.Br(), html.P(),
                         "Epoch: " + "{:2}/{}".format(live["epoch"], live["no_epochs"]), html.Br(), html.P(),
                         "Batch: " + "{:5}/{}".format(live["batch"], live["no_batches"]), html.Br(), html.P(),
                         "Time spent: " + time, html.Br(), html.P(),
                         "Loss: " + "{:7.4f}".format(live["loss"]), html.Br(),
                         "    Trend: ", html.P("{}%".format(live["loss_trend"]),
                                               style={'color': loss_color, "display" : "inline"}), html.Br(), html.P(),
                         "Accuracy: " + "{:8.4f}".format(live["acc"]), html.Br(),
                         "    Trend: ", html.P("{}%".format(live["acc_trend"]),
                                               style={'color': acc_color, "display" : "inline"}), html.Br(), html.P(),
                         "Memory usage: " + "{:5.0f} MB".format(live["memory"]), html.Br(), html.P(),
                         "Energy usage: " + "{:6.5f} Wh".format(live["energy"]), html.Br(), html.P(),
                         "CPU usage: " + "{}%".format(live["cpu"]), html.Br(), html.P(),
                         "ETA epoch: " + eta_epoch, html.Br(), html.P(),
                         "ETA training " + eta_train, html.Br()],

                        style=style),
                ], warning

            else:
                return [
                           html.P(
                               ["Phase: Testing", html.Br(), html.P(),
                                "Batch: " + "{:5}".format(live["batch"]), html.Br(), html.P(),
                                "Time spent: " + time, html.Br(), html.P(),
                                "Loss: " + "{:7.4f}".format(live["loss"]), html.Br(),
                                "    Trend: ", html.P("{}%".format(live["loss_trend"]),
                                                      style={'color': loss_color, "display": "inline"}), html.Br(),
                                html.P(),
                                "Accuracy: " + "{:8.4f}".format(live["acc"]), html.Br(),
                                "    Trend: ", html.P("{}%".format(live["acc_trend"]),
                                                      style={'color': acc_color, "display": "inline"}), html.Br(),
                                html.P(),
                                "Memory usage: " + "{:5.0f} MB".format(live["memory"]), html.Br(), html.P(),
                                "Energy usage: " + "{:6.5f} Wh".format(live["energy"]), html.Br(), html.P(),
                                "CPU usage: " + "{}%".format(live["cpu"]), html.Br(), html.P()],
                               style=style),
                       ], warning

        @self.app.callback([Output('epoch-summary-text', 'children'),
                            Output('epoch-table', 'data')],
                           [Input('interval-component', 'n_intervals')],
                           [State('epoch-table', 'data')])
        def update_epoch(n, rows):
            style = {'padding': '5px', 'fontSize': '16px'}
            epoch = metrics["epoch"]

            if epoch["epoch"] is None:
                return html.P(["No epoch has finished so far."], style=style), rows

            live = metrics["live"]
            time = str(int(epoch["time"]//60)) + "m" + str(int(epoch["time"] % 60)) + "s"

            if len(rows)+1 < int(live["epoch"]):
                rows.append({"epoch": "{}".format(epoch["epoch"]),
                             "loss": "{:6.4f}".format(epoch["loss"]),
                             "trend_loss": "{}%".format(epoch["loss_trend"]),
                             "acc": "{:6.4f}".format(epoch["acc"]),
                             "trend_acc": "{}%".format(epoch["acc_trend"]),
                             "time": time,
                             "time_batch": "{:4.2f}s".format(epoch["avg_time"])})

            loss_color = "green"
            acc_color = "green"
            if epoch["loss_trend"] > 0:
                loss_color = "red"
            if epoch["acc_trend"] < 0:
                acc_color = "red"

            return [
                html.P(
                    ["Summary of Epoch " + "{}".format(epoch["epoch"]), html.Br(), html.P(),
                     "Time taken: " + time, html.Br(), html.P(),
                     "Time spent on average on batch: " + "{:4.2f}s".format(epoch["avg_time"]), html.Br(), html.P(),
                     "Loss: " + "{:6.4f}".format(epoch["loss"]), html.Br(),
                     "    Trend: ", html.P("{}%".format(epoch["loss_trend"]),
                                                        style={'color': loss_color, "display": "inline"}),
                                                        html.Br(), html.P(),
                     "Accuracy: " + "{:6.4f}".format(epoch["acc"]), html.Br(),
                     "    Trend: ", html.P("{}%".format(epoch["acc_trend"]),
                                                        style={'color': acc_color, "display": "inline"}),
                                                        html.Br(), html.P(),
                     ],
                    style=style),
            ], rows

        @self.app.callback([Output('comparison-text', 'children')],
                           [Input('interval-component', 'n_intervals')])
        def update_comparison(n):
            style = {'padding': '5px', 'fontSize': '16px'}
            comp = metrics["comparison"]
            if not comp["available"]:
                return [html.P(["No comparison has been chosen or the comparison run has ended."], style=style)]
            live = metrics["live"]
            colors = ["red"]*7
            if live["loss"] > comp["loss"]:
                colors[0] = "green"
            if live["acc"] < comp["acc"]:
                colors[1] = "green"
            if live["memory"] > comp["memory"]:
                colors[2] = "green"
            if live["energy"] > comp["energy"]:
                colors[3] = "green"
            if live["cpu"] > comp["cpu"]:
                colors[4] = "green"
            if live["epoch"] < comp["epoch"] or live["epoch"] == comp["epoch"] \
                    and live["batch"] < comp["batch"]:
                colors[5] = "green"

            time = str(int(comp["time"] // 60)) + "m" + str(int(comp["time"] % 60)) + "s"

            return [
                html.P(

                    [html.P("Epoch: " + "{:2}".format(live["epoch"]),
                            style={'color': colors[5], "display": "inline"}), html.Br(), html.P(),
                     html.P("Batch: " + "{:5}".format(live["batch"]),
                            style={'color': colors[5], "display": "inline"}), html.Br(), html.P(),
                     "Time spent: " + time, html.Br(), html.P(),
                     html.P("Loss: " + "{:6.4f}".format(comp["loss"]) + ",",
                            style={'color': colors[0], "display": "inline"}), html.Br(),
                     "    Trend: ", "{}%".format(comp["loss_trend"]), html.Br(), html.P(),
                     html.P("Accuracy: " + "{:6.4f}".format(comp["acc"]) + ",",
                            style={'color': colors[1], "display": "inline"}), html.Br(),
                     "    Trend: ", "{}%".format(comp["acc_trend"]), html.Br(), html.P(),
                     html.P("Memory usage: " "{:5.0f} MB".format(comp["memory"]),
                            style={'color': colors[2], "display": "inline"}), html.Br(), html.P(),
                     html.P("Energy usage: " + "{:6.5f} Wh".format(comp["energy"]),
                            style={'color': colors[3], "display": "inline"}), html.Br(), html.P(),
                     html.P("CPU usage: " + "{}%".format(comp["cpu"]),
                            style={'color': colors[4], "display": "inline"}), html.Br()],
                    style=style),
            ]

        @self.app.callback(
            dash.dependencies.Output('comparison-title', 'children'),
            [dash.dependencies.Input('comparison-dropdown', 'value')])
        def update_output(value):
            if isinstance(value, list):
                return "Comparison with run from ---"
            metrics["comparison_choice"] = value
            return "Comparison with run from {}".format(value)

        self.app.run_server(debug=debug)
