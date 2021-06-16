import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import multiprocessing
import time


class WebApp:
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

    def __init__(self, rtb):
        self.rtb = rtb
        self.app = dash.Dash(__name__, external_stylesheets=self.external_stylesheets)

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
                                  value=timestamps
                              ),
                              html.Div(id='dd-output-container')
                          ]),
                          html.P(id="comparison-text")], style={'width': '49%', 'display': 'inline-block'}),
                html.Div([html.H4("Epoch Summary"),
                          html.P(id='epoch-summary-text')], style={"margin-top": "60px", 'width': '49%', 'display': 'inline-block', "height": "300px"}),
                html.Div([html.H4("Warnings"),
                          html.Div(html.P(id='warnings-text'), style= {"height": "300px", "maxHeight": "300px", "overflow": "scroll"})],
                         style={"margin-top": "60px", 'width': '49%', 'display': 'inline-block'}),
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
            s = metrics["live"]
            loss_color = "green"
            acc_color = "green"
            if s[3].startswith("+"):
                loss_color = "red"
                warningtext = "Epoch: {}, Batch: {} - Loss is increasing".format(s[0], s[1])
                warning = html.P([warning, warningtext, html.Br()])
            if s[5].startswith("-"):
                acc_color = "red"
                warningtext = "Epoch: {}, Batch: {} - Accuracy is decreasing".format(s[0], s[1])
                warning = html.P([warning, warningtext, html.Br()])
            return [
                html.P(

                    ["Epoch: " + s[0], html.Br(), html.P(),
                     "Batch: " + s[1], html.Br(), html.P(),
                     "Time spent: " + s[7], html.Br(), html.P(),
                     "Loss: " + s[2], html.Br(),
                     "    Trend: ", html.P(s[3], style={'color': loss_color, "display" : "inline"}), html.Br(), html.P(),
                     "Accuracy: " + s[4], html.Br(),
                     "    Trend: ", html.P(s[5], style={'color': acc_color, "display" : "inline"}), html.Br(), html.P(),
                     "Memory usage: " + s[6], html.Br(), html.P(),
                     "Energy usage: " + s[10], html.Br(), html.P(),
                     "CPU usage: " + s[11], html.Br(), html.P(),
                     "ETA epoch: " + s[8], html.Br(), html.P(),
                     "ETA training " + s[9], html.Br()],

                    style=style),
            ], warning

        @self.app.callback(Output('epoch-summary-text', 'children'),
                           Input('interval-component', 'n_intervals'))
        def update_epoch(n):
            style = {'padding': '5px', 'fontSize': '16px'}
            s = metrics["epoch"]

            loss_color = "green"
            acc_color = "green"
            if s[4].startswith("+"):
                loss_color = "red"
            if s[6].startswith("-"):
                acc_color = "red"

            return [
                html.P(
                    ["Summary of Epoch " + s[0], html.Br(), html.P(),
                     "Time taken: " + s[1], html.Br(), html.P(),
                     "Time spent on average on batch: " + s[2], html.Br(), html.P(),
                     "Loss: " + s[3], html.Br(),
                     "    Trend: ", html.P(s[4], style={'color': loss_color, "display": "inline"}), html.Br(), html.P(),
                     "Accuracy: " + s[5], html.Br(),
                     "    Trend: ", html.P(s[6], style={'color': acc_color, "display": "inline"}), html.Br(), html.P(),
                     ],
                    style=style),
            ]

        @self.app.callback(Output('comparison-text', 'children'),
                           Input('interval-component', 'n_intervals'))
        def update_comparison(n):
            style = {'padding': '5px', 'fontSize': '16px'}
            s = metrics["comparison"]

            loss_color = "red"
            acc_color = "green"
            if s[4].startswith("-"):
                loss_color = "green"
            if s[6].startswith("-"):
                acc_color = "red"
            return [
                html.P(

                    ["Epoch: " + s[1], html.Br(), html.P(),
                     "Batch: " + s[2], html.Br(), html.P(),
                     "Time spent: " + s[0] + "s", html.Br(), html.P(),
                     "Loss: " + s[3] + ",", html.Br(),
                     "    Trend: ", html.P(s[4] + "%", style={'color': loss_color, "display": "inline"}), html.Br(), html.P(),
                     "Accuracy: " + s[5] + ",", html.Br(),
                     "    Trend: ", html.P(s[6] + "%", style={'color': acc_color, "display": "inline"}), html.Br(), html.P(),
                     "Memory usage: " + s[7] + "MB", html.Br(), html.P(),
                     "Energy usage: " + s[9] + "Wh", html.Br(), html.P(),
                     "CPU usage: " + s[8] + "%", html.Br()],

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
