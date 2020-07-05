import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from typing import List, AnyStr
from enum import IntEnum

from dash.dependencies import Input, Output, State
from plotly.subplots import make_subplots
import numpy as np

import helmholtz1d


class ResultFunction:

    # maybe add more colors
    COLORS = ['crimson', 'darkslateblue']

    def __init__(self, name, x, y, color=None):
        self.name = name
        self.x = x
        self.y = y
        if color is not None:
            self.color = color
        else:
            self.color = ResultFunction.COLORS[0]


class PlotType(IntEnum):
    Real_Imaginary = 1,
    Magnitude_Phase = 2,
    Real = 3,
    Imaginary = 4,
    Magnitude = 5,
    Phase = 6


def split_complex_values(values, plot_type: PlotType):
    if plot_type == PlotType.Real_Imaginary:
        return [(np.real(values), 'real'), (np.imag(values), 'imaginary')]

    if plot_type == PlotType.Magnitude_Phase:
        return [(np.round(np.absolute(values), 4), 'amplitude'), (np.angle(values), 'phase')]

    if plot_type == PlotType.Real:
        return [(np.real(values), 'real')]

    if plot_type == PlotType.Imaginary:
        return [(np.imag(values), 'imaginary')]

    if plot_type == PlotType.Magnitude:
        return [(np.round(np.absolute(values), 4), 'amplitude')]

    if plot_type == PlotType.Phase:
        return [(np.angle(values), 'phase')]

    raise Exception("Not implemented plot type.")


def create_2d_plot(functions: List[ResultFunction],
                   plot_type: PlotType,
                   title: AnyStr,
                   x_axis: AnyStr,
                   y_axis: AnyStr):

    num_subplots = 1
    if plot_type == PlotType.Real_Imaginary or plot_type == PlotType.Magnitude_Phase:
        num_subplots = 2

    fig = make_subplots(num_subplots, 1)

    for f in functions:
        if f is None:
            continue

        x = f.x
        values = split_complex_values(f.y, plot_type)
        y1, label_1 = values[0][0], values[0][1]
        fig.add_trace(go.Scatter(x=x, y=y1,
                                 name=f.name,
                                 marker_color=f.color
                                 ), row=1, col=1)
        fig.update_yaxes(title_text="{} ({})".format(y_axis, label_1), row=1, col=1)

        if num_subplots > 1:
            y2, label_2 = values[1][0], values[1][1]
            fig.add_trace(go.Scatter(x=x, y=y2,
                                     showlegend=False,
                                     marker_color=f.color
                                     ), row=2, col=1)
            fig.update_yaxes(title_text="{} ({})".format(y_axis, label_2), row=2, col=1)

        fig.update_xaxes(title_text="{}".format(x_axis), row=num_subplots, col=1)

    return fig


class MainDashboard:
    def __init__(self):
        # dcc._css_dist[0]['relative_package_path'].append('style.css')
        external_stylesheets = [dbc.themes.LUX, 'style.css']
        self.app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
        self.app.title = 'Helmholtz 1D'
        self.set_layout()
        self.computed_func = None
        self.reference_func = None
        self.sim = None

        self.app.callback(
            [
                Output('plot_1', 'figure'),
                Output('plot_2', 'figure'),
            ],
            [
                Input('compute', 'n_clicks'),
            ],
            [
                State('select-1', 'value'),
                State('select-2', 'value'),
                State('duct-length', 'value'),
                State('num-elements', 'value'),
                State('rho', 'value'),
                State('c', 'value'),
                State('omega', 'value'),
                State('v', 'value'),
                State('beta', 'value'),
                State('p-fem', 'value'),
            ]
        )(self.compute)

    def compute_helmholtz(self, argv):
        L = argv[0]
        num_elements = argv[1]
        rho = argv[2]
        c = argv[3]
        omega = argv[4]
        velocity = argv[5]
        beta = argv[6]
        p_fem = argv[7]

        mesh = helmholtz1d.create_1d_mesh(L, num_elements)
        fluid_properties = helmholtz1d.FluidProperties(rho, c)
        load = helmholtz1d.VelocityLoad(omega, velocity=1j * omega * rho * velocity, node_index=0)
        impedance = helmholtz1d.Impedance(beta, node_index=len(mesh.node_coordinates) - 1)
        impedance.add_impedance(1-beta, node_index=0)
        self.sim = helmholtz1d.HelmholtzSimulation(mesh, load, impedance, fluid_properties)
        solution = self.sim.compute(helmholtz1d.ComputationParameters(p_fem))
        x, y = helmholtz1d.interpolate_higher_order_solution(mesh, solution, p_fem*10)
        return x, y

    def update_left_plot(self, func_type):
        func_1 = create_2d_plot([self.computed_func, self.reference_func],
                                plot_type=func_type,
                                title='Pressure Response',
                                x_axis='Length',
                                y_axis='Pressure')
        return func_1

    def update_right_plot(self, func_type):
        func_2 = create_2d_plot([self.computed_func, self.reference_func],
                                plot_type=func_type,
                                title='Pressure Response',
                                x_axis='Duct Length',
                                y_axis='Pressure')

        return func_2

    def compute(self, n, *argv):

        pt_1 = PlotType(int(argv[0]))
        pt_2 = PlotType(int(argv[1]))

        x, y = self.compute_helmholtz(argv[2:])
        self.computed_func = ResultFunction('Helmholtz', x, y)
        omega = self.sim.load.omega

        x = np.linspace(0, 2, 100)
        self.reference_func = ResultFunction('Analytical', x, np.exp(-1j * omega * x), ResultFunction.COLORS[1])

        return self.update_left_plot(pt_1), self.update_right_plot(pt_2)

    @staticmethod
    def plotting_widget(custom_id):
        return html.Div([
            dbc.Select(
                id="select-{}".format(custom_id),
                options=[
                    {"label": "Pressure Real/Imaginary", "value": '1'},
                    {"label": "Pressure Magnitude/Phase", "value": '2'},
                    {"label": "Pressure Real", "value": '3'},
                    {"label": "Pressure Imaginary", "value": '4'},
                    {"label": "Pressure Magnitude", "value": '5'},
                    {"label": "Pressure Phase", "value": '6'},
                ],
                value=custom_id
            ),
            dcc.Graph(id='plot_{}'.format(custom_id))
        ]
        )

    def set_layout(self):
        parameters_geometry = dbc.Form(
            [
                html.H3("Model Parameters"),
                dbc.InputGroup(
                    [
                        dbc.InputGroupAddon("Duct Length (m)"),
                        dbc.Input(id='duct-length', type='number', value=2, min=0, step='any')
                    ], className='input-group input-group-sm'
                ),
                dbc.InputGroup(
                    [
                        dbc.InputGroupAddon("Num Elements"),
                        dbc.Input(id='num-elements', type='number', value=1, min=0)
                    ],  className='input-group input-group-sm'
                )
            ],
            className='border'
        )

        parameters_fluid = dbc.Form(
            [
                html.H3("Fluid Properties"),
                dbc.InputGroup(
                    [
                        dbc.InputGroupAddon("Mass Density (rho)"),
                        dbc.Input(id='rho', type='number', value=1.0, min=0.0, step='any')
                    ],  className='input-group input-group-sm'
                ),
                dbc.InputGroup(
                    [
                        dbc.InputGroupAddon("Speed of Sound (c)"),
                        dbc.Input(id='c', type='number', value=1, min=0, step='any')
                    ],  className='input-group input-group-sm'
                )
            ],
            className='border'
        )

        velocity_load = dbc.Form(
            [
                html.H3("Velocity Load"),
                dbc.InputGroup(
                    [
                        dbc.InputGroupAddon("Frequency", addon_type='prepend'),
                        dbc.Input(id='omega', type='number', value=12, min=0, step='any')
                    ],  className='input-group input-group-sm'
                ),
                dbc.InputGroup(
                    [
                        dbc.InputGroupAddon("Velocity", addon_type='prepend'),
                        dbc.Input(id='v', type='number', value=1, min=0, step='any')
                    ],  className='input-group input-group-sm'
                )
            ],
            className='border'
        )

        impedance_bc = dbc.Form(
            [
                html.H3("Impedance BC"),
                dbc.InputGroup(
                    [
                        dbc.InputGroupAddon("beta", addon_type='prepend'),
                        dbc.Input(id='beta', type='number', value=1, step=0.1)
                    ],  className='input-group input-group-sm'
                )
            ],
            className='border'
        )

        solve_parameters = dbc.Form(
            [
                html.H3("Solve Parameters"),
                dbc.InputGroup(
                    [
                        dbc.InputGroupAddon('pFEM'),
                        dbc.Input(id='p-fem', type='number', value=1, min=1, max=8)
                    ],  className='input-group input-group-sm'
                ),
                dbc.FormGroup(
                    [
                        dbc.Button('Compute', id='compute', block=True, color="success", outline=True)
                    ], className='input-group input-group-sm', style={'margin-top': 5}
                )
            ], className='border'
        )

        center_plots = html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(html.Div(MainDashboard.plotting_widget('1'))),
                        dbc.Col(html.Div(MainDashboard.plotting_widget('2')))
                    ]
                )
            ]
        )

        main_layout = html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(html.Div(parameters_geometry)),
                        dbc.Col(html.Div(parameters_fluid)),
                        dbc.Col(html.Div(velocity_load)),
                        dbc.Col(html.Div(impedance_bc)),
                        dbc.Col(html.Div(solve_parameters))
                    ]
                ),
                dbc.Row(
                    dbc.Col(center_plots)
                )
            ]
        )

        self.app.layout = html.Div(main_layout)
