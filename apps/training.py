import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from app import app

import boto3

client = boto3.client('dynamodb')

layout = html.Div([

    # navbar
    dbc.Navbar(
        dbc.Container([

            dbc.NavbarToggler(id="navbar-toggler"),
            dbc.Collapse(
                dbc.Nav([
                    dbc.NavItem(dbc.NavLink("Home", href="/"), active=True),
                    dbc.NavItem(dbc.NavLink("Training", href="/training")),
                    dbc.NavItem(dbc.NavLink("Results", href="/results")),
                ], navbar=True),
                id="navbar-collapse",
                navbar=True,
            ),
        ],style={'max-width':'98%'}),
        color="light",
    ),

    dbc.Row([
        dbc.Col([
            # page title
            dbc.Row(dbc.Col([html.H3("Training Parameters")],style={'padding-bottom':'25px'},width=12)),
            #page content
            dbc.Row([
                # training parameters
                dbc.Col([
                    # dataset name
                    dbc.Row([
                        dbc.Col(
                            html.P("Name of GAN:")
                        ),
                        dbc.Col([
                            dcc.Input(type='text',value='default_gan',id='name-gan',required=True,style={'width':'100%'})
                        ],width=8),
                    ]),
                    # choose dataset
                    dbc.Row([
                        dbc.Col([
                            html.P('Dataset:',style={'display':'flex','flex-direction':'row','align-items':'center','padding-top':'20px'})
                        ]),
                        dbc.Col([
                            dcc.Dropdown(
                                options=[{'label':'MNIST', 'value':'mnist'}],
                                value='mnist',
                                id='choose-dataset-value-gan',
                                clearable=False,
                                style={'margin':'15px 0'}
                            ),
                        ],width=8),
                    ]),
                    # gen model
                    dbc.Row([
                        dbc.Col([html.P("Generator Model:")],width=5,style={'display':'flex','flex-direction':'row','align-items':'center','padding-top':'13px'}),
                        dbc.Col([
                            dcc.Dropdown(
                                options=[
                                    {'label':'Multi Layer Perceptron','value':'mlp'}
                                ],
                                value='mlp',
                                id='ml-model-gen',
                                clearable=False,
                                style={'margin':'15px 0'}
                            ),
                        ],width=7)
                    ],style={'display':'none'}),
                    # disc model
                    dbc.Row([
                        dbc.Col([html.P("Discriminator Model:")],width=5,style={'display':'flex','flex-direction':'row','align-items':'center','padding-top':'13px'}),
                        dbc.Col([
                            dcc.Dropdown(
                                options=[
                                    {'label':'Multi Layer Perceptron','value':'mlp'}
                                ],
                                value='mlp',
                                id='ml-model-disc',
                                clearable=False,
                                style={'margin':'15px 0'}
                            ),
                        ],width=7)
                    ],style={'display':'none'}),
                    # mlp selections
                    dbc.Row([
                        dbc.Col([
                            html.Div(id="mlp-selection-gan"),
                        ])
                    ]),
                    # buttons
                    dbc.Row([
                        dbc.Col([
                            dbc.Button("Train", id='train-gan-btn', outline=True, color="success", className="mr-1")
                        ],style={'text-align':'center'},id='train-gan'),
                    ],style={'padding':'0 0 0 0'})
                ],width=4),
                # graphs output
                dbc.Col(id='gan_graphs',style={'margin-top':'75px'})
            ])
        ],style={'padding':'30px','margin-top':'-20px'}),
    ]),
    # intervals
    html.Div([
        dcc.Interval(
            id='gan-interval',
            interval=1000*1, # in milliseconds
            n_intervals=0
        )
    ]),
    dcc.ConfirmDialog(
        id='training_gan',
        message='GAN PARAMETERS SUBMITTED',
                 
    ),
    html.Div(id='feature_columns',style={'display':'none'})
])