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

    html.Div([
        # page title
        dbc.Row(dbc.Col([html.H3('Generated Data Insights')],style={'padding-bottom':'25px'},width=12)),
        # Page Content
        dbc.Row([
            dbc.Col([
                html.Div(id='data_insight',style={'margin-left': '-10px'})                
            ],width=10),
            dbc.Col([
                dbc.Row([
                    dbc.Col([ 
                        dbc.Row([
                            dbc.Col([
                                html.P('Name:',style={'display':'flex','flex-direction':'row','align-items':'center','padding-top':'20px'})
                            ]),
                                dbc.Col([
                                    dcc.Dropdown(
                                        options=[{'label':'gen1','value':'gen1'}],
                                        value='gen1',
                                        id='choose-model',
                                        clearable=False,
                                        style={'margin':'15px 0'}
                                    ),
                                ],width=8),
                        ]),
                    ])
                ]),
                dbc.Row([
                    dbc.Col([ 
                        dbc.Row([
                            dbc.Col([
                                html.P('Epoch:',style={'display':'flex','flex-direction':'row','align-items':'center','padding-top':'20px'})
                            ]),
                                dbc.Col(id="epoch_display",width=8),
                        ]),
                    ])
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Row([
                            dbc.Col([
                                html.H4('Choose a random sample:')
                            ],style={'margin-top': '40px','margin-bottom': '20px'})
                        ]),
                        dbc.Row([
                            dbc.Col([
                                dbc.Button('Random',id='random_sample',className='success')
                            ])
                        ])
                    ])
                ]),
            ],width=2)
        ]),
    ],style={'margin':'0 auto','width':'90%'}),    
])