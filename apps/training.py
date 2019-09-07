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
            dbc.Row(dbc.Col([html.H3("Training Parameters")],style={'padding-bottom':'50px','padding-top':'25px'},width=12)),
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
                        ],style={'padding-bottom':'25px'},width=8),
                    ]),
                    # choose dataset
                    dbc.Row([
                        dbc.Col([
                            html.P('Dataset:',style={'display':'flex','flex-direction':'row','align-items':'center','padding-top':'20px'})
                        ]),
                        dbc.Col([
                            dcc.Dropdown(
                                options=[{'label':'MNIST', 'value':'mnist'},
                                          {'label':'Fashion-MNIST', 'value':'fashion_mnist'},
                                          {'label':'Kuzushiji-MNIST', 'value':'kmnist'},
                                          {'label':'EMNIST', 'value':'emnist'},
                                          {'label':'QMNIST', 'value':'qmnist'}],
                                value='mnist',
                                id='choose-dataset-value-gan',
                                clearable=False,
                                style={'margin':'15px 0'}
                            ),
                        ],style={'padding-bottom':'10px'},width=8),
                    ]),

                    dbc.Row([
                        dbc.Col(html.P("Batch-Size: "),width=3,style={'display':'flex','flex-direction':'row','align-items':'center'}),
                        dbc.Col(dcc.Dropdown(
                            options=[
                                {'label':'128','value':128},
                                {'label':'256','value':256},
                                {'label':'512','value':512},
                                {'label':'1024','value':1024},
                                {'label':'2048','value':2048},
                            ],
                            value='1024',
                            id='batch-size-gan',
                            clearable=False,
                            style={'margin':'15px 0'}
                        ),style={'padding-bottom':'25px'},width=9)
                    ]),
                    # epochs
                    dbc.Row([
                        dbc.Col(html.P("Epoch: "),width=2),
                        dbc.Col(html.P(id='epoch-gan'),width=1),
                        dbc.Col(dcc.Slider(
                            id='epoch-slider-gan',
                            value=50,
                            marks={i*50: (str(i*50)) for i in range(11)},
                            max=500,
                            step=50,
                            min=0
                        ),width=9)
                    ],style={'padding-bottom':'50px'}),

                    dbc.Row([
                        dbc.Col([
                            dbc.Row([
                                dbc.Col(html.P("Generator Learning Rate:"), width=6),
                                dbc.Col(html.P(id='lr-generator'), width=2, style={'text-align':'left'}),
                                dbc.Col(width=4)
                            ]),
                            dbc.Row([
                                dbc.Col(width=1),

                                dbc.Col(html.P("Base:"),width=2),

                                dbc.Col(dcc.Input(id='lr-base-gen', type='number', value=1, max=9, min=1, style={'width':'30px', 'text-align':'center'}),width=2),

                                dbc.Col(html.P("Power:"),width=2),

                                dbc.Col(dcc.Slider(id='lr-exponent-gen', value=-3, marks={(i-10):(str(i-10)) for i in range(11)}, max=0, min=-10, step=1)),
                            ])
                        ])
                    ],style={'padding-bottom':'25px'}),

                    dbc.Row([
                        dbc.Col([
                            dbc.Row([
                                dbc.Col(html.P("Discriminator Learning Rate:"), width=6),
                                dbc.Col(html.P(id='lr-discriminator'), width=2, style={'text-align':'left'}),
                                dbc.Col(width=3)
                            ]),
                            dbc.Row([
                                dbc.Col(width=1),

                                dbc.Col(html.P("Base:"),width=2),

                                dbc.Col(dcc.Input(id='lr-base-disc', type='number', value=1, max=9, min=1, style={'width':'30px', 'text-align':'center'}),width=2),

                                dbc.Col(html.P("Power:"),width=2),

                                dbc.Col(dcc.Slider(id='lr-exponent-disc', value=-3, marks={(i-10):(str(i-10)) for i in range(11)}, max=0, min=-10, step=1)),
                            ])
                        ])
                    ], style={'margin-bottom':'30px'}),
                   
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


# ======================================================================================================================
# =================================================== CALLBACKS ========================================================
# ======================================================================================================================



@app.callback(
    Output(component_id='gan_graphs',component_property="children"),
    [Input(component_id='gan-interval',component_property='n_intervals')],
)
def show_pause_reset_buttons(n):

    d_losses,g_losses = [],[]

    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H4('Loss of Generator',style={'text-align':'center'}),
                dcc.Graph(id='gen-loss',
                    figure = {
                        'data' : [
                            {'x':list(range(len(g_losses))), 'y':g_losses, 'type':'line', 'name':'Train'},
                        ],
                        'layout' : {
                            'title' : 'Loss of the Generator per iteration.',
                            'paper_bgcolor':'rgba(0,0,0,0)',
                        }
                    },
                )
            ],width=6),
            dbc.Col([
                html.H4('Loss of Discriminator',style={'text-align':'center'}),
                dcc.Graph(id='disc-loss',
                    figure = {
                        'data' : [
                            {'x':list(range(len(d_losses))), 'y':d_losses, 'type':'line', 'name':'Train'},
                        ],
                        'layout' : {
                            'title' : 'Loss of the Discriminator per iteration.',
                            'paper_bgcolor':'rgba(0,0,0,0)',
                        }
                    },
                )
            ],width=6),
        ])
    ])