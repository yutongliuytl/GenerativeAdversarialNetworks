import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from app import app

import boto3

s3 = boto3.resource("s3")
client_s3 = boto3.client('s3')
client_db = boto3.client('dynamodb')


layout = html.Div([

    # navbar
    dbc.Navbar(
        dbc.Container([

            dbc.NavbarToggler(id="navbar-toggler"),
            dbc.Collapse(
                dbc.Nav([
                    dbc.NavItem(dbc.NavLink("Home", href="/")),
                    dbc.NavItem(dbc.NavLink("Training", href="/training")),
                    dbc.NavItem(dbc.NavLink("Results", href="/results"), active=True),
                ], navbar=True),
                id="navbar-collapse",
                navbar=True,
            ),
        ],style={'max-width':'98%'}),
        color="light",
    ),

    html.Div([
        # page title
        dbc.Row(dbc.Col([html.H3('Generated Data Insights')],style={'padding':'35px 0 75px 0','margin-left':'-20px'},width=12)),
        # Page Content
        dbc.Row([
            dbc.Col([
                html.Div([],id='data_insight',style={'margin-left': '-10px'})                
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


# ======================================================================================================================
# =================================================== CALLBACKS ========================================================
# ======================================================================================================================
@app.callback(
    Output(component_id='data_insight', component_property='children'),
    [Input(component_id='random_sample',component_property='n_clicks'),],
    # [State(component_id='choose-dataset-value',component_property='value')]
)
def return_dataframe_random(n):
    name_of_file = "default_gan"
    img = client_s3.get_object(Bucket="gan-dashboard", Key="generated-images/{0}.jpeg".format(name_of_file))

    return dbc.Container([
            dbc.Col(html.Img(src="https://gan-dashboard.s3.amazonaws.com/generated-images/default_gan.jpeg"))
        ])


# @app.callback(
#     Output(component_id='epoch_display',component_property='children'),
#     [Input(component_id='choose-dataset-value',component_property='value')]
# )
# def return_epoch_list(dataset):

#     epochs = wgan_index[dataset]

#     return dcc.Dropdown(
#             options=epochs,
#             value=0,
#             id='choose-epoch',
#             clearable=False,
#             style={'margin':'15px 0'}
#         )

