# Dash Imports
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from app import app

#AWS Imports
import boto3

#ganlib Imports
from .ganlib.processing import Processing
from .ganlib.dcgan import DCGAN

import json

s3 = boto3.resource("s3")
client_s3 = boto3.client('s3')
client_db = boto3.client('dynamodb')


# ======================================================================================================================
# =================================================== COMPONENTS =======================================================
# ======================================================================================================================

layout = html.Div([

    # navbar
    dbc.Navbar(
        dbc.Container([

            dbc.NavbarToggler(id="navbar-toggler"),
            dbc.Collapse(
                dbc.Nav([
                    dbc.NavItem(dbc.NavLink("Home", href="/")),
                    dbc.NavItem(dbc.NavLink("Training", href="/training"), active=True),
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
            dbc.Row(dbc.Col([html.H3("Training Parameters")],style={'padding-bottom':'30px','padding-top':'10px'},width=12)),
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
                                {'label':'100','value':100},
                                {'label':'200','value':200},
                                {'label':'500','value':500},
                                {'label':'1000','value':1000},
                            ],
                            value=100,
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
                            ], style={'margin-bottom':'10px'}),
                            dbc.Row([
                                dbc.Col(width=1),

                                dbc.Col(html.P("Base:"),width=2),

                                dbc.Col(dcc.Input(id='lr-base-gen', type='number', value=1, max=9, min=1, style={'width':'30px', 'text-align':'center'}),width=2),

                                dbc.Col(html.P("Power:"),width=2),

                                dbc.Col(dcc.Slider(id='lr-exponent-gen', value=-4, marks={(i-10):(str(i-10)) for i in range(11)}, max=0, min=-10, step=1)),
                            ])
                        ])
                    ],style={'padding-bottom':'25px'}),

                    dbc.Row([
                        dbc.Col([
                            dbc.Row([
                                dbc.Col(html.P("Discriminator Learning Rate:"), width=8),
                                dbc.Col(html.P(id='lr-discriminator'), width=2, style={'text-align':'left'}),
                                dbc.Col(width=1)
                            ], style={'margin-bottom':'10px'}),
                            dbc.Row([
                                dbc.Col(width=1),

                                dbc.Col(html.P("Base:"),width=2),

                                dbc.Col(dcc.Input(id='lr-base-disc', type='number', value=1, max=9, min=1, style={'width':'30px', 'text-align':'center'}),width=2),

                                dbc.Col(html.P("Power:"),width=2),

                                dbc.Col(dcc.Slider(id='lr-exponent-disc', value=-4, marks={(i-10):(str(i-10)) for i in range(11)}, max=0, min=-10, step=1)),
                            ])
                        ])
                    ], style={'margin-bottom':'30px'}),
                   
                    # buttons
                    dbc.Row([
                        dbc.Col([
                            dbc.Button("Train Model", id='train-gan-btn', outline=True, color="success", className="mr-1")
                        ],style={'text-align':'center'},id='train-gan'),
                    ],style={'padding':'10px 0 0 0'})
                ],width=4),
                # graphs output
                dbc.Col(id='gan-graphs',style={'padding':'0 15px'})
            ])
        ],style={'padding':'25px 50px 40px 50px'}),
    ]),
    # intervals
    html.Div([
        dcc.Interval(
            id='gan-interval',
            interval=10000*1, # in milliseconds
            n_intervals=0
        )
    ]),
    dcc.ConfirmDialog(
        id='start-gan',
        message='GAN Parameters Submitted.\nStarted Training. Please be patient as the training takes some time.',
                 
    ),
    dcc.ConfirmDialog(
        id='finish-gan',
        message='Training is Completed!\nView Results on the next tab.',
                 
    ),
    html.Div(id='feature_columns',style={'display':'none'})
])


# ======================================================================================================================
# =================================================== CALLBACKS ========================================================
# ======================================================================================================================

@app.callback(
    [Output(component_id='start-gan',component_property='displayed'),
    Output(component_id='train-gan-btn',component_property='disabled')],
    [Input(component_id='train-gan-btn',component_property='n_clicks')]
)
def show_training_start(n):
    return (True, True)


@app.callback(
    Output(component_id='gan-graphs',component_property="children"),
    [Input(component_id='gan-interval',component_property='n_intervals'),
    Input(component_id='name-gan',component_property='value')],
)
def show_graphs(n,name):

    try:
        losses = client_s3.get_object(Bucket="gan-dashboard", Key="loss/{0}.txt".format(name))["Body"].read()
        losses = json.loads(losses)
        g_losses,d_losses = losses["g_loss"],losses["d_loss"]

    except:
        g_losses,d_losses = [],[]

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


@app.callback(
    Output(component_id='finish-gan',component_property='displayed'),
    [Input(component_id='train-gan-btn',component_property='n_clicks')],
    #wgan options
    [State(component_id='name-gan',component_property='value'),
    State(component_id='choose-dataset-value-gan',component_property='value'),
    State(component_id='batch-size-gan',component_property='value'),
    State(component_id='epoch-slider-gan',component_property='value'),
    State(component_id='lr-base-gen',component_property='value'),
    State(component_id='lr-exponent-gen',component_property='value'),
    State(component_id='lr-base-disc',component_property='value'),
    State(component_id='lr-exponent-disc',component_property='value')]
)

def gan_training(n_clicks,gan_name,dataset,batch_size,max_epoch,lr_base_gen,lr_exp_gen,lr_base_disc,lr_exp_disc):

    #Preprocessing
    process = Processing()
    train_loader,_ = process.fit_transform(dataset,batch_size)

    #Training
    model = DCGAN(gan_name,z_dim=100,dataset=process.train_dataset,lr_g=lr_base_gen*pow(10,lr_exp_gen),lr_d=lr_base_disc*pow(10,lr_exp_disc))
    D_losses,G_losses = model.fit(train_loader,max_epoch,batch_size)
    
    #Saving Models and Generated Images
    model.save_model(gan_name)

    return True