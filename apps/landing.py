import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from app import app


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

    dbc.Container([

        html.H1("Generative Adversarial Network Dashboard Homepage", style={'padding':'1%', 'textAlign': 'center','margin-top':'100px'}),

        dbc.Row([
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        html.H6('This dashboard allows users to modify the parameters of a chosen Generative Adversarial Network (GAN) architecture and train it on a chosen dataset. The user can then view the results generated.'),
                    ])
                ]),
            ]),

        ], style={'padding-bottom':'15px', 'padding-top':'15px', 'padding-left':'25px', 'margin':'25px 0px', 'textAlign': 'center'}),


        dbc.Row([
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        html.H3('Training'),
                    ])
                ]),
                dbc.Row([
                    dbc.Col([
                        html.H6('Train a GAN architecture'),
                    ])
                ]),
            ]),

        ], style={'background':'#B9E6E8', 'border-radius':'10px', 'padding-bottom':'15px', 'padding-top':'15px', 'padding-left':'25px', 'margin':'25px 0px', 'text-align':'center'}),


        dbc.Row([
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        html.H3('Results'),
                    ])
                ]),
                dbc.Row([
                    dbc.Col([
                        html.H6('View the generated images'),
                    ])
                ]),
            ]),

        ], style={'background':'#D3CBF5','border-radius':'10px', 'padding-bottom':'15px', 'padding-top':'15px', 'padding-left':'25px', 'margin':'25px 0px', 'text-align':'center'}),

    ]),
], style={'height':'100vh'}),