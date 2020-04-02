import dash
import dash_bootstrap_components as dbc
import dash_auth


# Keep this out of source code repository - save in a file or a database
VALID_USERNAME_PASSWORD_PAIRS = {
    'hello': 'world'
}

app = dash.Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP])

auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS
)

server = app.server
app.config.suppress_callback_exceptions = True
app.title = 'GAN Dashboard'