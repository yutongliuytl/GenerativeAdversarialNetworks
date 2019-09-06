import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

from app import app
from apps import landing
from apps import training 
from apps import results

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])

def display_page(pathname):

    if pathname == '/':
        return landing.layout
    elif pathname == '/training':
        return training.layout
    elif pathname == '/results':
        return results.layout
    else:
        return '404'

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port = 5000, debug=True)
