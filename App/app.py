# Imports
# ---------------------------------------------------
import torch
from torch import optim
import numpy as np
import pandas as pd
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import os
from os import walk
import json
import csv
import math

from TrainingLoops.Loops import test_loop2
from TrainingLoops.Loops import train_model
from DataProccesing.process import cleanse_n_sort
from DataProccesing.process import save_as_npy
from DataProccesing.process import condense_to_1000
from DataProccesing.process import measure_data_lenghts
from DataProccesing.process import clean_and_convert
from Datasets.SpectralDataset import spectral_dataloader

from Models.ResidualBlock import ResNet

# imports end
# ------------------------------------------------

# Init params
# ------------------------------------------------
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = fr'{str(dir_path)}'.replace('"', '')

# initialize Dash class that runs on Flask
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SPACELAB], suppress_callback_exceptions=True)

cuda = torch.cuda.is_available()  # Is a cuda enabled GPU avaliable
device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))  # assign device, cuda takes precedence

# Init params end
# ------------------------------------------------

# Util Function
# ------------------------------------------------

# Function to update dropdowns, that contain models
def model_dropdown_options(uc = True):
    model_options = []
    if uc is True:
        model_options.append('Create new')
    for (dirpath, dirnames, filenames) in walk(dir_path + '\\Model_params'):
            model_options.extend([x for x in filenames if x.endswith('.pt')])
            return model_options


# Function to load preexisting model state dictionary into model
def load_model(model, path):
    dir = dir_path + "\\Model_params\\Models\\"
    model.load_state_dict(torch.load(dir+path, map_location=lambda storage, loc:storage))

# Util Functions end
# ------------------------------------------------

# Function variables
# ------------------------------------------------

# refinement variables
numppy_arr_holder = []

# Model training Variables
model_arr = []

# Rastascan variables
predictions_arr = []
rasta_data = []

# Function variables end
# ------------------------------------------------


# Nav bar
# --------------------------------------------------------

# Navigation Html
app.layout = html.Div([

    dcc.Location(id='url', refresh=False),  # holds url location
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Rastascan Analysis", href="/Rastascan-Analysis")),
            dbc.DropdownMenu(
                children=[
                    dbc.DropdownMenuItem("More pages", header=True),
                    dbc.DropdownMenuItem("Model Training", href="/Model-Training"),
                    dbc.DropdownMenuItem("Data Refinement", href="/Data-Refinement"),
                    dbc.DropdownMenuItem("Dataset Creation", href="/Dataset-Creation"),
                    dbc.DropdownMenuItem("Model-testing", href="/Model-testing")
                ],
                nav=True,
                in_navbar=True,
                label="More",
            ),
        ],
        brand="Raman Analysis",
        brand_href="/",
        color="primary",
        dark=True,
    ),
    html.Div(id='page-content')  # holds the content of the different pages
])

# Update the page with page-content according to url
@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/Rastascan-Analysis':
        return rastascan_a
    elif pathname == '/Model-Training':
        return model_tr
    elif pathname == '/Data-Refinement':
        return data_re
    elif pathname == '/Dataset-Creation':
        return dataset_cr
    elif pathname == '/Model-testing':
        return model_testing
    elif pathname == '/':
        return home
    else:
        raise PreventUpdate  # should be page not found

# Nav bar end
# ---------------------------------------------------------

# Home page
# ----------------------------------------------------------

# Homepage Html
home = html.Div([
    dbc.Row([
        dbc.Col([
            html.P("This is a Application that has the objective of letting the Scientist at DFM conduct data-analysis, data-refinemet, dataset-creation and "
                   "Machine learning modeling and research with ease. It is currently being developed, and this is the first demo. "
                   "If you have any questions or need to get in contact, feel free to contact me on benjalundquist@gmail.com"),
            html.P("Best regards Benjamin Lundquist")
            ],
            width={'size': 4, 'offset': 4}
        )
    ], style={'padding-top': 150})
])

# Home page end
# ------------------------------------------------------------

# Model testing
# -----------------------------------------------------------

# Model testing Html
model_testing = html.Div([
    dbc.Row([
        dbc.Col([
            html.H5('This it the model testing page, chose model and test-set')
        ], width={'size': 4, 'offset': 4}, style={'paddingTop': 30}),
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Dropdown( # standardize this dropdown for 3 use cases
                id='model-choice-rasta',
                placeholder='Choose model',
                options=[{'label': k, 'value': k} for k in model_dropdown_options(uc=False)],
            )
        ], width={'size': 4, 'offset': 4},style={'paddingTop': 15} )
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Input(id='mt-dataset', placeholder='Insert path to dataset')
        ], width={'size': 4, 'offset': 4}, style={'paddingTop': 10})
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Button('Start test', id='mt-start-test-button')
        ], width={'size': 4, 'offset': 4}, style={'paddingTop': 10})
    ]),
    html.Div([], id='mt-div')
])

# Function for testing model with test data-set
@app.callback(
    [dash.dependencies.Output('mt-div', 'children')],
    [dash.dependencies.Input('mt-start-test-button', 'n_clicks')],
    [dash.dependencies.State('model-choice-rasta', 'value'),
     dash.dependencies.State('mt-dataset', 'value')]
)
def start_model_test(n_clicks, model_choice, dataset):
    if model_choice is not None and dataset is not None:
        try:
            X = np.load(fr'{str(dataset)}'.replace('"', '') + '\\' + 'X.npy')
            Y = np.load(fr'{str(dataset)}'.replace('"', '') + '\\' + 'Y.npy')
            with open(fr'{str(dataset)}'.replace('"', '') + '\\' + 'info.json') as f:
                info = json.load(f)
                info_len = len(info)
        except:
            return [
                dbc.Row([
                    dbc.Col([
                        dbc.Alert('Insert viable dataset path', color='primary')
                    ], style={'paddingTop': 10}, width={'size': 4, 'offset': 4})
                ])
            ]

        with open(dir_path+"\\Model_params"
                  + "\\" + model_choice.replace('pt', 'json')) as f:
            data = json.load(f)
        cnn = ResNet(data['hidden_sizes'], data['num_blocks'], input_dim=data['input_dim'],
                     in_channels=data['in_channels'], n_classes=data['n_classes'])
        cnn.load_state_dict(torch.load(
            dir_path+"\\Model_params" + "\\" + model_choice,
            map_location=lambda storage, loc: storage))
        idx = list(range(len(X)))
        np.random.shuffle(idx)
        dl_test = spectral_dataloader(X, Y, idxs=idx,
                                      batch_size=1, shuffle=False)
        test_size = len(dl_test.dataset)
        try:
            p, l, i, p_prob, acc = test_loop2(cnn, dl_test, test_size, data['n_classes'], device=device)
            return [dbc.Row([
                        dbc.Col([dbc.Alert('Accuracy was {:.2f}%'.format(acc*100), color='primary')], style={'paddingTop': 10},
                            width={'size': 4, 'offset': 4})
                        ])
                    ]
        except Exception as e:
            print(e)
            return [
                dbc.Row([
                    dbc.Col([
                        dbc.Alert('Model input dimension and datapoint dimension must match')
                    ], style={'paddingTop': 10}, width={'size': 4, 'offset': 4})
                ])
            ]
    else:
        raise PreventUpdate

# --------------------------------------------------------------
# Model testing end

# Rastascan plot
# ------------------------------------------------------------

# Rastascan html
rastascan_a = html.Div([
    dbc.Row([

        dbc.Col([
            html.H5(
                'This page is for getting a prediction map from a rastascan, for testing a model create a dataset for testing '
                'which holds none of the same data as the dataset/sets you used for training and refinement'
                ' and then go to "model testing"', style={'text-align': 'center'})
        ], width={'size': 4, 'offset': 4}, style={'paddingTop': 30}),

        dbc.Col([
            dcc.Dropdown(
                id='model-choice-rasta',
                placeholder='Choose model',
                options=[{'label': k, 'value': k} for k in model_dropdown_options(uc=False)],
            )
        ], width={'size': 4, 'offset': 4}, style={'paddingTop': 30}),


        dbc.Col([
            dbc.Input(id='input-on-submit', placeholder='Insert path to .csv file to '
                        'classify and plot', type='text')
        ], width={'size': 4, 'offset': 4}, style={'padding-top': 10}),

        dbc.Col([
            dbc.Button('Submit', id='submit-val',)
        ], width={'size': 4, 'offset': 4}, style={'padding-top': 10})

    ], style={'padding-top': 15}),

    html.Div([], id='rastascan-div'),
    html.Div([], id='allert-rasta'),

    dbc.Collapse([
        dbc.Input(id='rasta-graph-ic'),
        dbc.Input(id='rasta-choice-ic'),
    ], is_open=False)

])

# Rastascan Controller function
@app.callback(
    [dash.dependencies.Output('rastascan-div', 'children'), dash.dependencies.Output('allert-rasta', 'children')],
    [dash.dependencies.Input('rasta-graph-ic', 'value'), dash.dependencies.Input('rasta-choice-ic', 'value')]
)
def rastascan_controller(value1, value2):
    ctx = dash.callback_context
    listen = ctx.triggered[0]['prop_id'].split('.')[0]  # class-name of the component that triggered the func
    if listen == 'rasta-graph-ic' and value1 is not None:
        if value1 == 1:
            data = predictions_arr[0]
            data['index'] = data.index  # rename index column
            species = predictions_arr[1]
            fig = px.scatter(data, x='x', y='y', height=600, color='Predicted Species',
                             hover_data=species.values(), hover_name='index')
            return [
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='graph-rasta', figure=fig)
                    ], width={'size': 6, 'offset': 3}, style={'padding-top': 20}),
                    dbc.Col([
                        html.H5('To plot a individual datapoint(trace) click on a datapoint and then "Plot point". '
                                'Insert a path to save to and Press "Save all" to save all predictions')
                    ], width={'size': 4, 'offset': 4}, style={'padding-top': 15}),

                    dbc.Col([
                        dbc.FormGroup([
                            dbc.Input(id='save-path-rastapred', placeholder='Save path'),
                            dbc.Input(id='save-name-rastapred', placeholder='Name')
                        ], inline=True)
                    ], width={'size': 4, 'offset': 4}, style={'padding-top': 10}),

                    dbc.Col([
                        dbc.FormGroup([
                            dbc.Button('Plot point', id='plot-point', style={'width': '30%'}),
                            dbc.Button('Save point', id='save-point', style={'width': '30%', 'margin-left': '5%'},
                                       ),
                            dbc.Button('Save all', id='save-rastascan-array',
                                       style={'width': '30%', 'margin-left': '5%'})
                        ], inline=True)
                    ], width={'size': 4, 'offset': 4}, style={'padding-top': 10, 'padding-bottom': 15}),

                ]), []]
        elif value1 == 2:
            return [[], allert_params]
        else:
            raise PreventUpdate
    elif listen == 'rasta-choice-ic':
        if value2 == 1:
            return [dash.no_update,
                dbc.Row([
                    dbc.Col([
                        dbc.Alert("Succesfully saved prediction map", color="primary")
                    ], width={'size': 2, 'offset': 5})
                ])
            ]
        elif value2 == 2:
            return [dash.no_update, allert_params]

        elif value2 == 3:
            return [dash.no_update,
                    dbc.Row([
                        dbc.Col([
                            dbc.Alert("Succesfully saved point", color="primary")
                        ], width={'size': 2, 'offset': 5})
                    ])
                    ]

        elif value2.startswith('¤'):

            d = rasta_data[0][int(value2[1:])][1]
            coordinates = rasta_data[0][int(value2[1:])][0]
            w_len = [x for x in range(len(d))]
            df = pd.DataFrame([(x, y) for (x, y) in zip(w_len, d)],
                              columns=['counts', 'magnitude'])

            return [dash.no_update,
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(figure=px.line(df, x="counts", y="magnitude", render_mode='svg', title=str(coordinates)), id='graph_point')
                        ], width={'size': 8, 'offset': 2}, style={'padding-top': 15})
                    ])
                    ]
        else:
            raise PreventUpdate
    else:
        raise PreventUpdate


# Function for making prediction map from rastascan
@app.callback(
    [Output('rasta-graph-ic', 'value')],
    [Input('submit-val', 'n_clicks')],
    [dash.dependencies.State('input-on-submit', 'value'), dash.dependencies.State('model-choice-rasta', 'value')])
def update_output(n_clicks, value, path):
    if value is not None:
        try:
            predictions_arr.clear()
            rasta_data.clear()
            with open(dir_path+"\\Model_params"
                      + "\\" + path.replace('pt', 'json')) as f:
                data = json.load(f)
            with open(dir_path+"\\Model_params"
                      + "\\" + path.replace('.pt', '-info.json')) as f:
                species = json.load(f)
            cnn = ResNet(data['hidden_sizes'], data['num_blocks'], input_dim=data['input_dim'],
                         in_channels=data['in_channels'], n_classes=data['n_classes'])
            cnn.load_state_dict(torch.load(
                dir_path+"\\Model_params" + "\\" + path,
                map_location=lambda storage, loc: storage))
            if cuda: cnn.cuda()
            value = fr'{str(value)}'
            w, c, d = cleanse_n_sort(value.replace('"', ''))

            if len(w) != data['input_dim']:
                print(f"model input dimension {data['input_dim']}")
                print(f"datapoints dimesion {len(w)}")
                print(f"therefore datapoints dimension was cropped to {data['input_dim']}")
                data_cleaned = clean_and_convert(d, reshape_length=data['input_dim'])
            else:
                data_cleaned = clean_and_convert(d)
            y = [0. for x in range(len(data_cleaned))]
            idx = list(range(len(data_cleaned)))
            dl_test = spectral_dataloader(data_cleaned, y, idxs=idx,
                                          batch_size=1, shuffle=False)
            test_size = len(dl_test.dataset)
            try:
                p, l, i, p_prob, acc = test_loop2(cnn, dl_test, test_size, data['n_classes'], device=device)
                data = pd.DataFrame([(*xy, *prob, species[str(p)]) for (xy, prob, p) in zip(c, p_prob, p)],
                                    columns=['x', 'y', *species.values(), 'Predicted Species'])
            except Exception as e:
                print(e)
                return [2]
            data = data.rename_axis('Index')
            predictions_arr.append(data)
            predictions_arr.append(species)
            rasta_data.append(list(zip(c, data_cleaned)))
            return [1]
        except:
            return [2]
    else:
        raise PreventUpdate

# listening function for plotting point, saving point and saving whole prediction map
@app.callback(
    [dash.dependencies.Output('rasta-choice-ic', 'value')],
    [dash.dependencies.Input('plot-point', 'n_clicks'),
     dash.dependencies.Input('save-rastascan-array', 'n_clicks'),
     dash.dependencies.Input('save-point', 'n_clicks')],
    [dash.dependencies.State('graph-rasta', 'clickData'), dash.dependencies.State('save-path-rastapred', 'value'),
     dash.dependencies.State('save-name-rastapred', 'value')]
)
def save_rastainfo(n_clicks, n_clicks2, n_clicks3, data, path, name):

    ctx = dash.callback_context
    listen = ctx.triggered[0]['prop_id'].split('.')[0]

    if listen == 'plot-point' and data is not None:
        try:
            idx = data['points'][0]['hovertext']
            return [str('¤' + str(idx))]
        except:
            return [2]
    elif listen == 'save-rastascan-array' and name is not None and path is not None:
        try:
            predictions = predictions_arr[0]
            try:
                del predictions['index']
            except:
                pass
            predictions.to_csv(fr'{str(path)}'.replace('"', '') + '\\{}.csv'.format(name))
            return [1]
        except Exception as e:
            print(e)
            return [2]
    elif listen == 'save-point' and data is not None and name is not None and path is not None:
        try:
            idx = data['points'][0]['hovertext']
            try:
                frame = (pd.Series.to_frame(predictions_arr[0].iloc[idx]).transpose())
                frame.index.name = 'Index'
                try:
                    del frame['index']
                except:
                    pass
                dirname = fr'{str(path)}'.replace('"', '') + '\\' + str(name) + '-datapoint'
                os.mkdir(dirname)
                frame.to_csv(dirname + '\\prediction.csv')
                with open(dirname + '\\data.csv', 'w') as f:
                    write = csv.writer(f)
                    write.writerow(list(rasta_data[0][idx][1]))

            except Exception as e:
                print(e)
                return [2]
            return [3]
        except:
            return [2]
    else:
        raise PreventUpdate


# function for updating the model choice dropdown
@app.callback(
     [dash.dependencies.Output('model-choice-rasta', 'options')],
     [dash.dependencies.Input('model-choice-rasta', 'value')]
)
def update_resta_dropdown(value):
    if value is None:
        return [[{'label': k, 'value': k} for k in model_dropdown_options(uc=False)]]
    else:
        raise PreventUpdate


# Rastascan plot end
# ------------------------------------------------------------


# Model training
# ------------------------------------------------------------

# model training html
model_tr = html.Div([
    dbc.Row([
        dbc.Col(
            html.H5('This is The model training and creation page, Choose a existing model and the data that you want to use.'
            ' Or create a new model and save it.', style={'text-align': 'center'}),
            width={'size': 4, 'offset': 4}

         )
    ], style={'padding': '40px'}),

    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='model-choice',
                placeholder='Choose model, or create',
                options=[{'label': k, 'value': k} for k in model_dropdown_options()],
            ),
        ], width={'size': 2, 'offset': 5}),

    ]),
    html.Div(id='mt-output'),

    dbc.Collapse([
        dbc.Input(id='more-options-proc', type='number'),
        dbc.Input(id='start-training-proc', type='text'),
        dbc.Input(id='save-model-proc', type='text')
    ], is_open=False),

])

# html for directory path
data_choice = html.Div([
    dbc.Row([
        dbc.Col([
            dbc.Input(id='data-choice-dataset', placeholder='Insert path to dataset directory', type='text')
        ], style={'padding-top': 10}, width={'size': 2, 'offset': 5}),
    ])
])

# model training controller function
@app.callback(
    [dash.dependencies.Output('mt-output', 'children')],
    [dash.dependencies.Input('more-options-proc', 'value'),
     dash.dependencies.Input('start-training-proc', 'value'),
     dash.dependencies.Input('save-model-proc', 'value')]
)
def model_training_controller(value1, value2, value3):
    ctx = dash.callback_context
    listen = ctx.triggered[0]['prop_id'].split('.')[0]

    if listen == 'more-options-proc':
        if value1 == 1:  # 1 and 2 are duplicates, because i plan to include more options for new model
            return [[
                dbc.Row([
                    dbc.Col([
                        dbc.Input(
                            id='n_epochs_input',
                            placeholder='Number of Epochs',
                            type='number'
                        )
                    ], style={'padding-top': 10}, width={'size': 2, 'offset': 5}),
                ]),
                data_choice,
                dbc.Row([
                    dbc.Col([
                        dbc.Button('Start training', id='start_training_button', n_clicks=None)
                    ], style={'padding-top': 10}, width={'size': 2, 'offset': 5})
                ])
            ]]
        elif value1 == 2:
            return [[
                dbc.Row([
                    dbc.Col([
                        dbc.Input(
                            id='n_epochs_input',
                            placeholder='Number of Epochs',
                            type='number'
                        )
                    ], style={'padding-top': 10}, width={'size': 2, 'offset': 5}),
                ]),
                data_choice,
                dbc.Row([
                    dbc.Col([
                        dbc.Button('Start training', id='start_training_button', n_clicks=None)
                    ], style={'padding-top': 10}, width={'size': 2, 'offset': 5})
                ])
            ]]
        else:
            raise PreventUpdate
    elif listen == 'start-training-proc':
        if value2 is None:
            raise PreventUpdate

        elif value2 == '2':
            return [allert_params]
        else:
            if 'None' in value2:
                s_dest = None  # if new model, name is none
                split = value2.index('¤')
                acc = float(value2[split+1: -1])
            else:
                split = value2.index('¤')
                s_dest = value2[0:split]  # existing model name
                acc = float(value2[split+1:])

            return [[
                dbc.Row([
                    dbc.Col([
                        dbc.Alert('Training complete, best acc was {:.4f}%'.format(acc * 100), color="primary")
                    ], width={'size': 2, 'offset': 5}, style={'padding-top': 10}),
                    dbc.Col([
                        dbc.Form([
                            dbc.Input(id='name-model', placeholder='Name of model', type='text',
                                      style={'width': '60%'}, value=s_dest),
                            dbc.Button('save', id='save-model-params',
                                       style={'width': '38%', 'margin-left': '2%'})
                        ], inline=True)
                    ], width={'size': 2, 'offset': 5})
                ])
            ]]
    elif listen == 'save-model-proc':
        if value3 is None:
            raise PreventUpdate
        elif value3 != '2':
            return [
                dbc.Row([
                    dbc.Col([
                        dbc.Alert(f"succesfully saved model and associate data with primary name: {value3}",
                                  color='primary')
                    ], width={'size': 2, 'offset': 5}, style={'padding-top': 10})
                ])
            ]
        else:
            return [allert_params]
    else:
        raise PreventUpdate


# function for passing if new or existing model was chosen
@app.callback(
    [dash.dependencies.Output('more-options-proc', 'value')],
    [dash.dependencies.Input('model-choice', 'value')]
)
def model_customize(model_choice):
    if model_choice == 'Create new':
        return [1]
    elif model_choice is not None:
        return [2]
    else:
        raise PreventUpdate


# Function for initializing and loading model and then training it
@app.callback(
    [dash.dependencies.Output('start-training-proc', 'value')],
    [dash.dependencies.Input('start_training_button', 'n_clicks')],
    [dash.dependencies.State('n_epochs_input', 'value'),
     dash.dependencies.State('data-choice-dataset', 'value'),
     dash.dependencies.State('model-choice', 'value')]
)
def choose_data_nm(n_clicks1, epochs, dataset, model_choice):

    if epochs is not None and dataset is not None and model_choice is not None:
        model_params = None
        try:
            X = np.load(fr'{str(dataset)}'.replace('"', '') + '\\' + 'X.npy')
            Y = np.load(fr'{str(dataset)}'.replace('"', '') + '\\' + 'Y.npy')
            with open(fr'{str(dataset)}'.replace('"', '') + '\\' + 'info.json') as f:
                info = json.load(f)
                info_len = len(info)  # num classes
            n_epochs = epochs
            if model_choice == 'Create new':  # New Model
                layers = 6
                hidden_size = 100
                block_size = 2
                hidden_sizes = [hidden_size] * layers
                num_blocks = [block_size] * layers
                input_dim = len(X[0])
                in_channels = 64
                num_classes = info_len
                model_params = {'hidden_sizes': hidden_sizes, 'num_blocks': num_blocks, 'input_dim': input_dim,
                            'in_channels': in_channels, 'n_classes': num_classes}

                cnn = ResNet(hidden_sizes, num_blocks, input_dim=input_dim,
                        in_channels=in_channels, n_classes=num_classes)
                s_dest = 'None'
            else:  # load existing model
                with open(dir_path+"\\Model_params"
                          + "\\" + model_choice.replace('pt', 'json')) as f:
                    data = json.load(f)
                cnn = ResNet(data['hidden_sizes'], data['num_blocks'], input_dim=data['input_dim'],
                             in_channels=data['in_channels'], n_classes=data['n_classes'])
                model_params = data
                cnn.load_state_dict(torch.load(
                    dir_path+"\\Model_params" + "\\" + model_choice,
                    map_location=lambda storage, loc: storage))
                s_dest = model_choice.replace('.pt', '')

            if cuda: cnn.cuda()
            p_val = 0.1
            n_val = int(math.ceil((len(X) * p_val)))
            idx_tr = list(range(len(X)))
            np.random.shuffle(idx_tr)
            idx_val = idx_tr[:n_val]
            idx_tr = idx_tr[n_val:]

            optimizer = optim.Adam(cnn.parameters(), lr=1e-3, betas=(0.5, 0.999))

            dl_tr = spectral_dataloader(X, Y, idxs=idx_tr,
                                        batch_size=10, shuffle=True)
            dl_val = spectral_dataloader(X, Y, idxs=idx_val,
                                        batch_size=10, shuffle=False)

            dataloader_dict = {'train': dl_tr, 'val': dl_val}

            dataset_sizes = {'train': len(dl_tr.dataset), 'val': len(dl_val.dataset)}

            model, acc = train_model(
                model=cnn, optimizer=optimizer, num_epochs=n_epochs,
                dl=dataloader_dict, dataset_sizes=dataset_sizes, device=device
            )
            model_arr.clear()
            model_arr.append(model)
            model_arr.append(model_params)
            model_arr.append(info)

            return [s_dest + '¤' + str(float(acc))]
        except Exception as e:
            print(e)
            return ['2']

    else:
        raise PreventUpdate

# Function for saving model
@app.callback(
    [dash.dependencies.Output('save-model-proc', 'value')],
    [dash.dependencies.Input('save-model-params', 'n_clicks')],
    [dash.dependencies.State('name-model', 'value'), dash.dependencies.State('model-choice', 'value')]
)
def save_model_and_params(n_clicks, name, model_choice):
    if name is not None and n_clicks is not None:
        try:
            torch.save(model_arr[0].state_dict(), dir_path+"\\Model_params" +
                       "\\" + name + ".pt")
            if (name + '.pt') != model_choice:  # if new model name
                with open(dir_path+"\\Model_params" + "\\" + name + ".json",
                          'w') as fp:
                    json.dump(model_arr[1], fp)
                with open(dir_path+"\\Model_params" + "\\" + name + '-info' + ".json",
                          'w') as fp:
                    json.dump(model_arr[2], fp)
            return [name]
        except Exception as e:
            print(e)
            return ['2']
    else:
        raise PreventUpdate


# update model choice dropdown
@app.callback(
     [dash.dependencies.Output('model-choice', 'options'), dash.dependencies.Output('model-choice', 'value')],
     [dash.dependencies.Input('save-model-proc', 'value'), dash.dependencies.Input('model-choice', 'value')]
)
def update_mc_dropdown(value1, value2):
    ctx = dash.callback_context
    listen = ctx.triggered[0]['prop_id'].split('.')[0]
    if value1 is not None and listen != 'model-choice':
        return [[{'label': k, 'value': k} for k in model_dropdown_options()], None]
    elif value2 is None:
        return [[{'label': k, 'value': k} for k in model_dropdown_options()], dash.no_update]
    else:
        raise PreventUpdate


# Model training end
# --------------------------------------------------------

# Data refinement
# --------------------------------------------------------

# Html for data refinement
data_re = html.Div([
    dbc.Row([
        dbc.Col(
            html.H5(
                'This is the data refinement page, here you can upload a .csv file'
                ' And refine it into .npy arrays and save them for use in dataset creation.'
                ' For the stanford resnet model the length of the datapoints/traces needs to a multiple of 5. '
                'Furthermore, when later creating a dataset all npy arrays datapoints need to have the same length'
                , style={'text-align': 'center'}
            ),
            width={'size': 4, 'offset': 4}
        )
    ], style={'padding-top': '50px'}
    ),
    dbc.Row([
        dbc.Col(
            dbc.Input(id='data-refinement-path', placeholder='Insert path to .csv file', type='text'),
            width={'size': 2, 'offset': 5}, style={'padding-top': 15}
        ),
        dbc.Col(
            dbc.Button('Apply', id='refinement-prepare-button')
            , width={'size': 2, 'offset': 5}, style={'padding-top': 10}
        )
    ]),
    html.Div([
    ], id='refinement-prepare'),

    html.Div([
    ], id='refinement-div', style={'padding-top': 40}),

    html.Div([
    ], id='refinement-end', style={'padding-top': 15}),

    html.Div([
    ], id='refinement-alerts', style={'padding-top': 15}),

    dbc.Collapse([
        dbc.Input(id='refinement-prepare-ic', type='text'),
        dbc.Input(id='refinement-ic1', type='text'),
        dbc.Input(id='refinement-ic2', type='text')
    ], is_open=False)

])

# Refinement controller function
@app.callback(
    [dash.dependencies.Output('refinement-div', 'children'), dash.dependencies.Output('refinement-end', 'children'),
     dash.dependencies.Output('refinement-alerts', 'children'),
     dash.dependencies.Output('refinement-prepare', 'children')],
    [dash.dependencies.Input('refinement-ic1', 'value'), dash.dependencies.Input('refinement-ic2', 'value'),
     dash.dependencies.Input('refinement-prepare-ic', 'value')]
)
def refinement_controller(value, value2, value3):
    ctx = dash.callback_context
    listen = ctx.triggered[0]['prop_id'].split('.')[0]
    if listen == 'refinement-ic1':
        if value is None:
            raise PreventUpdate
        elif value == '1':
            figures = []
            w_len = [x for x in range(len(numppy_arr_holder[0][0]))]
            for i in range(len(numppy_arr_holder[0])):

                df = pd.DataFrame([(x, y) for (x, y) in zip(w_len, numppy_arr_holder[0][i])], columns=['counts', 'magnitude'])

                figures.append(dbc.Row([
                    dbc.Col([
                        dcc.Graph(figure=px.line(df, x="counts", y="magnitude", render_mode='svg'), id='graph-{}'.format(i))
                    ], width={'size': 8, 'offset': 2}, style={'padding-top': 15}),
                    dbc.Col([
                        dbc.RadioItems(
                            options=[
                                {"label": "Signal", "value": 1},
                                {"label": "Background", "value": 2},
                                {"label": "Discard", "value": 3}
                            ],
                            id={
                                'type': 'checklist_refinement',
                                'index': i
                            },
                            inline=True,
                            switch=True

                        )
                    ], width={'size': 7, 'offset': 3})
                ])
                )
            figures.append(dbc.Row([
                dbc.Col(
                    dbc.Form([
                        dbc.Button('Signal rest', id='refinement-signal-rest'),
                        dbc.Button('Background rest', id='refinement-background-rest', style={'margin-left': 4}),
                        dbc.Button('Discard rest', id='refinement-discard-rest', style={'margin-left': 4})
                        ],
                        inline=True
                    ),
                    width={'size': 8, 'offset': 2}
                )
            ], style={'padding-top': 30}))
            figures.append((dbc.Row([dbc.Col(
                dbc.Input(id='refinement-save-path', placeholder='directory, to save to', type='text'),
                style={'padding-top': 10}, width={'size': 8, 'offset': 2})])))
            figures.append(dbc.Row([dbc.Col(dbc.Input(id='signal-name', placeholder='Name of signal file', type='text'),
                                            style={'padding-top': 10}, width={'size': 8, 'offset': 2})]))
            figures.append(dbc.Row([dbc.Col(dbc.Button('Save Arrays', id='create-datasets'), style={'padding': 15},
                                            width={'size': 8, 'offset': 2})]))
            return [figures, [], [], dash.no_update]
        else:
            return [[], [], allert_params, dash.no_update]
    if listen == 'refinement-ic2':
        if value2 is None:
            raise PreventUpdate
        elif value2 == '2':
            return [dash.no_update, dash.no_update, allert_params, dash.no_update]
        elif value2 == '1':
            return [dash.no_update, dash.no_update,
                    dbc.Row([
                        dbc.Col([
                            dbc.Alert("Succesfully saved data as npy arrays at specified path", color="primary")
                        ], width={'size': 2, 'offset': 5}, style={'padding-top': 15})
                    ]), dash.no_update
                    ]
        elif value2 == '0':
            return [dash.no_update, dash.no_update,
                    dbc.Row([
                        dbc.Col([
                            dbc.Alert("Nothing saved for nothing is signal or background", color="primary")
                        ], width={'size': 2, 'offset': 5}, style={'padding-top': 15})
                    ]), dash.no_update
                    ]
    elif listen == 'refinement-prepare-ic':
        if value3 is None:
            raise PreventUpdate
        elif value3 == '2':
            return [[], [], allert_params, []]
        else:
            split = value3.index('¤')
            row_count = value3[:split]
            data_len = value3[split+1:]
            return [
                [], [], [],
                [
                    dbc.Col([
                        dbc.Label('Desired data length', html_for='refinement-data-len'),
                        dbc.Input(id='refinement-data-len', placeholder='Crop to this length', type='number',
                                  value=data_len),
                        ], width={'size': 2, 'offset': 5}, style={'padding-top': 10}),
                    dbc.Col([
                        dbc.Form([
                            dbc.Label('examine start - end', html_for='refinement-num-rows'),
                            dbc.Input(id='refinement-num-rows-start', placeholder='Number of rows to examine', type='number',
                                      value=0, style={'width': '50%'}),
                            dbc.Input(id='refinement-num-rows-end', placeholder='Number of rows to examine', type='number',
                                      value=int(row_count), style={'width': '50%'}),

                        ], inline=True)
                    ], width={'size': 2, 'offset': 5}, style={'padding-top': 10}),
                    dbc.Col(
                        dbc.Button('Submit and plot', id='data-refinement-plot-button')
                        , width={'size': 2, 'offset': 5}, style={'padding-top': 10}
                    ),
                    dbc.Col(
                        dbc.RadioItems( # Shall be made into checkboxes instead
                            options=[
                                {"label": "Apply Zhangfit", "value": 1},
                                {"label": "Dont apply Zhangfit", "value": 2}
                            ],
                            id='refinement-feature-e',
                            inline=True,
                            switch=True,
                            value=1
                        ), width={'size': 2, 'offset': 5}, style={'padding-top': 10}
                    )

                ]
            ]

    else:
        raise PreventUpdate


# function for measuring number of rows and data length
@app.callback(
    [dash.dependencies.Output('refinement-prepare-ic', 'value')],
    [dash.dependencies.Input('refinement-prepare-button', 'n_clicks')],
    [dash.dependencies.State('data-refinement-path', 'value')]
)
def prepare_refinement(n_clicks, path):
    if path is not None:
        path = fr'{str(path)}'.replace('"', '')
        try:
            row_count, data_len = measure_data_lenghts(path)
            return [str(row_count) + '¤' + str(data_len)]
        except:
            return ['2']
    else:
        raise PreventUpdate


# Function for refining rastascan
@app.callback(
    [dash.dependencies.Output('refinement-ic1', 'value')],
    [dash.dependencies.Input('data-refinement-plot-button', 'n_clicks')],
    [dash.dependencies.State('data-refinement-path', 'value'),
     dash.dependencies.State('refinement-num-rows-start', 'value'),
     dash.dependencies.State('refinement-num-rows-end', 'value'),
     dash.dependencies.State('refinement-data-len', 'value'),
     dash.dependencies.State('refinement-feature-e', 'value')]
)
def start_refinement(n_clicks, path, data_start, data_end, data_len, zhang):
    if path is not None and n_clicks is not None:
        try:
            path = fr'{str(path)}'.replace('"', '')
            w, c, d = cleanse_n_sort(path, slicer=(data_start, data_end))
            if zhang == 1:
                zhang = True
            else:
                zhang = False
            if int(data_len) != int(len(d[0])):
                reshape_len = data_len
            else:
                reshape_len = False
            try:
                data = clean_and_convert(d, zhang=zhang, reshape_length=reshape_len)
            except Exception as e:
                print(e)
            numppy_arr_holder.clear()
            numppy_arr_holder.append(data)
            return ['1']
        except:
            return ['2']
    else:
        raise PreventUpdate


# Function for flipping unsigned traces to desired mode
@app.callback(
    [dash.dependencies.Output({'type': 'checklist_refinement', 'index': dash.dependencies.ALL}, 'value')],
    [dash.dependencies.Input('refinement-signal-rest', 'n_clicks'),
     dash.dependencies.Input('refinement-background-rest', 'n_clicks'),
     dash.dependencies.Input('refinement-discard-rest', 'n_clicks')],
    [dash.dependencies.State({'type': 'checklist_refinement', 'index': dash.dependencies.ALL}, 'value')]
)
def check_rest_refinement(signal_click, background_click, discard_click, *args):
    ctx = dash.callback_context
    listen = ctx.triggered[0]['prop_id'].split('.')[0]
    if listen == 'refinement-signal-rest':
        return [[x if x is not None else 1 for x in args[0]]]
    elif listen == 'refinement-background-rest':
        return [[x if x is not None else 2 for x in args[0]]]
    elif listen == 'refinement-discard-rest':
        return [[x if x is not None else 3 for x in args[0]]]
    else:
        raise PreventUpdate


# function for sorting traces according to their mark and saving them
@app.callback(
    [dash.dependencies.Output('refinement-ic2', 'value')],
    [dash.dependencies.Input('create-datasets', 'n_clicks')],
    [dash.dependencies.State({'type': 'checklist_refinement', 'index': dash.dependencies.ALL}, 'value'),
     dash.dependencies.State('refinement-save-path', 'value'), dash.dependencies.State('signal-name', 'value')]
)
def create_npy_arr(n_clicks, *args):
    signal = []
    background = []
    try:
        if args[1] is not None or args[2] is not None:
            path = fr'{str()}'.replace('"', '')
            for i, v in enumerate(args[0]):
                if v == 1:
                    signal.append(numppy_arr_holder[0][i])
                if v == 2:
                    background.append(numppy_arr_holder[0][i])
            if signal:
                filepath = fr'{str(args[1])}'.replace('"', '') + '\\' + args[2]
                np.save(filepath, np.array(signal))
            if background:
                filepath = fr'{str(args[1])}'.replace('"', '') + '\\background-' + args[2]
                np.save(filepath, np.array(background))
            if not signal and not background:
                return ['0']
            return ['1']
        else:
            raise PreventUpdate
    except:
        return ['2']


# Data refinement end
# ------------------------------------------------------------------


# Dataset creation
# -------------------------------------------------------------------

# Dataset creation Html
dataset_cr = html.Div([
    dbc.Row([
        dbc.Col([
            html.H5(
                'This is the dataset creation page, here you can specify npy arrays you want to merge into a dataset'
                ' and save the dataset for model training or model testing. '
                'Ideally the total length of a class(sum of lengths of arrays used for a class) '
                ' should be the same for each class, to avoid a biased model.'
                ' Furthermore the label numbers shall start from 0 and be sequential. '
                'eg(0,1,2) for classes x,k,l or eg(0,0,1,2,2) for classes x,k,l. '
                'Finally the labels and names should corespond, eg for labels (0,0,1,2) '
                'valid names could be (ecoli, ecoli, epidermis, background)'
                , style={'text-align': 'center'}
            )
        ], width={'size': 4, 'offset': 4})
    ], style={'padding-top': '50px'}),
    dbc.Row([
        dbc.Col([
            dbc.Form([
                dbc.Input(id='num-arr-dc-inp', placeholder='Number of arrays', type='number', style={'width': '80%'}),
                dbc.Button('Apply', id='num-arr-dc-but', style={'width': '18%', 'margin-left': '2%'}),
            ], inline=True)
        ], style={'padding-top': 15}, width={'size': 4, 'offset': 4})
    ]),
    html.Div([], id='dataset-creation-div'),
    html.Div([], id='dc-end')
])


# Function for returned the right amount of array inputs
@app.callback(
    [dash.dependencies.Output('dataset-creation-div', 'children')],
    [dash.dependencies.Input('num-arr-dc-but', 'n_clicks')],
    [dash.dependencies.State('num-arr-dc-inp', 'value')]
)
def dc_num_arr(n_clicks, value):
    if value is not None:
        arr = []
        for i in range(value):
            arr.append(dbc.Row([
                dbc.Col([
                    dbc.Form([
                        dbc.Input(
                            id={
                                'type': 'dcc-arr-multi-inp',
                                'index': i
                            },
                            placeholder='insert path to array',
                            type='text',
                            style={'width': '25%'}
                        ),
                        dbc.Input(
                            id={
                                'type': 'dcc-label-multi-inp',
                                'index': i
                            },
                            placeholder='label of arr/class',
                            type='number',
                            style={'margin-left': '2%', 'width': '8%'},
                            value=i
                        ),
                        dbc.Input(
                            id={
                                'type': 'dcc-name-class-multi-inp',
                                'index': i
                            },
                            placeholder='Name of bioelement',
                            type='text',
                            style={'margin-left': '2%', 'width': '25%'}

                        ),
                        dbc.Input(
                            id={
                                'type': 'dcc-array-len',
                                'index': i
                            },
                            type='number',
                            placeholder='press calc-len to calculate len',
                            style={'margin-left': '2%', 'width': '36%'}

                        )
                    ], inline=True)


                ], style={'padding-top': 10}, width={'size': 6, 'offset': 3})

            ])
            )
        arr.append(dbc.Row([
            dbc.Col([
                dbc.Form([
                    dbc.Button('Calc-len', id='calc-lengths', style={'width': '14%'}),
                    dbc.Input(id='dc-savepath', placeholder='directory to save array', type='text',
                              style={'width': '40%', 'marginLeft': '2%'}),
                    dbc.Input(id='dc-name-arr', placeholder='name', type='text', style={'width': '30%', 'margin-left': '2%'}),
                    dbc.Button('Save', id='dc-save-arr-but', style={'width': '10%', 'margin-left': '2%'})
                ], inline=True)
            ], width={'size': 6, 'offset': 3})
        ], style={'padding-top': 10, 'paddingBottom': 10}))
        return [arr]
    else:
        raise PreventUpdate

# Function for calculating length of arrays, so that they can be cropped if needed
@app.callback(
    [dash.dependencies.Output({'type': 'dcc-array-len', 'index': dash.dependencies.ALL}, 'value')],
    [dash.dependencies.Input('calc-lengths', 'n_clicks')],
    [dash.dependencies.State({'type': 'dcc-arr-multi-inp', 'index': dash.dependencies.ALL}, 'value')]
)
def calc_lengths(n_clicks, *args):
    if None not in args[0]:
        lengths = []
        for index, i in enumerate(args[0]):
            temp = np.load(fr'{str(args[0][index])}'.replace('"', ''))
            lengths.append(len(temp))
        return [lengths]
    else:
        raise PreventUpdate


# Function for creating dataset
@app.callback(
    [dash.dependencies.Output('dc-end', 'children')],
    [dash.dependencies.Input('dc-save-arr-but', 'n_clicks')],
    [dash.dependencies.State({'type': 'dcc-arr-multi-inp', 'index': dash.dependencies.ALL}, 'value'),
     dash.dependencies.State({'type': 'dcc-label-multi-inp', 'index': dash.dependencies.ALL}, 'value'),
     dash.dependencies.State({'type': 'dcc-name-class-multi-inp', 'index': dash.dependencies.ALL}, 'value'),
     dash.dependencies.State('dc-savepath', 'value'),
     dash.dependencies.State('dc-name-arr', 'value'),
     dash.dependencies.State({'type': 'dcc-array-len', 'index': dash.dependencies.ALL}, 'value')
     ]
)
def save_arr_dc(n_clicks, *args):
    if args[0][0] is None:
        raise PreventUpdate
    try:
        try:
            for i in range(len(args)-3):
                if None in args[i]:
                    return [allert_params]
        except Exception as e:
            print(e)
        arr_x = []
        arr_y = []
        for index, i in enumerate(args[0]):  # loading and gathering arrays
            temp = np.load(fr'{str(args[0][index])}'.replace('"', ''))
            if args[5][index] is not None:
                try:
                    temp = temp[:args[5][index]]  # crop array
                    arr_x.append(temp)
                except Exception as e:
                    print(e)
            else:
                arr_x.append(temp)
            print(len(temp[0]))
            arr_y.append(np.array([args[1][index] for v in range(len(temp))]))
        try:
            X = np.vstack([v for v in arr_x])  # Stack arrays in sequence vertically (row wise).
            Y = np.hstack([v for v in arr_y])  # Stack arrays in sequence horizontally (column wise).
        except Exception as e:
            print(e)
            return [
                dbc.Row([
                    dbc.Col([
                        dbc.Alert('All the input arrays must have data of the same shape', color="primary")
                    ], width={'size': 4, 'offset': 4}, style={'padding-top': 15})
                ])
            ]
        dirName = fr'{str(args[3])}'.replace('"', '') + '\\' + args[4]
        if not os.path.exists(dirName):
            os.mkdir(dirName)
        else:
            return [file_already_exists]
        np.save(dirName + '\\' + 'X', X)
        np.save(dirName + '\\' + 'Y', Y)
        with open(dirName + '\\' + 'info.json', 'w') as f:
            info_dict = {}
            for i, v in enumerate(args[2]):
                info_dict[args[1][i]] = v
            json.dump(info_dict, f)


        return [dbc.Row([
                    dbc.Col([
                        dbc.Alert(f"Succesfully crated dataset and saved to {args[3]}", color="primary")
                    ], width={'size': 4, 'offset': 4}, style={'padding-top': 15})
                    ])
                ]
    except:
        return [allert_params]

# Dataset creation end
# -------------------------------------------------------------------

# Utils
# -------------------------------------------------------------------

# Utility html error msg for many function
allert_params = html.Div([
    dbc.Row([
        dbc.Col([
            dbc.Alert(f"Something went wrong check paths/params", color="primary")
        ], width={'size': 4, 'offset': 4}, style={'padding-top': 15})
    ])
])

# Html error msg for when trying to override file
file_already_exists = html.Div([
    dbc.Row([
        dbc.Col([
            dbc.Alert('A file with that name already exists in the Directory, choose another name', color="primary")
        ], width={'size': 4, 'offset': 4}, style={'padding-top': 15})
    ])
])

# Utils end
# ---------------------------------------------------------------------


# Run App
if __name__ == "__main__":
    app.run_server(debug=True)  # debug mode, should be turned off for production



