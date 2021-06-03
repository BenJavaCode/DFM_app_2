"""
Author: Benjamin Lundquist Thomsen
Date: 03-06-2021

Description: This is the main app file, to launch the app: right click and then 'run app'.
             The content of this file, is the logic for building the app,
             and the logic for interactions, and layout components(view),
             for each page and its constituent functions.

"""

# Imports
# ---------------------------------------------------
import torch
from torch import optim
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
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
from DataProccesing.process import measure_data_lengths
from DataProccesing.process import clean_and_convert
from Datasets.SpectralDataset import spectral_dataloader

from Models.ResidualBlock import ResNet


# Init params
# ------------------------------------------------

# FOR GETTING THE USERS PATH TO THE APP DIRECTORY
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = fr'{str(dir_path)}'.replace('"', '')
# -


# Initialize Dash class that runs on Flask
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SPACELAB], suppress_callback_exceptions=True)

# Is a cuda enabled GPU available (boolean)
cuda = torch.cuda.is_available()
# assign device, cuda takes precedence
device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))


# Util Function
# ------------------------------------------------

def model_dropdown_options(model_creation=True):

    """
    model_dropdown_options(model_creation)
    Description: Updates dropdowns, that contain model names from 'model params'
    Params: model_creation = a boolean value.
            True is for model training/creation, and lets the user create a model from scratch.
            False is for model testing, and does not let the user create a new model.
    Latest update: 03-06-2021. Added more comments.
    """

    model_options = []
    if model_creation is True:
        model_options.append('Create new')
    for (dirpath, dirnames, filenames) in walk(dir_path + '\\Model_params'):
            model_options.extend([x for x in filenames if x.endswith('.pt')])
            return model_options


def load_model(model, path):

    """
    load_model(model, path)
    Description: Loads preexisting model state dictionary into model instance
    Params: model = model instance,
            path = internal path to model state dict
    Latest update: 03-06-2021. Added more comments.
    """

    dir = dir_path + "\\Model_params\\Models\\"
    model.load_state_dict(torch.load(dir+path, map_location=lambda storage, loc: storage))


# Global function variables
# ------------------------------------------------

# For storing .npy traces from rastascan, used for refinement
refinement_arr_holder = []

# Used for storing model params, model info, model state dict, when training model
model_arr = []

# Used for storing prediction map and species info, when testing model on rastascan
predictions_arr = []

# Variable used when testing model on rastascan
# For storing cleaned data and coordinates. Used when plotting point and saving point
rasta_data = []


# Utils
# -------------------------------------------------------------------

# Utility HTML error msg. Used by many functions
alert_params = html.Div([
    dbc.Row([
        dbc.Col([
            dbc.Alert(f"Something went wrong check paths/params", color="primary")
        ], width={'size': 4, 'offset': 4}, style={'padding-top': 15})
    ])
])

# HTML error msg for when trying to override file
file_already_exists = html.Div([
    dbc.Row([
        dbc.Col([
            dbc.Alert('A file with that name already exists in the Directory, choose another name', color="primary")
        ], width={'size': 4, 'offset': 4}, style={'padding-top': 15})
    ])
])

# Navigation HTML and page div
app.layout = html.Div([

    dcc.Location(id='url', refresh=False),  # Holds url location string
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
    html.Div(id='page-content')  # Holds the content of the different pages
])



@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):

    """
    display_page(pathname)
    Description: Update the page(page-content component) with page-content according to url
    Params: pathname = url pathname property. Holds current location/url. If this value changes, this function is activated
    Latest update: 03-06-2021. Added more comments.
    """

    if pathname == '/Rastascan-Analysis':
        return rastascan_analysis
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
        raise PreventUpdate  # Should consider making 404 page


# Home page
# ----------------------------------------------------------

# Homepage Html
home = html.Div([
    dbc.Row([
        dbc.Col([
            html.P("This is a Application that has the objective of letting the Scientist at DFM"
                   " conduct data-analysis, data-refinemet, dataset-creation and "
                   "Machine learning modeling and research with ease. "
                   "It is currently being developed, and this is the first demo. "
                   "If you have any questions or need to get in contact, "
                   "feel free to contact me on benjalundquist@gmail.com"),
            html.P("Best regards Benjamin Lundquist")
            ],
            width={'size': 4, 'offset': 4}
        )
    ], style={'padding-top': 150})
])


# Model testing
# -----------------------------------------------------------

# Model testing initial HTML
model_testing = html.Div([
    dbc.Row([
        dbc.Col([
            html.H5('This it the model testing page, chose model and test-set.')
        ], width={'size': 4, 'offset': 4}, style={'paddingTop': 30}),
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='model-choice-test',
                placeholder='Choose model',
                options=[{'label': k, 'value': k} for k in model_dropdown_options(model_creation=False)],
            )
        ], width={'size': 4, 'offset': 4}, style={'paddingTop': 15} )
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Input(id='lt-dataset', placeholder='Insert path to dataset')
        ], width={'size': 4, 'offset': 4}, style={'paddingTop': 10})
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Button('Start test', id='lt-start-test-button')
        ], width={'size': 4, 'offset': 4}, style={'paddingTop': 10})
    ]),
    html.Div([], id='lt-div')
])


@app.callback(
    [dash.dependencies.Output('lt-div', 'children')],
    [dash.dependencies.Input('lt-start-test-button', 'n_clicks')],
    [dash.dependencies.State('model-choice-test', 'value'),
     dash.dependencies.State('lt-dataset', 'value')]
)
def start_model_test(n_clicks, model_choice, dataset):

    """
    start_model_test(n_clicks, model_choice, dataset)
    Description: For testing model on labeled dataset, and outputting
                 result as percentage and confusion matrix.
    Params: n_clicks = lt-start-test-button click property.
                        When n_clicks changes, this function is activated
            model_choice = model-choice-test value property.
                           Holds the chosen model name.
            dataset = lt-dataset value property.
                      Holds the path to the dataset, that the user chose.
    Latest update: 03-06-2021. Added more comments.
                               Refactored variable names.
    """

    if model_choice is not None and dataset is not None:
        try:
            # LOAD DATASET DATA, LABELS AND INFO INTO VARIABLES.
            X = np.load(fr'{str(dataset)}'.replace('"', '') + '\\' + 'X.npy')
            Y = np.load(fr'{str(dataset)}'.replace('"', '') + '\\' + 'Y.npy')
            with open(fr'{str(dataset)}'.replace('"', '') + '\\' + 'info.json') as f:
                info = json.load(f)
                info_len = len(info)
            # -
        except:
            return [
                dbc.Row([
                    dbc.Col([
                        dbc.Alert('Insert viable dataset path', color='primary')
                    ], style={'paddingTop': 10}, width={'size': 4, 'offset': 4})
                ])
            ]
        # Load model initiation params
        with open(dir_path+"\\Model_params"
                  + "\\" + model_choice.replace('pt', 'json')) as f:
            params = json.load(f)

        # Instantiate model with initiation params
        cnn = ResNet(params['hidden_sizes'], params['num_blocks'], input_dim=params['input_dim'],
                     in_channels=params['in_channels'], n_classes=params['n_classes'])

        # Load model state dict into model
        cnn.load_state_dict(torch.load(
            dir_path+"\\Model_params" + "\\" + model_choice,
            map_location=lambda storage, loc: storage))  # For loading model to CPU

        # Sending the model to device
        cnn.to(device)

        # CREATE SHUFFLED IDX LIST AND CREATING DATA-LOADER
        idx = list(range(len(X)))
        np.random.shuffle(idx)
        dl_test = spectral_dataloader(X, Y, idxs=idx,
                                      batch_size=1, shuffle=False)
        # -

        # Var used by loop, to calculate accuracy
        test_size = len(dl_test.dataset)

        try:
            # Test model
            preds, labels, inputs, predict_prob, acc = test_loop2(
                cnn, dl_test, test_size, params['n_classes'], device=device)

            # CONFUSION MATRIX
            confusion_data = []
            info_data = ['_' + x for x in info.values()]  # Labels for confusion matrix. '_' is for number bug fix

            # Creating matrix-list of appropriate dimensions
            for i in range(info_len):
                confusion_data.append([0 for i in range(info_len)])

            # For filling the matrix-list with the predicted classes
            for i, v in enumerate(preds):
                confusion_data[labels[i]][v] += 1

            confusion_data.reverse()  # Reverse to get matrix, left to right
            fig = ff.create_annotated_heatmap(confusion_data, x=info_data, y=list(reversed(info_data)))  # Create matrix
            # -

            return [dbc.Row([
                        dbc.Col([dbc.Alert('Accuracy was {:.2f}%'.format(acc*100), color='primary')], style={'paddingTop': 20},
                            width={'size': 4, 'offset': 4}),
                        dbc.Col([
                            dcc.Graph(id='confusion-matrix', figure=fig)
                        ], width={'size': 6, 'offset': 3})
                        ])
                    ]
        except Exception as e:
            print(e)
            return [
                dbc.Row([
                    dbc.Col([
                        dbc.Alert('Model input dimension and datapoint dimension must match')
                    ], style={'paddingTop': 10}, width={'size': 4, 'offset': 4})
                ]),
            ]
    else:
        raise PreventUpdate


# Rastascan analysis / Model testing on unlabeled data
# ------------------------------------------------------------

# Rastascan analysis initial HTML
rastascan_analysis = html.Div([
    dbc.Row([
        dbc.Col([
            html.H5(
                'This page is for getting a prediction map from a rastascan,'
                ' for testing a model create a dataset for testing '
                'which holds none of the same data as the dataset/sets you used for training and refinement'
                ' and then go to "model testing"', style={'text-align': 'center'})
        ], width={'size': 4, 'offset': 4}, style={'paddingTop': 30}),
        dbc.Col([
            dcc.Dropdown(
                id='model-choice-test',
                placeholder='Choose model',
                options=[{'label': k, 'value': k} for k in model_dropdown_options(model_creation=False)],
            )
        ], width={'size': 4, 'offset': 4}, style={'paddingTop': 30}),
        dbc.Col([
            dbc.Input(id='rastascan-input-path', placeholder='Insert path to .csv file to '
                        'classify and plot', type='text')
        ], width={'size': 4, 'offset': 4}, style={'padding-top': 10}),
        dbc.Col([
            dbc.Button('Submit', id='start-rasta-test',)
        ], width={'size': 4, 'offset': 4}, style={'padding-top': 10})

    ], style={'padding-top': 15}),

    html.Div([], id='rastascan-div'),
    html.Div([], id='allert-rasta'),

    # Hidden function states, the observer(view_controller) observes these
    dbc.Collapse([
        dbc.Input(id='make-prediction-map-subject'),
        dbc.Input(id='rastatest-save-or-plot-subject'),
    ], is_open=False)

])


@app.callback(
    [dash.dependencies.Output('rastascan-div', 'children'),
     dash.dependencies.Output('allert-rasta', 'children')],
    [dash.dependencies.Input('make-prediction-map-subject', 'value'),
     dash.dependencies.Input('rastatest-save-or-plot-subject', 'value')]
)
def rastascan_view_controller(graph_state, choice_state):

    """
    rastascan_view_controller(graph_state, choice_state)
    Description: Rastascan observer function. It manages view for Rastascan analysis
    Params: graph_state = make-prediction-map-subject value property. 1 is success, 2 is error.
            choice_state = rastatest-save-or-plot-subject value property, 1 is save pred map,
                           2 is error, 3 is save point, other non value is plot point.
    Latest update: 03-06-2021. Added more comments.
                               Refactored variable names.
    """

    # GET INITIATOR INPUTS ID AS STRING
    ctx = dash.callback_context
    listen = ctx.triggered[0]['prop_id'].split('.')[0]  # ID of input as string
    # -

    if listen == 'make-prediction-map-subject' and graph_state is not None:
        if graph_state == 1:
            data = predictions_arr[0]  # Prediction map ad pandas data-frame
            data['index'] = data.index  # Rename index column
            species = predictions_arr[1]  # Species name: pred acc

            # Plotly figure containing prediction map
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
        elif graph_state == 2:
            return [[], alert_params]
        else:
            raise PreventUpdate
    elif listen == 'rastatest-save-or-plot-subject':
        if choice_state == 1:
            return [dash.no_update,
                dbc.Row([
                    dbc.Col([
                        dbc.Alert("Succesfully saved prediction map", color="primary")
                    ], width={'size': 2, 'offset': 5})
                ])
            ]
        elif choice_state == 2:
            return [dash.no_update, alert_params]
        elif choice_state == 3:
            return [dash.no_update,
                    dbc.Row([
                        dbc.Col([
                            dbc.Alert("Succesfully saved point", color="primary")
                        ], width={'size': 2, 'offset': 5})
                    ])
                    ]
        elif choice_state.startswith('¤'):
            data = rasta_data[0][int(choice_state[1:])][1]  # Data of selected trace
            coordinates = rasta_data[0][int(choice_state[1:])][0]  # Coordinates of selected trace
            w_len = [x for x in range(len(data))]
            df = pd.DataFrame([(x, y) for (x, y) in zip(w_len, data)],
                              columns=['counts', 'magnitude'])
            return [dash.no_update,
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(figure=px.line(df, x="counts", y="magnitude", render_mode='svg',
                                                     title=str(coordinates)), id='graph_point')
                        ], width={'size': 8, 'offset': 2}, style={'padding-top': 15})
                    ])
                    ]
        else:
            raise PreventUpdate
    else:
        raise PreventUpdate


@app.callback(
    [Output('make-prediction-map-subject', 'value')],
    [Input('start-rasta-test', 'n_clicks')],
    [dash.dependencies.State('rastascan-input-path', 'value'),
     dash.dependencies.State('model-choice-test', 'value')])
def make_prediction_map(n_clicks, rastascan_path, model_choice):

    """
    make_prediction_map(n_clicks, rastascan_path, model_choice)
    Description: Function for making prediction map from rastascan.
                 It tests a model on an unlabeled dataset, and creates global variables
                 for plotting and saving.
    Params: n_clicks = start-rasta-test n_clicks property. If n_clicks changes, this function is activated.
            rastascan_path = rastascan-input-path value property. It should be a path to a .csv file containing a rastascan.
            model_chice = model-choice-test value property. It is the chosen model for the test.
    Latest update: 03-06-2021. Added more comments.
                              Refactored variable names.
    """

    if rastascan_path is not None:
        try:
            # CLEAR GLOBAL VARS
            predictions_arr.clear()
            rasta_data.clear()
            # -

            # LOAD MODEL INIT PARAMS AND MODEL INFO INTO VARIABLES
            with open(dir_path+"\\Model_params"
                      + "\\" + model_choice.replace('pt', 'json')) as f:
                params = json.load(f)
            with open(dir_path+"\\Model_params"
                      + "\\" + model_choice.replace('.pt', '-info.json')) as f:
                species = json.load(f)
            # -

            # INITIALIZE MODEL WITH INIT PARAMS, AND LOAD MODEL STATE DICT INTO MODEL
            cnn = ResNet(params['hidden_sizes'], params['num_blocks'], input_dim=params['input_dim'],
                         in_channels=params['in_channels'], n_classes=params['n_classes'])
            cnn.load_state_dict(torch.load(
                dir_path+"\\Model_params" + "\\" + model_choice,
                map_location=lambda storage, loc: storage))
            # -

            cnn.to(device)  # Model to device
            rastascan_path = fr'{str(rastascan_path)}'  # Make string a raw string, to ignore \
            wavelen, coordinates, trace_data = cleanse_n_sort(rastascan_path.replace('"', ''))

            # CROP DATA IF NEEDED, IF SO PRINT THAT DATA WAS CROPPED
            if len(wavelen) != params['input_dim']:
                print(f"model input dimension {params['input_dim']}")
                print(f"datapoints dimesion {len(wavelen)}")
                print(f"therefore datapoints dimension was cropped to {params['input_dim']}")
                data_cleaned = clean_and_convert(trace_data, reshape_length=params['input_dim'])  # Reshape data
            else:
                data_cleaned = clean_and_convert(trace_data)
            # -

            # CREATE LABELS, IDX LIST, DATA-LOADER, DATASET SIZE VAR
            y = [0. for x in range(len(data_cleaned))]  # label-set of 0's, because its unlabeled(acc dont matter)
            idx = list(range(len(data_cleaned)))
            dl_test = spectral_dataloader(data_cleaned, y, idxs=idx,
                                          batch_size=1, shuffle=False)
            test_size = len(dl_test.dataset)  # Used in loop for calculating accuracy
            # -

            try:
                # TEST MODEL AND MAKE DATA-FRAME FOR PLOTTING
                preds, labels, inputs, preds_prob, acc = test_loop2(cnn, dl_test, test_size, params['n_classes'], device=device)
                data_frame = pd.DataFrame([(*xy, *prob, species[str(p)]) for (xy, prob, p) in zip(coordinates, preds_prob, preds)],
                                          columns=['x', 'y', *species.values(), 'Predicted Species'])
                # -
            except Exception as e:
                print(e)
                return [2]

            data_frame = data_frame.rename_axis('Index')  # Rename index column
            predictions_arr.append(data_frame)  # Dataframe to global var, used for whole pred map
            predictions_arr.append(species)  # Species dict to global var, used for whole pred map
            rasta_data.append(list(zip(coordinates, data_cleaned)))  # To global var used for individual points

            return [1]
        except:
            return [2]
    else:
        raise PreventUpdate


@app.callback(
    [dash.dependencies.Output('rastatest-save-or-plot-subject', 'value')],
    [dash.dependencies.Input('plot-point', 'n_clicks'),
     dash.dependencies.Input('save-rastascan-array', 'n_clicks'),
     dash.dependencies.Input('save-point', 'n_clicks')],
    [dash.dependencies.State('graph-rasta', 'clickData'),
     dash.dependencies.State('save-path-rastapred', 'value'),
     dash.dependencies.State('save-name-rastapred', 'value')]
)
def rastatest_save_or_plot(n_clicks, n_clicks2, n_clicks3, point, save_path, name):

    """
    rastatest_save_or_plot(n_clicks, n_clicks2, n_clicks3, point, save_path, name)
    Description: Lets you save the whole prediction-map. lets you save an individual traces original
                 data, and its prediction. Lets you plot a point
    Params: (n_clicks, n_clicks2, n_clicks3) = (plot-point, save-rastascan-array, save-point) n_clicks property.
                                                If any of these change, this function is activated.
             point = graph-rasta clickData property. It is a dict, containing some of a points values,
                     from the data-frame it belongs to.
             save_path = save-path-rastapred value property. It is the path the user wants to save the data to.
             name = save-name-rastapred value property. It is the name the user wants to give the files, they want to save.
    Latest update: 03-06-2021. Added more comments
    """

    # GET INITIATOR INPUTS ID AS STRING
    ctx = dash.callback_context
    listen = ctx.triggered[0]['prop_id'].split('.')[0]  # ID of input as string
    # -

    if listen == 'plot-point' and point is not None:
        try:
            idx = point['points'][0]['hovertext']  # hovertext contains point ID
            return [str('¤' + str(idx))]
        except:
            return [2]
    elif listen == 'save-rastascan-array' and name is not None and save_path is not None:
        try:
            predictions = predictions_arr[0]
            try:
                del predictions['index']
            except:
                pass
            predictions.to_csv(fr'{str(save_path)}'.replace('"', '') + '\\{}.csv'.format(name))  # Save prediction map
            return [1]
        except Exception as e:
            print(e)
            return [2]
    elif listen == 'save-point' and point is not None and name is not None and save_path is not None:
        try:
            idx = point['points'][0]['hovertext']  # Hovertext contains point ID
            try:
                # Get single prediction as frame.
                # iloc is for getting by idx, transpose is for making the data a column
                frame = (pd.Series.to_frame(predictions_arr[0].iloc[idx]).transpose())
                frame.index.name = 'Index'  # Rename index column
                try:
                    del frame['index']
                except:
                    pass

                # MAKE DIRECTORY
                dirname = fr'{str(save_path)}'.replace('"', '') + '\\' + str(name) + '-datapoint'
                os.mkdir(dirname)
                # -

                # SAVE POINTS PREDICTION, AND POINTS CLEANSED DATA
                frame.to_csv(dirname + '\\prediction.csv')
                with open(dirname + '\\data.csv', 'w') as f:
                    write = csv.writer(f)
                    write.writerow(list(rasta_data[0][idx][1]))
                # -

            except Exception as e:
                print(e)
                return [2]
            return [3]
        except:
            return [2]
    else:
        raise PreventUpdate


@app.callback(
     [dash.dependencies.Output('model-choice-test', 'options')],
     [dash.dependencies.Input('model-choice-test', 'value')]
)
def update_test_dropdown(model_choice):

    """
    update_test_dropdown(model_choice)
    Description: Function for updating the model choice dropdown.
    Params: model_choice = model-choice-test value property. It is the chosen model.
    Latest updated: 03-06-2021. Added comments
    """

    if model_choice is None:
        return [[{'label': k, 'value': k} for k in model_dropdown_options(model_creation=False)]]
    else:
        raise PreventUpdate


# Model training
# ------------------------------------------------------------

# Model training initial HTML
model_tr = html.Div([
    dbc.Row([
        dbc.Col(
            html.H5('This is The model training and creation page, '
                    'Choose a existing model and the data that you want to use.'
                    ' Or create a new model and save it.',
                    style={'text-align': 'center'}),
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

    # Hidden function states, the observer(view_controller) observes these
    dbc.Collapse([
        dbc.Input(id='model-customize-subject', type='number'),
        dbc.Input(id='train-model-subject', type='text'),
        dbc.Input(id='save-model-and-params-subject', type='text')
    ], is_open=False),

])

# HTML for directory path input
data_choice = html.Div([
    dbc.Row([
        dbc.Col([
            dbc.Input(id='data-choice-dataset', placeholder='Insert path to dataset directory', type='text')
        ], style={'padding-top': 10}, width={'size': 2, 'offset': 5}),
    ])
])



@app.callback(
    [dash.dependencies.Output('mt-output', 'children')],
    [dash.dependencies.Input('model-customize-subject', 'value'),
     dash.dependencies.Input('train-model-subject', 'value'),
     dash.dependencies.Input('save-model-and-params-subject', 'value')]
)
def model_training_view_controller(model_choice_state, name_and_acc_state, save_model_state):

    """
    model_training_view_controller(model_choice_state, name_and_acc_state, save_model_state)
    Description: Model training observer function. It manages view for model training
    Params: model_choice_state = model-customize-subject value property. It holds the type of model, either
                                 'Create new'=1 or a existing models name=2
            name_and_acc_state = train-model-subject value property. It holds the suggested name, for the model
                                  and the accuracy of the best model state, during training.
            save_model_state = save-model-and-params-subject value property. It holds the chosen model name.
    Latest update: 03-06-2021. Added more comments
    """

    # GET INITIATOR INPUTS ID AS STRING
    ctx = dash.callback_context
    listen = ctx.triggered[0]['prop_id'].split('.')[0]  # ID of input as string
    # -

    if listen == 'model-customize-subject':
        if model_choice_state == 1:  # 1 and 2 are duplicates, because i plan to include more options for new model
            return [[
                dbc.Row([
                    dbc.Col([
                        dbc.Input(
                            id='n-epochs-input',
                            placeholder='Number of Epochs',
                            type='number'
                        )
                    ], style={'padding-top': 10}, width={'size': 2, 'offset': 5}),
                ]),
                data_choice,
                dbc.Row([
                    dbc.Col([
                        dbc.Button('Start training', id='start-training-button', n_clicks=None)
                    ], style={'padding-top': 10}, width={'size': 2, 'offset': 5})
                ])
            ]]
        elif model_choice_state == 2:
            return [[
                dbc.Row([
                    dbc.Col([
                        dbc.Input(
                            id='n-epochs-input',
                            placeholder='Number of Epochs',
                            type='number'
                        )
                    ], style={'padding-top': 10}, width={'size': 2, 'offset': 5}),
                ]),
                data_choice,
                dbc.Row([
                    dbc.Col([
                        dbc.Button('Start training', id='start-training-button', n_clicks=None)
                    ], style={'padding-top': 10}, width={'size': 2, 'offset': 5})
                ])
            ]]
        else:
            raise PreventUpdate
    elif listen == 'train-model-subject':
        if name_and_acc_state is None:
            raise PreventUpdate
        elif name_and_acc_state == '2':
            return [alert_params]
        else:
            if 'None' in name_and_acc_state:
                s_dest = None  # if new model, suggested name is none
                split = name_and_acc_state.index('¤')
                acc = float(name_and_acc_state[split+1:])
            else:
                split = name_and_acc_state.index('¤')
                s_dest = name_and_acc_state[0:split]  # Existing model name, used as suggestion for model name
                acc = float(name_and_acc_state[split+1:])
            return [[
                dbc.Row([
                    dbc.Col([
                        dbc.Alert('Training complete, best acc was {:.2f}%'.format(acc * 100), color="primary")
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
    elif listen == 'save-model-and-params-subject':
        if save_model_state is None:
            raise PreventUpdate
        elif save_model_state != '2':
            return [
                dbc.Row([
                    dbc.Col([
                        dbc.Alert(f"succesfully saved model and associate data with primary name: {save_model_state}",
                                  color='primary')
                    ], width={'size': 2, 'offset': 5}, style={'padding-top': 10})
                ])
            ]
        else:
            return [alert_params]
    else:
        raise PreventUpdate


@app.callback(
    [dash.dependencies.Output('model-customize-subject', 'value')],
    [dash.dependencies.Input('model-choice', 'value')]
)
def customize_model(model_choice):

    """
    customize_model(model_choice)
    Description: for parsing if new or existing model was chosen
    Params: model_choice = model-choice value property. It holds the name of the chosen model,
                           either 'Create new' or a existing models name
    Latest update: 03-06-2021. Added more comments.
    """

    if model_choice == 'Create new':
        return [1]
    elif model_choice is not None:
        return [2]
    else:
        raise PreventUpdate


@app.callback(
    [dash.dependencies.Output('train-model-subject', 'value')],
    [dash.dependencies.Input('start-training-button', 'n_clicks')],
    [dash.dependencies.State('n-epochs-input', 'value'),
     dash.dependencies.State('data-choice-dataset', 'value'),
     dash.dependencies.State('model-choice', 'value')]
)
def train_model_instance(n_clicks, epochs, dataset, model_choice):

    """
    train_model_instance(n_clicks1, epochs, dataset, model_choice)
    Description: Function for initializing and loading model and then training it.
                Also stores model init params, info, and state dict in global var.
    Params: n_clicks = start-training-button n_clicks property. If this value changes, this function is activated.
            epochs = n-epochs-input value property. holds users choice for number of epochs.
            dataset = data-choice-dataset value property. It holds the path to the chosen dataset.
            model_choice = model-choice value property. It holds the type/name of the chosen model.
    Latest update: 03-06-22. Deleted model_params = None,
                             Deleted n_epochs = epochs,
                             Deleted model_params = params,
                             Added more comments.
                             Refactored variable names.
    """

    if epochs is not None and dataset is not None and model_choice is not None:
        try:

            # LOAD DATASET DATA AND LABELS, LOAD DATASET INFO
            X = np.load(fr'{str(dataset)}'.replace('"', '') + '\\' + 'X.npy')
            Y = np.load(fr'{str(dataset)}'.replace('"', '') + '\\' + 'Y.npy')
            with open(fr'{str(dataset)}'.replace('"', '') + '\\' + 'info.json') as f:
                info = json.load(f)
                info_len = len(info)  # num classes
            # -

            # CREATE NEW MODEL
            if model_choice == 'Create new':
                layers = 6
                hidden_size = 100
                block_size = 2
                hidden_sizes = [hidden_size] * layers
                num_blocks = [block_size] * layers
                input_dim = len(X[0])
                in_channels = 64
                num_classes = info_len

                # Creating var for saving model params later
                params = {'hidden_sizes': hidden_sizes, 'num_blocks': num_blocks, 'input_dim': input_dim,
                            'in_channels': in_channels, 'n_classes': num_classes}

                # Initiating model, with params
                cnn = ResNet(hidden_sizes, num_blocks, input_dim=input_dim,
                        in_channels=in_channels, n_classes=num_classes)

                s_dest = 'None'  # Part of return value. Signifies that it is a new model
            # -

            # LOAD EXISTING MODEL
            else:
                # Load chosen models init params, into variable
                with open(dir_path+"\\Model_params"
                          + "\\" + model_choice.replace('pt', 'json')) as f:
                    params = json.load(f)

                # Init model with params
                cnn = ResNet(params['hidden_sizes'], params['num_blocks'], input_dim=params['input_dim'],
                             in_channels=params['in_channels'], n_classes=params['n_classes'])

                # Load chosen model state dict into model instance
                cnn.load_state_dict(torch.load(
                    dir_path+"\\Model_params" + "\\" + model_choice,
                    map_location=lambda storage, loc: storage))  # Load state dict to CPU

                s_dest = model_choice.replace('.pt', '')  # Part of return value. Holds name of chosen model
            # -

            cnn.to(device)  # Model to device

            # MAKE TRAIN AND TEST IDX LISTS. SPLIT IS VAL 10% TRAIN 90%
            p_val = 0.1
            n_val = int(math.ceil((len(X) * p_val)))
            idx_tr = list(range(len(X)))
            np.random.shuffle(idx_tr)
            idx_val = idx_tr[:n_val]
            idx_tr = idx_tr[n_val:]
            # -

            optimizer = optim.Adam(cnn.parameters(), lr=1e-3, betas=(0.5, 0.999))  # Optim with Stanford resnet params

            # DATA-LOADERS
            dl_tr = spectral_dataloader(X, Y, idxs=idx_tr,
                                        batch_size=10, shuffle=True)
            dl_val = spectral_dataloader(X, Y, idxs=idx_val,
                                        batch_size=10, shuffle=False)
            # -

            dataloader_dict = {'train': dl_tr, 'val': dl_val}  # used by loop to both train and val

            dataset_sizes = {'train': len(dl_tr.dataset), 'val': len(dl_val.dataset)}  # Used by loop to calc accuracy

            # Train model

            model, acc = train_model(
                model=cnn, optimizer=optimizer, num_epochs=epochs,
                dl=dataloader_dict, dataset_sizes=dataset_sizes, device=device
            )

            model_arr.clear()  # Clear global var

            # STORE TO GLOBAL VAR, MODEL STATE DICT, MODEL INIT PARAMS, DATASET INFO
            model_arr.append(model)
            model_arr.append(params)
            model_arr.append(info)
            # -

            return [s_dest + '¤' + str(float(acc))]
        except Exception as e:
            print(e)
            return ['2']

    else:
        raise PreventUpdate


@app.callback(
    [dash.dependencies.Output('save-model-and-params-subject', 'value')],
    [dash.dependencies.Input('save-model-params', 'n_clicks')],
    [dash.dependencies.State('name-model', 'value'), dash.dependencies.State('model-choice', 'value')]
)
def save_model_and_params(n_clicks, name, model_choice):

    """
    save_model_and_params(n_clicks, name, model_choice)
    Description: For saving model after training.
    Params: n_clicks = save-model-params n_clicks property. If this value changes, this function is activated
            name = name-model value property. Chosen model name, model files will be given this as primary name.
            model_choice = model-choice value property. The chosen model, either 'Create new' or existing model name
    Latest update: 02-06-2021. Added more comments
    """

    if name is not None and n_clicks is not None:
        try:
            # Save model state dict
            torch.save(model_arr[0].state_dict(), dir_path+"\\Model_params" +
                       "\\" + name + ".pt")

            # IF NEW MODEL NAME
            if (name + '.pt') != model_choice:

                # Save model params
                with open(dir_path+"\\Model_params" + "\\" + name + ".json",
                          'w') as fp:
                    json.dump(model_arr[1], fp)

                # Save model info
                with open(dir_path+"\\Model_params" + "\\" + name + '-info' + ".json",
                          'w') as fp:
                    json.dump(model_arr[2], fp)

            return [name]
        except Exception as e:
            print(e)
            return ['2']
    else:
        raise PreventUpdate


@app.callback(
     [dash.dependencies.Output('model-choice', 'options'), dash.dependencies.Output('model-choice', 'value')],
     [dash.dependencies.Input('save-model-and-params-subject', 'value'),
      dash.dependencies.Input('model-choice', 'value')]
)
def update_mc_dropdown(just_saved_model, model_choice):

    """
    update_mc_dropdown(just_saved_model, model_choice)
    Description: Updates the model dropdown, when new model is created and on 'load'
    Params: just_saved_model = save-model-and-params-subject value property.
                                Used for checking, if model was created, or updated.
            model_choice = model-choice value property.
                           Used for controlling if model was created/updated
                           and for updating dropdown on 'reload'
    Latest update: 02-06-22. Added more comments.

    """

    # GET INITIATOR INPUTS ID AS STRING
    ctx = dash.callback_context
    listen = ctx.triggered[0]['prop_id'].split('.')[0]  # ID of input as string
    # -

    # If model was just saved
    if just_saved_model is not None and listen != 'model-choice':
        return [[{'label': k, 'value': k} for k in model_dropdown_options()], None]

    # If model_choice component just 'loaded' or it activated the function but was None
    elif model_choice is None:
        return [[{'label': k, 'value': k} for k in model_dropdown_options()], dash.no_update]
    else:
        raise PreventUpdate


# Data refinement
# --------------------------------------------------------

# Html for initial data-refinement
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

    # Hidden function states, the observer(view_controller) observes these
    dbc.Collapse([
        dbc.Input(id='prepare-refinement-subject', type='text'),
        dbc.Input(id='start-refinement-subject', type='text'),
        dbc.Input(id='save-refined-data-subject', type='text')
    ], is_open=False)

])


@app.callback(
    [dash.dependencies.Output('refinement-div', 'children'), dash.dependencies.Output('refinement-end', 'children'),
     dash.dependencies.Output('refinement-alerts', 'children'),
     dash.dependencies.Output('refinement-prepare', 'children')],
    [dash.dependencies.Input('prepare-refinement-subject', 'value'),
     dash.dependencies.Input('start-refinement-subject', 'value'),
     dash.dependencies.Input('save-refined-data-subject', 'value')]
)
def refinement_view_controller(prepare_refinement_state, start_refinement_state, save_refined_data_state):

    """
    refinement_view_controller(prepare_refinement_state, start_refinement_state, save_refined_data_state)
    Description: Refinement observer function. It manages view for data refinement.
    Params: prepare_refinement_state = prepare-refinement-subject value property.
                                       If success, it holds the length of each trace(amount of measurements)
                                       and the number of traces. If error it is '2'
            start_refinement_state = start-refinement-subject value property.
                                     It is either '1' for success, '2' for error.
            save_refined_data_state = save-refined-data-subject value property.
                                      It holds '0' if user did not set any traces to background or signal,
                                      signifying, that nothing was saved.
                                      It holds '1' if success. It holds '2' if error.
    Latest update: 03-06-2021. Changed code layout, so prepare refinement goes first.
                               Added more comments.
    """

    # GET INITIATOR INPUTS ID AS STRING
    ctx = dash.callback_context
    listen = ctx.triggered[0]['prop_id'].split('.')[0]  # ID of input as string
    # -

    if listen == 'prepare-refinement-subject':
        if prepare_refinement_state is None:
            raise PreventUpdate
        elif prepare_refinement_state == '2':
            return [[], [], alert_params, []]
        else:

            # SPLIT prepare-refinement-subject VALUE INTO NUMBER OF ROWS AND LENGTH OF TRACES
            split = prepare_refinement_state.index('¤')
            row_count = prepare_refinement_state[:split]
            data_len = prepare_refinement_state[split+1:]
            # -

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
                        dbc.RadioItems(  # Should be made into checkboxes instead
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

    elif listen == 'start-refinement-subject':
        if start_refinement_state is None:
            raise PreventUpdate
        elif start_refinement_state == '1':

            figures = []
            w_len = [x for x in range(len(refinement_arr_holder[0][0]))]  # len of first trace, not really wavelength

            # Append a trace plot, and radioitems for each trace
            for i in range(len(refinement_arr_holder[0])):

                df = pd.DataFrame([(x, y) for (x, y) in zip(w_len, refinement_arr_holder[0][i])], columns=['counts', 'magnitude'])

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
                                'type': 'checklist-refinement',
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
            return [[], [], alert_params, dash.no_update]

    elif listen == 'save-refined-data-subject':
        if save_refined_data_state is None:
            raise PreventUpdate
        elif save_refined_data_state == '2':
            return [dash.no_update, dash.no_update, alert_params, dash.no_update]
        elif save_refined_data_state == '1':
            return [dash.no_update, dash.no_update,
                    dbc.Row([
                        dbc.Col([
                            dbc.Alert("Succesfully saved data as npy arrays at specified path", color="primary")
                        ], width={'size': 2, 'offset': 5}, style={'padding-top': 15})
                    ]), dash.no_update
                    ]
        elif save_refined_data_state == '0':
            return [dash.no_update, dash.no_update,
                    dbc.Row([
                        dbc.Col([
                            dbc.Alert("Nothing saved for nothing is signal or background", color="primary")
                        ], width={'size': 2, 'offset': 5}, style={'padding-top': 15})
                    ]), dash.no_update
                    ]

    else:
        raise PreventUpdate


@app.callback(
    [dash.dependencies.Output('prepare-refinement-subject', 'value')],
    [dash.dependencies.Input('refinement-prepare-button', 'n_clicks')],
    [dash.dependencies.State('data-refinement-path', 'value')]
)
def prepare_refinement(n_clicks, path):

    """
    prepare_refinement(n_clicks, path)
    Description: For measuring number of rows and data length, so that user may know these.
    Params: n_clicks = refinement-prepare-button n_clicks property. If this value changes, this function is activated.
            path = data-refinement-path value property.
                   It holds the path to the rastascan, that the user wants to refine
    Latest update: 03-06-2021. Added more comments.
    """

    if path is not None:
        path = fr'{str(path)}'.replace('"', '')  # Make path rawstring to ignore \ and remove quotation marks
        try:
            row_count, data_len = measure_data_lengths(path)  # Get number of traces and length of traces.
            return [str(row_count) + '¤' + str(data_len)]
        except:
            return ['2']
    else:
        raise PreventUpdate


@app.callback(
    [dash.dependencies.Output('start-refinement-subject', 'value')],
    [dash.dependencies.Input('data-refinement-plot-button', 'n_clicks')],
    [dash.dependencies.State('data-refinement-path', 'value'),
     dash.dependencies.State('refinement-num-rows-start', 'value'),
     dash.dependencies.State('refinement-num-rows-end', 'value'),
     dash.dependencies.State('refinement-data-len', 'value'),
     dash.dependencies.State('refinement-feature-e', 'value')]
)
def start_refinement(n_clicks, path, data_start, data_end, data_len, zhang):

    """
    start_refinement(n_clicks, path, data_start, data_end, data_len, zhang)
    Description: Refines as rastascan. Lets the user determine, the slice of the rastascan,
                 that the user wants to refine. Also lets the user set what length, the data shall have
                 after refinement. Lastly, it allows the user to remove or not remove background noise.
    Params: n_clicks = data-refinement-plot-button n_clicks property. If this value changes, this function is activated
            path = data-refinement-path value property.
                   It holds the path, to the rastascan, that was given by the user.
            data_start = refinement-num-rows-start value property.
                         It holds the start of the slice, that the user wants to examine.
            data_end = refinement-num-rows-end value property.
                        It holds the end of the slice, that the user wants to examine.
            data_len = refinement-data-len value property.
                        It holds the length that the data will have after refinement, it is defined by the user
            zhang = refinement-feature-e value property. It is a boolean value, specified by the user.
                    It determines, if the traces shall have their background removed, or not.
    Latest update:  03-06-2021. Added more comments.
    """

    if path is not None and n_clicks is not None:
        try:
            path = fr'{str(path)}'.replace('"', '')  # Rawstring to remove \ and "
            wavelengths, coordinates, raw_data = cleanse_n_sort(path, slicer=(data_start, data_end))  # sorts scan

            # REMOVE BACKGROUND OR NOT
            if zhang == 1:
                zhang = True
            else:
                zhang = False
            # -

            # RESHAPE TRACE LENGTHS OR NOT
            if int(data_len) != int(len(raw_data[0])):
                reshape_len = data_len
            else:
                reshape_len = False
            # -

            try:
                # Applies background removal if chosen, reshapes if chosen, and normalises between 0-1
                data = clean_and_convert(raw_data, zhang=zhang, reshape_length=reshape_len)
            except Exception as e:
                print(e)

            refinement_arr_holder.clear()
            refinement_arr_holder.append(data)

            return ['1']
        except:
            return ['2']
    else:
        raise PreventUpdate


@app.callback(
    [dash.dependencies.Output({'type': 'checklist-refinement', 'index': dash.dependencies.ALL}, 'value')],
    [dash.dependencies.Input('refinement-signal-rest', 'n_clicks'),
     dash.dependencies.Input('refinement-background-rest', 'n_clicks'),
     dash.dependencies.Input('refinement-discard-rest', 'n_clicks')],
    [dash.dependencies.State({'type': 'checklist-refinement', 'index': dash.dependencies.ALL}, 'value')]
)
def flip_rest_refinement(signal_click, background_click, discard_click, *args):

    """
    flip_rest_refinement(signal_click, background_click, discard_click, *args)
    Description: Lets the user flip all traces that are not marked, to either signal, background or discard.
    Params: signal_click = refinement-signal-rest n_clicks property. Is for flipping rest to signal.
            background_click = refinement-background-rest n_clicks property. Is for flipping rest to background.
            discard_click = refinement-discard-rest n_clicks property. Is for flipping rest to discard.
            *args = ALL checklist-refinement components value properties.
                    Is for checking each trace, for its radioitems status.
                    If the status of a trace is none, it will be flipped, to whatever the user specified.
    Latest update: 03-06-2021. Added more comments.
    """

    # GET INITIATOR INPUTS ID AS STRING
    ctx = dash.callback_context
    listen = ctx.triggered[0]['prop_id'].split('.')[0]  # ID of input as string
    # -

    if listen == 'refinement-signal-rest':
        return [[x if x is not None else 1 for x in args[0]]]
    elif listen == 'refinement-background-rest':
        return [[x if x is not None else 2 for x in args[0]]]
    elif listen == 'refinement-discard-rest':
        return [[x if x is not None else 3 for x in args[0]]]
    else:
        raise PreventUpdate


@app.callback(
    [dash.dependencies.Output('save-refined-data-subject', 'value')],
    [dash.dependencies.Input('create-datasets', 'n_clicks')],
    [dash.dependencies.State({'type': 'checklist-refinement', 'index': dash.dependencies.ALL}, 'value'),
     dash.dependencies.State('refinement-save-path', 'value'),
     dash.dependencies.State('signal-name', 'value')]
)
def save_refined_data(n_clicks, *args):

    """
    save_refined_data(n_clicks, *args)
    Description: For sorting traces according to their mark and saving them.
    Params: n_clicks = create-datasets n_clicks value. If this value changes, this function is activated.
            *args[0] = ALL checklist-refinement components value properties.
                       These contain the chosen marks(background, signal or discard).
            *args[1] = refinement-save-path value property.
                        It is the path that the user specified, that the file/files
                        should be saved to.
            *args[2] = signal-name value property.
                        It is the primary name, that will be given to the file/files

    Latest update: 03-06-2021. Added more comments.
                               Deleted redundatn variable 'path'
    """

    signal = []
    background = []
    try:
        if args[1] is not None or args[2] is not None:
            for i, v in enumerate(args[0]):
                if v == 1:
                    signal.append(refinement_arr_holder[0][i])  # Get data from global var
                if v == 2:
                    background.append(refinement_arr_holder[0][i])
            if signal:
                filepath = fr'{str(args[1])}'.replace('"', '') + '\\' + args[2]
                np.save(filepath, np.array(signal))
            if background:
                filepath = fr'{str(args[1])}'.replace('"', '') + '\\background-' + args[2]
                np.save(filepath, np.array(background))
            if not signal and not background:  # No traces where marked as signal or background
                return ['0']
            return ['1']
        else:
            raise PreventUpdate
    except:
        return ['2']


# Dataset creation
# -------------------------------------------------------------------

# Dataset creation initial HTML
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


@app.callback(
    [dash.dependencies.Output('dataset-creation-div', 'children')],
    [dash.dependencies.Input('num-arr-dc-but', 'n_clicks')],
    [dash.dependencies.State('num-arr-dc-inp', 'value')]
)
def dc_num_arr(n_clicks, number_of_arrays):

    """
    dc_num_arr(n_clicks, number_of_arrays)
    Description: Returns the amount of array inputs, that the user specified.
    Params: n_clicks = num-arr-dc-but n_clicks property. It activates the function when changed, or context is loaded.
            number_of_arrays = num-arr-dc-inp value property. It is the number of arrays, that the user wants to combine,
                                into a dataset.
    Latest update: 03-06-2021. Added more comments.
    """

    if number_of_arrays is not None and number_of_arrays > 0:

        arr = []

        for i in range(number_of_arrays):
            arr.append(dbc.Row([
                dbc.Col([
                    dbc.Form([
                        dbc.Input(
                            id={
                                'type': 'dcc-arr-path-multi-inp',
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
                    dbc.Input(id='dc-dataset-name', placeholder='name', type='text', style={'width': '30%', 'margin-left': '2%'}),
                    dbc.Button('Save', id='dc-save-arr-but', style={'width': '10%', 'margin-left': '2%'})
                ], inline=True)
            ], width={'size': 6, 'offset': 3})
        ], style={'padding-top': 10, 'paddingBottom': 10}))
        return [arr]
    else:
        raise PreventUpdate


@app.callback(
    [dash.dependencies.Output({'type': 'dcc-array-len', 'index': dash.dependencies.ALL}, 'value')],
    [dash.dependencies.Input('calc-lengths', 'n_clicks')],
    [dash.dependencies.State({'type': 'dcc-arr-path-multi-inp', 'index': dash.dependencies.ALL}, 'value')]
)
def dc_calc_array_lengths(n_clicks, *args):

    """
    dc_calc_array_lengths(n_clicks, *args)
    Description: For calculating length of arrays, so that they can be cropped if needed.
    Params: n_clicks = calc-lengths n_clicks property. If n_clicks changes, this function is activated.
            *args = ALL dcc-arr-path-multi-inp value properties.
                    Holds the user specified paths, to the arrays,
                    that the dataset, will be build from.
    Latest update: 03-06-2021. Added more comments.
    """

    if None not in args[0]:
        lengths = []
        for index, i in enumerate(args[0]):
            temp = np.load(fr'{str(args[0][index])}'.replace('"', ''))  # Path to .npy array
            lengths.append(len(temp))  # Append the length of the array to the list
        return [lengths]
    else:
        raise PreventUpdate


@app.callback(
    [dash.dependencies.Output('dc-end', 'children')],
    [dash.dependencies.Input('dc-save-arr-but', 'n_clicks')],
    [dash.dependencies.State({'type': 'dcc-arr-path-multi-inp', 'index': dash.dependencies.ALL}, 'value'),
     dash.dependencies.State({'type': 'dcc-label-multi-inp', 'index': dash.dependencies.ALL}, 'value'),
     dash.dependencies.State({'type': 'dcc-name-class-multi-inp', 'index': dash.dependencies.ALL}, 'value'),
     dash.dependencies.State('dc-savepath', 'value'),
     dash.dependencies.State('dc-dataset-name', 'value'),
     dash.dependencies.State({'type': 'dcc-array-len', 'index': dash.dependencies.ALL}, 'value')
     ]
)
def save_arr_dc(n_clicks, *args):

    """
    save_arr_dc(n_clicks, *args)
    Description: For creating a directory and saving to it, the dataset files, X:data, Y:labels, info:dataset structure info.
    Params: n_clicks = dc-save-arr-but n_clicks property. If n_clicks changes, this function is activated.
            *args[0] = ALL dcc-arr-path-multi-inp value properties.
                       Holds the paths to the arrays that the user wants to combine into a dataset.
            *args[1] = ALL dcc-label-multi-inp value properties.
                       Holds the labels that the user has given each array.
            *args[2] = ALL dcc-name-class-multi-inp value properties
                       Holds the names that the user has given to each array/class.
            *args[3] = dc-savepath value property
                       The path that the user specified, the dataset will be saved to.
            *args[4] = dc-dataset-name value property
                       The name that the user specified, the dataset directory will have.
            *args[5] = ALL dcc-array-len value properties.
                       The amount of traces from each array, that will be used. Specified by the user.
    Latest update: 03-06-2021. Added more comments.
                               Refactored variable names.
    """

    if args[0][0] is None:
        raise PreventUpdate
    try:
        try:
            # CHECK IF ANY VALUE IN THE FIRST 3 PARAMETERS IN ARGS IS NONE, AND ALERT IF SO.
            for i in range(len(args)-3):
                if None in args[i]:
                    return [alert_params]
            # -
        except Exception as e:
            print(e)

        arr_x = []
        arr_y = []

        # LOADING AND CROPPING ARRAYS
        for index, i in enumerate(args[0]):
            temp = np.load(fr'{str(args[0][index])}'.replace('"', ''))
            if args[5][index] is not None:
                try:
                    temp = temp[:args[5][index]]  # crop array
                    arr_x.append(temp)
                except Exception as e:
                    print(e)
            else:
                arr_x.append(temp)  # Fill X list with arrays
        # -

            print(len(temp[0]))  # For internal displaying of arrays lengths

            arr_y.append(np.array([args[1][index] for v in range(len(temp))]))  # Fill Y list with labels, from u-input

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

        # CREATE DIRECTORY FOR DATASET
        dir_name = fr'{str(args[3])}'.replace('"', '') + '\\' + args[4]
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        else:
            return [file_already_exists]
        # -

        # SAVE DATA X SET, AND LABEL Y SET, AND DATASET INFO
        np.save(dir_name + '\\' + 'X', X)
        np.save(dir_name + '\\' + 'Y', Y)
        with open(dir_name + '\\' + 'info.json', 'w') as f:
            info_dict = {}
            for i, v in enumerate(args[2]):
                info_dict[args[1][i]] = v
            json.dump(info_dict, f)
        # -

        return [dbc.Row([
                    dbc.Col([
                        dbc.Alert(f"Succesfully created dataset and saved to {args[3]}", color="primary")
                    ], width={'size': 4, 'offset': 4}, style={'padding-top': 15})
                    ])
                ]
    except Exception as e:
        print(e)
        return [alert_params]


# Run App
if __name__ == "__main__":
    app.run_server(debug=True)  # debug mode, should be turned off for production



