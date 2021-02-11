import os #For reading all files in cache
import pickle #For saving fetched data
import time #For benchmarking performance
import pandas as pd #Dataframe manipulation
import numpy as np #Matrix operations
import plotly_express as px #plotting
from plotly.offline import plot #Offline plotting

import plotly.graph_objects as go
import dash #For creating plotly dashboards
import dash_core_components as dcc #Higher-level components that are interactive and are generated with JavaScript, HTML, and CSS through the React.js library
import dash_html_components as html #For converting python into HTML
from dash.dependencies import Input, Output #For interacting components
import json #Temporary json dumps

cache_dir = 'cache'

#Directories to read
sub_dir_defaults = cache_dir + '/defaults'
sub_dir_alts = cache_dir + '/alts'

#Directories for writing
pokemon_csv = 'pokemon.csv'
names_per_type_csv = 'names_per_type.csv'
type_stacked_csv = 'type_stacked.csv'
type_square_csv = 'type_square.csv'

#The poke-api replaces spaces with hypens, but some pokemon genuinely have hyphens.
pokemon_with_hyphens = ['hakamo-o',
                       'ho-oh',
                       'jangmo-o',
                       'kommo-o',
                       'porygon-z']

#Some pokemon genuinely have spaces in their names.
pokemon_with_spaces = ['mr mime',
                       'mime jr',
                       'mr rime',
                       'type null',
                       'tapu koko',
                       'tapu lele',
                       'tapu bulu',
                       'tapu fini']

#Traditional pokemon type ordering
type_order = ['Normal',
              'Fire',
              'Water',
              'Grass',
              'Electric',
              'Ice',
              'Fighting',
              'Poison',
              'Ground',
              'Flying',
              'Psychic',
              'Bug',
              'Rock',
              'Ghost',
              'Dragon',
              'Dark',
              'Steel',
              'Fairy']

#Read data from  local cache
def read_data(df, is_default_indicator, pickle_directory):

    for filename in os.listdir(pickle_directory):
        pokemon = pickle.load(open(pickle_directory+'/'+filename, 'rb'))
        name = pokemon['name']
        #A form may not be guarenteed to have a type
        Type1 = pokemon['types'][0 ]['type']['name'] if len(pokemon['types']) > 0 else None
        Type2 = pokemon['types'][1]['type']['name'] if len(pokemon['types']) > 1 else None

        image = pokemon['name']+'.png'

        pokedex_number = int(pokemon['species']['url'].split('/')[-2])
        pokeapi_number = int(pokemon['id'])

        df.loc[len(df)] =[name,Type1,Type2,is_default_indicator,image,pokedex_number,pokeapi_number]

#Create type effectiveness square matrix
def generate_type_effectiveness_df(dict_types, game_type):

    #'po' for normal pokemon games
    if(game_type == 'po'):
        super_effective = 2
        not_very_effective = 0.5
        no_effect = 0
    elif(game_type == 'go'):
        super_effective = 1.6
        not_very_effective = 1/1.6
        no_effect = 1/1.6*1/1.6
    else:
        raise Exception('ERROR! Unrecognized game_type: ' + game_type)

    #Starting matrix for the normal pokemon games
    df_effectiveness = pd.DataFrame(data = 1.0, columns=list(dict_types.keys()),index = list(dict_types.keys()))

    for attack_type in list(dict_types.keys()):
      for defense_type in dict_types[attack_type]['damage_relations']['double_damage_to']:
          df_effectiveness[defense_type['name']][attack_type] = df_effectiveness[defense_type['name']][attack_type] * super_effective

      for defense_type in dict_types[attack_type]['damage_relations']['half_damage_to']:
          df_effectiveness[defense_type['name']][attack_type] = df_effectiveness[defense_type['name']][attack_type] * not_very_effective

      for defense_type in dict_types[attack_type]['damage_relations']['no_damage_to']:
          df_effectiveness[defense_type['name']][attack_type] = df_effectiveness[defense_type['name']][attack_type] * no_effect

    df_effectiveness.columns = [name.capitalize() for name in df_effectiveness.columns]
    df_effectiveness.index = [name.capitalize() for name in df_effectiveness.index]
    return df_effectiveness

#Create stacked effectivenss
def generate_effectiveness_stacked_df(df_effectiveness, game_type):
    #'po' for normal pokemon games
    if(game_type == 'po'):
        immune_value = 0
        resistantx2_value = 0.25
        resistant_value = 0.5
        normal_value = 1.0
        effective_value = 2.0
        effectivex2_value = 4.0
    elif(game_type == 'go'):
        immune_value = 1/1.6/1.6/1.6
        resistantx2_value = 1/1.6/1.6
        resistant_value = 1/1.6
        normal_value = 1.0
        effective_value = 1.6
        effectivex2_value = 1.6*1.6
    else:
        raise Exception('ERROR! Unrecognized game_type: ' + game_type)

    effectiveness_names = ['Immune','2x Resistant','Resistant','Effective','Super Effective','2x Super Effective']
    df_effectiveness_stacked = pd.DataFrame(columns=['Type 1','Type 2']+effectiveness_names,)
    for type_name1 in type_order:
        for type_name2 in type_order:
            immune = []
            resistantx2 = []
            resistant = []
            normal = []
            effective = []
            effectivex2 = []
            if type_name1 != type_name2:
                type_combined = df_effectiveness[type_name1] * df_effectiveness[type_name2]
            else:
                type_combined = df_effectiveness[type_name1] #Monotyping can't double up on itself.

            #Place the attacking type into the appropriate effectiveness bucket
            for type_attacking in type_order:
                if type_combined[type_attacking] == immune_value:
                    immune.append(type_attacking)
                elif type_combined[type_attacking] == resistantx2_value:
                    resistantx2.append(type_attacking)
                elif type_combined[type_attacking] == resistant_value:
                    resistant.append(type_attacking)
                elif type_combined[type_attacking] == effective_value:
                    effective.append(type_attacking)
                elif type_combined[type_attacking] == effectivex2_value:
                    effectivex2.append(type_attacking)
                elif type_combined[type_attacking] == normal_value:
                    normal.append(type_attacking)
                else:
                    raise Exception('ERROR! Unrecognized effectiveness: ' + str(type_combined[type_attacking]))

            #Change formatting to be read by d3 later
            max_length = max(len(immune),len(resistantx2),len(resistant),len(normal),len(effective),len(effectivex2))
            if len(immune) < max_length:
                immune.extend([""]*(max_length-len(immune)))
            if len(resistantx2) < max_length:
                resistantx2.extend([""]*(max_length-len(resistantx2)))
            if len(resistant) < max_length:
                resistant.extend([""]*(max_length-len(resistant)))
            if len(normal) < max_length:
                normal.extend([""]*(max_length-len(normal)))
            if len(effective) < max_length:
                effective.extend([""]*(max_length-len(effective)))
            if len(effectivex2) < max_length:
                effectivex2.extend([""]*(max_length-len(effectivex2)))

            df_effectiveness_stacked = df_effectiveness_stacked.append({'Type 1':type_name1, 'Type 2':type_name2, 'Immune':immune, '2x Resistant':resistantx2, 'Resistant':resistant, 'Effective':normal, 'Super Effective':effective, '2x Super Effective':effectivex2},ignore_index=True)

    #Change formatting to be read by d3 later
    for name in effectiveness_names:
        df_effectiveness_stacked[name] = df_effectiveness_stacked[name].apply(lambda x: '\t'.join(x))

    return df_effectiveness_stacked

#Create type count square matrix
def generate_type_square(df_pokemon, type_order):

    #Count type permutations and convert into a square matrx.
    df_type_count = pd.crosstab(df_pokemon['Type 1'],df_pokemon['Type 2'])

    #Bulbapedia ordering
    #type_order = list(dict_types.keys())

    df_type_count = df_type_count[type_order].reindex(type_order)

    #The order of the typing in pokemon doesn't really matter, so we'll add the upper and lower squares.
    upper = df_type_count.where(np.triu(np.ones(df_type_count.shape)).astype(np.bool)).fillna(0)

    #Exclude the diagonal so we don't double count
    lower = df_type_count.where(np.tril(np.ones(df_type_count.shape),-1).astype(np.bool)).transpose().fillna(0)

    #Also have to flip axis names from the transpose, so we can add the lower to the upper
    lower = lower.rename_axis(lower.columns.name).rename_axis(lower.index.name, axis = 1)
    df_type_square = (upper + lower).where(np.triu(np.ones(df_type_count.shape)).astype(np.bool)).iloc[:,::-1] #iloc here reverses column order

    #Convert double to int
    df_type_square = df_type_square.astype('Int64')

    return df_type_square

#Create type matrix in a stacked form
def generate_type_stacked(df_type_square,type_order):
    #Stack the data for better reading from d3
    df_type_stacked = df_type_square.stack().to_frame(name='Count').reindex(pd.MultiIndex.from_product([type_order,type_order[::-1]],names=['Type 1', 'Type 2'])).reset_index()
    #Drop rows where Count is NA
    df_type_stacked = df_type_stacked[df_type_stacked['Count'].notna()]
    #Duplicate non-mono type rows, so we have a symmetric matrix.
    df_type_stacked = df_type_stacked[df_type_stacked['Type 1'] != df_type_stacked['Type 2']].append(df_type_stacked[df_type_stacked['Type 1']==df_type_stacked['Type 2']]).append(pd.DataFrame(df_type_stacked[df_type_stacked['Type 1'] != df_type_stacked['Type 2']][['Type 2','Type 1','Count']].values,columns=['Type 1','Type 2','Count']))
    return df_type_stacked

#Find pokemon belonging to each typing.
def generate_list_per_type(df_pokemon,column_name):
    names_per_type = df_pokemon.groupby(['Type 1','Type 2'])[column_name].apply(list)
    #Convert multiindex to columns
    names_per_type = names_per_type.reset_index()
    #We need to duplicate some rows and reverse the type ordering, due to type order not mattering, and we want to ensure the table will return the same list regardless of the order queried.
    names_per_type = names_per_type[names_per_type['Type 1'] == names_per_type['Type 2']].append(names_per_type[names_per_type['Type 1'] != names_per_type['Type 2']]).append(pd.DataFrame(names_per_type[names_per_type['Type 1'] != names_per_type['Type 2']][['Type 2','Type 1',column_name]].values,columns=['Type 1','Type 2',column_name]))

    #Change formatting to be read by d3 later
    names_per_type[column_name] = names_per_type[column_name].apply(lambda x: '\t'.join(x))

    return names_per_type

def make_plots(df_type_square, df_pokemon):

    #plot(px.imshow(df_type_square))
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

    #app = dash.Dash("PokeGraph", external_stylesheets=external_stylesheets)
    app = dash.Dash("PokeGraph")

    fig_type = px.imshow(df_type_square)

    #fig_pokemon = px.(df_pokemon)

    app.layout = html.Div(children=[

        html.Div([
            html.H1(children='PokeGraph'),
        ]),

        html.Div(children='''
            Type distributions
        '''),

        html.Div([
            dcc.Graph(
                id='graph-type',
                figure=fig_type
            ),
        ]),

        html.Div([
            dcc.Graph(
                id='graph-pokemon'
                #figure=fig_pokemon
            ),
        ]),

    ])

    @app.callback(
        Output('click-data', 'children'),
        Input('basic-interactions', 'clickData'))
    def display_click_data(clickData):
        return json.dumps(clickData, indent=2)

    if __name__ == '__main__':
        app.run_server(debug=False)


if __name__ == '__main__':

    start_time = time.time()

    #Extract type dictionary
    dict_types = pickle.load(open(cache_dir+'/dict_types.pkl', 'rb'))

    #Create effectiveness charts
    game_type = 'po'
    df_effectiveness_po = generate_type_effectiveness_df(dict_types, game_type)
    df_effectiveness_stacked_po = generate_effectiveness_stacked_df(df_effectiveness_po, game_type)
    game_type = 'go'
    df_effectiveness_go = generate_type_effectiveness_df(dict_types, game_type)
    df_effectiveness_stacked_go = generate_effectiveness_stacked_df(df_effectiveness_go, game_type)


    #Read the poke data
    df_pokemon = pd.DataFrame(columns=['name','Type 1','Type 2','is_default','image','pokedex_number','pokeapi_number'],)
    is_default_indicator = 1
    read_data(df_pokemon, is_default_indicator, sub_dir_defaults)
    is_default_indicator = 0
    read_data(df_pokemon, is_default_indicator, sub_dir_alts)
    df_pokemon = df_pokemon.sort_values(by=['pokedex_number','pokeapi_number'])

    #Transform pokemon typing
    #Drop all-None type pokemon (most likely an error in pokeapi)
    df_pokemon = df_pokemon[~(df_pokemon['Type 1'].isnull() & df_pokemon['Type 2'].isnull())]
    #Drop all totem pokemon (irrelevant)
    df_pokemon = df_pokemon[~df_pokemon['name'].str.contains('totem')]
    #Double up on type if it's a mono-type pokemon, just make further calculations easier.
    df_pokemon.loc[df_pokemon['Type 2'].isnull(),'Type 2'] = df_pokemon[df_pokemon['Type 2'].isnull()]['Type 1']
    #Capitalize types
    df_pokemon['Type 1'] = df_pokemon['Type 1'].str.capitalize()
    df_pokemon['Type 2'] = df_pokemon['Type 2'].str.capitalize()
    #Arrange types in alphabetical order so our groupbys will not treat Water-Pychic as different than Pychic-Water, etc.
    df_pokemon[['Type 1','Type 2']] = np.sort(df_pokemon[['Type 1','Type 2']],1)

    #Format pokemon names
    #Remove hyphens; note some pokemon genuinely do have hypens in their name.
    df_pokemon.loc[~df_pokemon['name'].isin(pokemon_with_hyphens),'name'] = df_pokemon.loc[~df_pokemon['name'].isin(pokemon_with_hyphens),'name'].str.replace('-',' ')
    #For pokemon without spaces in the name, add parenthesis around all words beyond the first in the name (and remove the parenthesis if they capture nothing)
    df_pokemon.loc[~df_pokemon['name'].isin(pokemon_with_spaces),'name'] = df_pokemon.loc[~df_pokemon['name'].isin(pokemon_with_spaces),'name'].str.replace(r'^(\S+\s*)(.*)', r'\1(\2)').str.replace('\(\)','').str.title()
    #Finally add title capitalization. In genuinely hypenated names, the letters after the hypen are not capitalized.
    df_pokemon.loc[~df_pokemon['name'].isin(pokemon_with_hyphens),'name'] = df_pokemon.loc[~df_pokemon['name'].isin(pokemon_with_hyphens),'name'].str.title()

    #Summarize the poke data
    df_type_square = generate_type_square(df_pokemon, type_order)
    df_type_stacked = generate_type_stacked(df_type_square, type_order)
    #Find pokemon belonging to each typing.
    names_per_type = generate_list_per_type(df_pokemon,'name')
    #Find image names per typing
    images_per_type = generate_list_per_type(df_pokemon,'image')

    #Add the individual pokemon names to the main stacked dataframe

    df_type_stacked = pd.merge(df_type_stacked, df_effectiveness_stacked_po, how='left',on=['Type 1','Type 2'])
    df_type_stacked = pd.merge(df_type_stacked, names_per_type, how='left',on=['Type 1','Type 2'])
    df_type_stacked = pd.merge(df_type_stacked, images_per_type, how='left',on=['Type 1','Type 2'])

    #Save dataframes to csvs
    df_type_stacked.to_csv(type_stacked_csv,index = False)

    #plot data
    #make_plots(df_type_square)

    print('Data summary finished! ' + str(time.time() - start_time) + ' seconds.')