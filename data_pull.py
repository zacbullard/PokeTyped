import requests #For querying APIs
import os #For directory manipulation
import pickle #For saving fetched data
import time #For benchmarking performance

#I use PokeAPI to retrieve the data on the pokemon.
#The PokeAPI is RESTful, i.e. the API conforms to an architectural style that is clean and scalable.
poke_api = 'https://pokeapi.co/api/v2/'

#Directories for writing
cache_dir = 'cache'
sub_dir_species = cache_dir + '/species'
sub_dir_defaults = cache_dir + '/defaults'
sub_dir_alts = cache_dir + '/alts'
poke_image_dir = 'icons_pokemon'

#Save data from pokeapi into a local cache
def pull_species_data(sub_dir_species):
    #Retrieve the current number of pokemon in existence, using the National pokedex number.
    max_pokemon = requests.get(poke_api + 'pokemon-species/?limit=0').json()['count']

    for i in range(1,max_pokemon+1):
        #Call a pokemon's data request
        print("Getting pokemon "+str(i)+"...")
        try:
            pokemon = requests.get(poke_api + '/pokemon-species/'+str(i)).json()
        except ValueError:
            print("Missing species data from API! "+ variety['pokemon']['name'])
                with open(cache_dir+"/missing_species.txt", 'a') as f:
                        f.write(str(i)+'\n')

        pickle.dump(pokemon, open(sub_dir_species+'/'+str(i)+'.pkl', 'wb' ) )

#Pokemon can have several forms, so we need to pull data on each form
def pull_form_data(sub_dir_species, sub_dir_defaults, sub_dir_alts):

    for filename in os.listdir(sub_dir_species):
        species = pickle.load(open(sub_dir_species+'/'+filename, 'rb'))
        for variety in species['varieties']:
            #Call a pokemon's data request
            print("Getting variety "+variety['pokemon']['name']+"...")
            try:
                pokemon = requests.get(poke_api + '/pokemon/'+variety['pokemon']['name']).json()
            except ValueError:
                print("Missing form data from API! "+ variety['pokemon']['name'])
                with open(cache_dir+"/missing_form.txt", 'a') as f:
                        f.write(variety['pokemon']['name']+'\n')

            #Category of form
            sub_dir = None
            if variety['is_default'] == True:
                sub_dir = sub_dir_defaults
            elif variety['is_default'] == False:
                sub_dir = sub_dir_alts
            else:
                raise Exception('Unknown is_default value: ' + variety['is_default'])

            pickle.dump(pokemon, open(sub_dir+'/'+variety['pokemon']['name']+'.pkl', 'wb'))

#Download information on each type
def pull_types(cache_dir):
    #Retrieve the current number of types in existence.
    max_types = requests.get(poke_api + 'type/?limit=0').json()['count']

    excluded_types = ["unknown","shadow"]

    dict_types = {}

    for i in range(1,max_types -len(excluded_types) +1):
        #Call a pokemon's data request
        type = requests.get(poke_api + 'type/'+str(i)).json()
        if type['name'] not in excluded_types:
            dict_types[type['name']] = type

    pickle.dump(dict_types, open(cache_dir+'/dict_types.pkl', 'wb' ) )


#Download images for each pokemon
def pull_images(poke_image_dir, sub_dir_defaults, sub_dir_alts, cache_dir):

    #Remove old "missing sprites" file if it existsif os.path.exists(filename):
    if os.path.exists(cache_dir+"/missing_sprites.txt"):
        os.remove(cache_dir+"/missing_sprites.txt")

    for variety_dir in [sub_dir_defaults, sub_dir_alts]:
        for filename in os.listdir(variety_dir):
            variety = pickle.load(open(variety_dir+'/'+filename, 'rb'))

            if variety['sprites']['front_default'] is None:
                print("No sprite for "+ variety['name'])
                with open(cache_dir+"/missing_sprites.txt", 'a') as f:
                        f.write(variety['name']+'\n')
            else:
                print("Getting image for variety "+variety['name']+" at location "+variety['sprites']['front_default']+"...")
                image_request = requests.get(variety['sprites']['front_default'])
                if image_request.status_code == 200:
                    with open(poke_image_dir+"/"+variety['name']+'.png', 'wb') as f:
                        f.write(image_request.content)

if __name__ == '__main__':

    start_time = time.time()

    sub_dirs = [cache_dir,
                sub_dir_species,
                sub_dir_defaults,
                sub_dir_alts,
                poke_image_dir
                ]

    #Create directory structure
    for sub_dir in sub_dirs:
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)

    pull_species_data(sub_dir_species)

    pull_form_data(sub_dir_species, sub_dir_defaults, sub_dir_alts)

    pull_types(cache_dir)

    pull_images(poke_image_dir, sub_dir_defaults, sub_dir_alts, cache_dir)

    print('Data pull finished! ' + str(time.time() - start_time) + ' seconds.')

