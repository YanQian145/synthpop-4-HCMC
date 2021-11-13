#libraries

import pandas as pd
import re
import numpy as nps
import math
import os
import time
from pandas.core.common import flatten
import random
import itertools
from collections import Counter
import pickle
import multiprocessing
from pandas import DataFrame
import time
import itertools
import scipy as sp
pd.options.mode.chained_assignment = None  # default='warn'




def initialization(district, ward, merged_census_HH, census_HH, HH_level_table, POP_level_table ):
    #initialize HH count
    HH_level_table[district][ward]['tot_HHI'] = 0
    # initialize individual count
    POP_level_table[district][ward]['tot_POPI'] = 0
    # calc proba for HH acting as weight
    HH_level_table[district][ward]['proba'] = (1/HH_level_table[district][ward]['tot_census']) * (HH_level_table[district][ward]['tot_HH']- HH_level_table[district][ward]['tot_HHI'])/((HH_level_table[district][ward]['tot_HH']- HH_level_table[district][ward]['tot_HHI']).sum())
    # merge households in census with calculated proba weight    
    merged_census_HH[district] = pd.merge(census_HH[district], HH_level_table[district][ward][['HH_type','proba']] , how='left', left_on=['HH_type'], right_on = ['HH_type'])

    return HH_level_table, POP_level_table, merged_census_HH



def pick_and_check_HH_desirable(i_max, district, ward, HH_level_table, POP_level_table, merged_census_HH, percent):
    # random draw on district level
    sp.random.seed()
    random_draw =  merged_census_HH[district].sample(n=1,  weights= merged_census_HH[district]['proba'])
    # get hh type and age_gender of members in HH        
    ID_HH_type = random_draw.iloc[0]['HH_type']
    ID_POP_type_list = random_draw.iloc[0]['ID_match_ag']    

    ## -- HH level
    # get the tot HH and tot HHI for that corresponding HH ID of random draw
    nb_current_hh_of_type = HH_level_table[district][ward]['tot_HHI'][HH_level_table[district][ward]['HH_type'] == ID_HH_type].iloc[0]
    nb_total_hh_of_type = HH_level_table[district][ward]['tot_HH'][HH_level_table[district][ward]['HH_type'] == ID_HH_type].iloc[0]

    ## -- IND level
    # get all the rows where age_gender matches with the chosen one and check if current nb < desired
    match_pop = POP_level_table[district][ward].loc[POP_level_table[district][ward]['ID_match_ag'].isin(ID_POP_type_list)]

    dd = dict(Counter(ID_POP_type_list))
    df = pd.DataFrame.from_dict(dd, orient='index').reset_index()
    df = df.rename(columns={"index": "ID_match_ag", 0: "count"})

    df2 = pd.merge(match_pop, df )

    # if tot popi is smaller than pop with x percentage, put 0, else 1
    counter_match_pop = np.where(df2['tot_POPI'] + df2['count']<= df2['tot_POP']+ df2['tot_POP']*(percent/100) , 0, 1)
    
    # check
    if (nb_current_hh_of_type <= nb_total_hh_of_type + nb_total_hh_of_type*(percent/100)):
        if (sum(counter_match_pop) == 0):
            continue_looking = False
            i_max = 0
            return continue_looking,random_draw, i_max
    
        elif (sum(counter_match_pop) != 0) :
            #look until i_max reaches 10 000
            if i_max <= 10000:
                continue_looking = True
                POP_level_table[district][ward]['diff'] = (POP_level_table[district][ward]['tot_POP'] - POP_level_table[district][ward]['tot_POPI'])
                i_max = i_max + 1
                return continue_looking, random_draw, i_max  
            elif i_max > 10000:
                continue_looking = False
                return continue_looking, random_draw, i_max



def update_counters(district, ward, merged_census_HH, HH_level_table, POP_level_table, census_HH, synthetic_population, random_draw, iteration_nb):
    ID_HH_type = random_draw.iloc[0]['HH_type']
    ID_POP_type_list = random_draw.iloc[0]['ID_match_ag']

    ## HH level
    # get the tot HH and toh HHI for that corresponding HH ID of random draw
    nb_current_hh_of_type = HH_level_table[district][ward]['tot_HHI'][HH_level_table[district][ward]['HH_type'] == ID_HH_type].iloc[0]
    nb_total_hh_of_type = HH_level_table[district][ward]['tot_HH'][HH_level_table[district][ward]['HH_type'] == ID_HH_type].iloc[0]
    
    ## POP level
    # get all the rows where age_gender matches with the chosen one
    match_pop = POP_level_table[district][ward].loc[POP_level_table[district][ward]['ID_match_ag'].isin(ID_POP_type_list)]
    
    # update HH
    HH_level_table[district][ward]['tot_HHI'][HH_level_table[district][ward]['HH_type'] == ID_HH_type ] = HH_level_table[district][ward]['tot_HHI']+1
    #update proba
    HH_level_table[district][ward]['proba'] = (1/HH_level_table[district][ward]['tot_census']) * (HH_level_table[district][ward]['tot_HH']- HH_level_table[district][ward]['tot_HHI'])/((HH_level_table[district][ward]['tot_HH']- HH_level_table[district][ward]['tot_HHI']).sum())

    # merge households in census with calculated proba weight
    merged_census_HH[district] = pd.merge(census_HH[district], HH_level_table[district][ward][['HH_type','proba']], how='left', left_on=['HH_type'], right_on = ['HH_type'])   
    
    ## update POP
    # get nb of age-gender groups per HH
    _dict_counts_per_gender_age = pd.DataFrame.from_dict(Counter(ID_POP_type_list), orient='index').reset_index()
    _dict_counts_per_gender_age = _dict_counts_per_gender_age.rename(columns={"index": "ID_match_ag", 0: "count"})
    # transform into dataframe, and merge with POP dd to sum together counts
    _temp_dict_POP = pd.merge(POP_level_table[district][ward], _dict_counts_per_gender_age , how='left', left_on=['ID_match_ag'], right_on = ['ID_match_ag'])
    _temp_dict_POP['count'] = _temp_dict_POP['count'].fillna(0) # fill NAs with 0
    # sum counts
    _temp_dict_POP['tot_POPI'] = _temp_dict_POP['tot_POPI'] + _temp_dict_POP['count'] 
    # drop the count column
    _temp_dict_POP = _temp_dict_POP.drop(columns=['count'])
    # update dict POP
    POP_level_table[district][ward] = _temp_dict_POP.copy()
        
    ## clone HH
    random_draw['id_for_later'] = iteration_nb
    # transform into list
    _flatten_rando_draw= random_draw.values.flatten().tolist()
    # clone the new HH to synth pop list
    synthetic_population[district][ward].append(_flatten_rando_draw)
    
    return HH_level_table, POP_level_table, synthetic_population, iteration_nb, merged_census_HH


#generate the HHheads for x times for all districts
def generate_HH_head(simulation, district):

    percent = 10

    # Load pickle input data
    with open('data/2019_census.pickle', 'rb') as handle:
        dict_df_census = pickle.load(handle)
    with open('data/2019_households_total.pickle', 'rb') as handle:
        dict_households = pickle.load(handle)
    with open('data/2019_age_gender_total.pickle', 'rb') as handle:
        dict_POP = pickle.load(handle)


    ## ------------ PREPARATION STEP 
    ##  get HHs of census
    # & get table of hh types with count in census and total count

    dict_HH = dict()
    dict_df_census_HH =  dict()
    dict_df_census_HH =  {elem : pd.DataFrame for elem in dict_df_census.keys()}

    dict_households_c =  dict_households.copy()

    # assign district to key variable
    key = district

    # merge HH  freq theo with census, create nested dictionary: k1 = district, k2 = wards
    dict_df_census_HH[key] =  pd.DataFrame(dict_df_census[key].groupby(['ID_2','ward_nb', 'HH_type'])['ID_match_ag'].apply(list).reset_index())
    _temp = pd.DataFrame(dict_df_census_HH[key][['ward_nb', 'HH_type']].value_counts().reset_index(name= 'tot_census'))
    dict_households_c[key] = pd.merge(dict_households_c[key], _temp, how = 'left')

    ## fill 0 values with 1, i,e, for those that don't have entry in census
    dict_households_c[key] = dict_households_c[key].fillna(1)
    
    # transform into nested dictiionary with district and wards as keys
    UniqueWard= dict_households_c[key].ward_nb.unique()
    tt = {elem : pd.DataFrame for elem in UniqueWard}

    for ward in tt.keys():
        tt[ward] = dict_households_c[key][['HH_type', 'tot_census', 'tot_HH']][:][dict_households_c[key].ward_nb == ward].reset_index(drop=True)
        # if tot HH is zero, then tot_census is too
        tt[ward]['tot_census'][tt[ward]['tot_HH'] == 0] = 0
    dict_HH[key] = tt

    ### ------- SYNTH POPULATION GENERATION

    # empty nested dict with k1 = district, k2 = wards,
    synth_pop_list =  dict()
    synth_pop_list =  {elem : dict() for elem in dict_households.keys()}

    UniqueWard= dict_households[district].ward_nb.unique()
    tt = {elem : list() for elem in UniqueWard}
    synth_pop_list[district] = tt

    merged_census_HH_proba = dict_df_census_HH.copy()

    for ward, df in dict_HH[district].items():

        print('start district ', district, ' , ward ' , ward,  ' with x HH: ', dict_HH[district][ward]['tot_HH'].sum(), ' simulation ', str(simulation))

        start_time = pd.to_datetime("today")
        iteration_nb = 0

        iteration_max = dict_HH[district][ward]['tot_HH'].sum()

        # initialize counting tables (B) and compute selection proba (C)
        dict_HH, dict_POP, merged_census_HH_proba = initialization(district, ward, merged_census_HH = merged_census_HH_proba, census_HH = dict_df_census_HH, HH_level_table = dict_HH, POP_level_table= dict_POP)

        # while we do not exceed the number of HH, select random HH (D) and check its desirability (E), then update the count and cloen the HH (F)
        while (merged_census_HH_proba[district]['proba'].sum() > 0) & (iteration_nb <= iteration_max):
            continue_looking = True
            i_max = 0
            while continue_looking:
                continue_looking, desirable_HH, i_max = pick_and_check_HH_desirable(i_max = i_max, district = district, ward = ward, HH_level_table = dict_HH , POP_level_table = dict_POP, merged_census_HH = merged_census_HH_proba, percent = percent)
                
            dict_HH, dict_POP, synth_pop_list, iteration_nb, merged_census_HH_proba = update_counters(district = district, ward=ward, merged_census_HH = merged_census_HH_proba, HH_level_table = dict_HH, POP_level_table = dict_POP,census_HH = dict_df_census_HH, synthetic_population = synth_pop_list, random_draw = desirable_HH, iteration_nb = iteration_nb)
            iteration_nb = iteration_nb + 1

        # check time of run 
        end_time =  pd.to_datetime("today")
        diff =  end_time - start_time
        text = "generated HH for district_" + str(district) + " ward_" + str(ward)  + " from " + str(start_time) + " to " + str(end_time) + " sim_" + str(simulation) +  " - total_" +  str(diff)
        print(text)


    # Store data, for each simulation create a folder with all the synth pop
    out_folder = 'output/HH_head/sim'+ str(simulation)
    
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    #synth pop
    with open(out_folder +'/synthHH_dis' + str(district) + '.pickle', 'wb') as handle:
        pickle.dump(synth_pop_list, handle, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
     nb_cpu = multiprocessing.cpu_count()
     print("nb cpu ", nb_cpu)
     p  = multiprocessing.Pool(30) 

     simulation = list(range(0,30)) 
     district = ['760', '761', '762', '763', '764', '765', '766', '767', '768', '769', '770', '771', '772', '773', '774', '775', '776', '777', '778', '783', '784', '785', '786', '787']

     arg = list(itertools.product(simulation, district))  

     p.starmap(generate_HH_head, arg)

     p.close()
     
