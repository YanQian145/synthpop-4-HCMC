import pandas as pd
import multiprocessing
import re
import math
import os
import time

import random
import itertools 
pd.options.mode.chained_assignment = None  # default='warn'

from collections import Counter
import pickle


from os import listdir
from os.path import isfile, join
from collections import Counter



def get_HH(district):
    # Load census 
    with open('data/dict_census.pickle', 'rb') as handle:
        dict_df_census = pickle.load(handle)

    district = district
    simu = 13
     
    HH_head_path = 'output/HH_head/sim'+ str(simu)+ '/synthHH_dis' + district + '.pickle'

    start_time = pd.to_datetime("today")

    with open(HH_head_path, 'rb') as handle:
        synth_pop_list = pickle.load(handle)

    d_fin = {}
    k = 0
   
    for wards in synth_pop_list[district].keys():
        print(district, wards)
        tt = synth_pop_list[district][wards]
        lst2 = [item[0] for item in tt]

        z = Counter(lst2)
        j = 0
        d = {}

        for hh, nb in z.items():
            for i in range(nb):
                for entry in dict_df_census[district]:
                    if entry['HH_ID'] == hh:
                        d[j] = {"HH_ID": entry['HH_ID'], "HH_ind": entry["HH_ind"]}
                        j = j + 1 
        d_fin[wards] = d

    folder_path = 'output/HH/sim'+ str(simu) + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with open(folder_path+ 'HH_'+ str(district) + '.pickle', 'wb') as handle:
        pickle.dump(d_fin, handle, protocol=pickle.HIGHEST_PROTOCOL)


    end_time =  pd.to_datetime("today")
    diff =  end_time - start_time
    text = "generated HH for " + str(district) + " from " + str(start_time) + " to " + str(end_time) + " - total duration: " +  str(diff)
    print(text)


if __name__ == "__main__":
    nb_cpu = multiprocessing.cpu_count()
    print("nb cpu ", nb_cpu)
    p  = multiprocessing.Pool(30) 
    district_list = ['760', '761', '762', '763', '764', '765', '766', '767', '768', '769', '770', '771', '772', '773', '774', '775', '776', '777', '778', '783', '784', '785', '786', '787']

    p.map(get_HH, district_list)

    p.close()