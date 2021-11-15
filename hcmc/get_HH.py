import multiprocessing
import os

import pandas as pd

import pickle

from collections import Counter

pd.options.mode.chained_assignment = None  # default='warn'


def get_HH(district):
    # Load census
    pkl_file = os.path.join(
        os.path.dirname(__file__), "data", "dict_census.pickle"
    )  # Load pickle file. __file__ is the path of this file.
    with open(pkl_file, "rb") as handle:
        dict_df_census = pickle.load(handle)

    district = district
    simu = 13

    HH_head_path = os.path.join(
        "output", "HH_head", f"sim{simu}", f"synthHH_dis{district}.pkl"
    )

    start_time = pd.to_datetime("today")

    with open(HH_head_path, "rb") as handle:
        synth_pop_list = pickle.load(handle)

    d_fin = {}

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
                    if entry["HH_ID"] == hh:
                        d[j] = {"HH_ID": entry["HH_ID"], "HH_ind": entry["HH_ind"]}
                        j = j + 1
        d_fin[wards] = d

    folder_path = os.path.join("output", "HH", f"sim{simu}")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    pkl_file = os.path.join(folder_path, f"HH_dis{district}.pkl")
    with open(pkl_file, "wb") as handle:
        pickle.dump(d_fin, handle, protocol=pickle.HIGHEST_PROTOCOL)

    end_time = pd.to_datetime("today")
    diff = end_time - start_time

    text = f"generated HH for {district} from {start_time} to {end_time} - total duration: {diff}"
    print(text)


if __name__ == "__main__":
    nb_cpu = multiprocessing.cpu_count()
    print("nb cpu ", nb_cpu)
    p = multiprocessing.Pool(nb_cpu)
    district_list = [
        "760",
        "761",
        "762",
        "763",
        "764",
        "765",
        "766",
        "767",
        "768",
        "769",
        "770",
        "771",
        "772",
        "773",
        "774",
        "775",
        "776",
        "777",
        "778",
        "783",
        "784",
        "785",
        "786",
        "787",
    ]

    p.map(get_HH, district_list)

    p.close()
