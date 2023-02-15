# -*- coding: utf-8 -*-

__author__ = "Piochi, LF"
__email__ = "lpiochi@cnc.uc.pt"
__group__ = "Data-Driven Molecular Design"
__group_leader__ = "Irina S. Moreira"
__project__ = "DELFOS"

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import random
import pickle
from delfos_resources import DEFAULT_LOCATION, SPLITS_LOCATION, open_txt, LEAVE_CELL_LOC

np.random.seed(42)
random.seed(42)

"""
This function will append and concatenate a list of single-cell expression data into the desired shape. 
Cell lines with less available single-cells than the defined threshold will contain NaNs.
By default, this will select only 10 single cells from each cell line and will expect files with 2000 features/genes.
"""
def format_single_cells(single_cell_data_list, num_of_cells = 10, num_of_sc_features = 2000):

    def pad_with_nan(A, r, c):
       out = np.empty((r, c))
       out[:] = np.nan
       r_, c_ = np.shape(A)
       out[0:r_, 0:c_] = A
       return out
    
    sc_total_train = []
    for i in single_cell_data_list:
        cell_name = str(i.iloc[1,0])
        start = i.iloc[:num_of_cells, 1:]
        temp = pad_with_nan(start, num_of_cells, num_of_sc_features)
        temp = pd.DataFrame(temp, columns = start.columns.tolist())
        temp.insert(0, 'CELL_LINE_NAME', cell_name)
        sc_total_train.append(temp)
    sc_total = pd.concat(sc_total_train, ignore_index = True)
    
    # Standardize the data
    sc_total_clean = sc_total.iloc[:, 1:]
    scaler = StandardScaler()
    sc_total_scaled = scaler.fit(sc_total_clean)
    pickle.dump(scaler, open(SPLITS_LOCATION + "single-cell/sc_total_scaler.pkl", 'wb'))
    sc_total_scaled = scaler.transform(sc_total_clean)
    sc_total_scaled_train = pd.DataFrame(sc_total_scaled, columns = sc_total_clean.columns.tolist())
    sc_total_scaled_train.insert(0, 'CELL_LINE_NAME', sc_total["CELL_LINE_NAME"])
    
    return sc_total_scaled_train


"""
This function will save individual files containing data from each cell line, replacing any missing values for the 
median value of expression for that gene for that cell line. Either "train" or "validation" must be chosen as mode.
"""
def register_sc_per_line(sc_file, mode = "train"):

    open_cell_validation = open_txt(LEAVE_CELL_LOC)

    if mode == "train":
        sc_data = sc_file[~sc_file.iloc[:,0].isin(open_cell_validation)]
    if mode == "validation":
        sc_data = sc_file[sc_file.iloc[:,0].isin(open_cell_validation)]

    cell_line_list = list(sc_data["CELL_LINE_NAME"].unique())

    for sel_cel in cell_line_list:

        df = sc_data.loc[sc_data.CELL_LINE_NAME == sel_cel]

        for i in df.columns[df.isnull().any(axis=0)]:
            df[i].fillna(df[i].median(), inplace=True)

        if mode == "train":
            df.to_hdf(SPLITS_LOCATION + "single-cell/individual_tt/" + sel_cel + ".h5", 'w')
        if mode == "validation":
            df.to_hdf(SPLITS_LOCATION + "single-cell/individual_valid/" + sel_cel + ".h5", 'w')



# Read single-cell expression files and add them into a list.
sc_files = os.listdir(DEFAULT_LOCATION + "data/single-cell/individual")
single = []
for i in sc_files:
    temp = pd.read_csv(DEFAULT_LOCATION + "data/single-cell/individual/" +  i)
    temp = temp.sample(frac = 1)
    temp = temp.iloc[:50, 1:]
    if temp.shape[0] > 0:
        single.append(temp)

# Concatenate sc data
sc_total = format_single_cells(single)

# Register the data per cell line
register_sc_per_line(sc_total, mode = "train")
register_sc_per_line(sc_total, mode = "validation")