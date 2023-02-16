# -*- coding: utf-8 -*-

__author__ = "Piochi, LF"
__email__ = "lpiochi@cnc.uc.pt"
__group__ = "Data-Driven Molecular Design"
__group_leader__ = "Irina S. Moreira"
__project__ = "DELFOS"

import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
import pickle
from delfos_resources import DEFAULT_LOCATION, TRAIN_SPLIT, TEST_SPLIT, CELL_SPLIT, DRUG_SPLIT, open_txt

random.seed(42)
np.random.seed(42)

# Integrate the data with their respective train-test and leave-out splits ## consider creating a class
def preprocessing(feature, feature_name,
                  input_train = TRAIN_SPLIT, 
                  input_test = TEST_SPLIT, 
                  input_val_cell = CELL_SPLIT, 
                  input_val_drug = DRUG_SPLIT):
    
    if feature_name == "mordred" or feature_name == "drugtax":
        LOC = DEFAULT_LOCATION + "data/processed/drug/"
        merg = "DRUG_NAME"
    else:
        LOC = DEFAULT_LOCATION + "data/processed/bulk/"
        merg = "CELL_LINE_NAME"

    # Remove zero-variance features
    def drop_zero_var_train(train_dataset):
        dropped = train_dataset.loc[:, train_dataset.std() > 0]
        # removed_cols = np.intersect1d(dropped.columns, train_dataset.columns)
        # with open(input_csv[:] + "_train_" + "dropped.txt", "w") as outfile:
        #     outfile.write("\n".join(removed_cols))
        return dropped
    
    # Read the file containing the training drug-cell line pairs and merge with the dataset - this merge will take place
    # on the "CELL_LINE_NAME" or "DRUG_NAME" columns, depending on whether it is a drug or cell dataset.
    train_file = pd.read_csv(input_train)
    train_data = train_file.merge(feature, how = 'left')
    
    # Retrieve the mean value of repeated drug-cell line pairs # Create two-mode function: one for train and other for test/eval
    train_data = train_data.groupby(['CELL_LINE_NAME', 'DRUG_NAME']).mean().reset_index()
    train_data_val = train_data.iloc[:, 3:].select_dtypes(['number'])
    train_data_val = drop_zero_var_train(train_data_val)
    zero_var_drop_list = list(train_data_val.columns)   
    
    # Repeat the procedure, excluding removing zero-variance features, for the remaining datasets
    # ### CREATE FUNCTION HERE
    test_file = pd.read_csv(input_test)
    test_data = test_file.merge(feature, how = 'left')
    test_data = test_data.groupby(['CELL_LINE_NAME', 'DRUG_NAME']).mean().reset_index()
    test_data_val = test_data.iloc[:, 3:].select_dtypes(['number'])
    test_data_val = test_data_val.loc[:, test_data_val.columns.isin(zero_var_drop_list)]
        
    val_cell_file = pd.read_csv(input_val_cell)
    val_cell_data = val_cell_file.merge(feature, how = 'left')
    val_cell_data = val_cell_data.groupby(['CELL_LINE_NAME', 'DRUG_NAME']).mean().reset_index()
    val_cell_data_val = val_cell_data.iloc[:, 3:].select_dtypes(['number'])
    val_cell_data_val = val_cell_data_val.loc[:, val_cell_data_val.columns.isin(zero_var_drop_list)]
    
    val_drug_file = pd.read_csv(input_val_drug)
    val_drug_data = val_drug_file.merge(feature, how = 'left')
    val_drug_data = val_drug_data.groupby(['CELL_LINE_NAME', 'DRUG_NAME']).mean().reset_index()
    val_drug_data_val = val_drug_data.iloc[:, 3:].select_dtypes(['number'])
    val_drug_data_val = val_drug_data_val.loc[:, val_drug_data_val.columns.isin(zero_var_drop_list)]
    
    scaler = StandardScaler()
    
    # Fit the scaler to the training data
    sc_train = scaler.fit(train_data_val)
    
    # Save the fitted scaler as a pkl file
    pickle.dump(scaler, open(LOC + str(feature_name) + "_scaler.pkl", 'wb'))
    
    # Transform the datasets according to the fitted values
    sc_train = scaler.transform(train_data_val)
    sc_test = scaler.transform(test_data_val)
    sc_val_cell = scaler.transform(val_cell_data_val)
    sc_val_drug = scaler.transform(val_drug_data_val)

    # Replace any missing values with zeros
    train_data_val_df = pd.DataFrame(sc_train, columns = train_data_val.columns, index = None)
    for i in train_data_val_df.columns[train_data_val_df.isnull().any(axis=0)]:
        train_data_val_df[i].fillna(0, inplace=True)
    train_data_val_df = pd.concat([train_data.iloc[:,0:3].reset_index(drop = True), train_data_val_df], axis = 1)
    
    test_data_val_df = pd.DataFrame(sc_test, columns = test_data_val.columns, index = None)
    for i in test_data_val_df.columns[test_data_val_df.isnull().any(axis=0)]:
        test_data_val_df[i].fillna(0, inplace=True)
    test_data_val_df = pd.concat([test_data.iloc[:,0:3], test_data_val_df], axis = 1)
        
    val_cell_data_df = pd.DataFrame(sc_val_cell, columns = val_cell_data_val.columns, index = None)
    for i in val_cell_data_df.columns[val_cell_data_df.isnull().any(axis=0)]:
        val_cell_data_df[i].fillna(0, inplace=True)
    val_cell_data_df = pd.concat([val_cell_data.iloc[:,0:3], val_cell_data_df], axis = 1)

    val_drug_data_df = pd.DataFrame(sc_val_drug, columns = val_drug_data_val.columns, index = None)
    for i in val_drug_data_df.columns[val_drug_data_df.isnull().any(axis=0)]:
        val_drug_data_df[i].fillna(0, inplace=True)
    val_drug_data_df = pd.concat([val_drug_data.iloc[:,0:3], val_drug_data_df], axis = 1)
    
    # Save the integrated and normalized datasets
    train_data_val_df.to_csv(LOC + str(feature_name) + "_" + "train_features.csv", sep = ",", index = True)
    test_data_val_df.to_csv(LOC + str(feature_name) + "_" + "test_features.csv", sep = ",", index = True)
    val_cell_data_df.to_csv(LOC + str(feature_name) + "_" + "valcell_features.csv", sep = ",", index = True)
    val_drug_data_df.to_csv(LOC + str(feature_name) + "_" + "valdrug_features.csv", sep = ",", index = True)
    
    print("train" + "_" + str(train_data_val_df.shape))
    print("test" + "_" + str(test_data_val_df.shape))
    print("cell_val" + "_" + str(val_cell_data_df.shape))
    print("drug_val" + "_" + str(val_drug_data_df.shape))

feature_dict = {
    "mordred" : pd.read_hdf(DEFAULT_LOCATION + "data/h5/mordred_feat.h5"),
    "drugtax" : pd.read_hdf(DEFAULT_LOCATION + "data/h5/drugtax_feat.h5"),
    "chromatin" : pd.read_hdf(DEFAULT_LOCATION + "data/h5/CCLE_chromatin.h5"),
    "copynumber" : pd.read_hdf(DEFAULT_LOCATION + "data/h5/CCLE_copynumber.h5"),
    "expression" : pd.read_hdf(DEFAULT_LOCATION + "data/h5/CCLE_expression.h5"),
    "methylation" : pd.read_hdf(DEFAULT_LOCATION + "data/h5/CCLE_methylation.h5"),
    "mirna" : pd.read_hdf(DEFAULT_LOCATION + "data/h5/CCLE_mirna.h5")
}

for key in feature_dict:
    preprocessing(feature = feature_dict[key], feature_name = key)
