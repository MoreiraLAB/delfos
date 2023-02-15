# -*- coding: utf-8 -*-

__author__ = "Piochi, LF"
__email__ = "lpiochi@cnc.uc.pt"
__group__ = "Data-Driven Molecular Design"
__group_leader__ = "Irina S. Moreira"
__project__ = "DELFOS"

import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from delfos_resources import open_txt, RANDOM_STATE, TARGET_LOC, LEAVE_CELL_LOC, LEAVE_DRUG_LOC, \
TRAIN_SPLIT, TEST_SPLIT, CELL_SPLIT, DRUG_SPLIT

random.seed(42)
np.random.seed(42)

# Randomly select unique drugs and cell lines for subsequent splits into leave-cell-out and leave-drug-out datasets
def validation_splits(target_hdf = TARGET_LOC):
    
    target = pd.read_hdf(target_hdf)
    cells = pd.Series(target["CELL_LINE_NAME"].unique())
    cell_valid = cells.sample(int(0.1*len(cells)), random_state = RANDOM_STATE)

    drugs = pd.Series(target["DRUG_NAME"].unique())
    drug_valid = drugs.sample(int(0.1*len(drugs)), random_state = RANDOM_STATE)

    def register(dataset, regist):
        with open(regist, 'w') as log:
            for entry in dataset:
                log.write(entry + '\n')
        
    register(cell_valid, LEAVE_CELL_LOC)
    register(drug_valid, LEAVE_DRUG_LOC)


# Extract the leave-out unique elements and perform a train-test split on the remaining drug-cell line pairs
def tt_splits(target_hdf = TARGET_LOC,
              cell_valid = LEAVE_CELL_LOC,
              drug_valid = LEAVE_DRUG_LOC):
    
    cell_valid = open_txt(cell_valid)
    drug_valid = open_txt(drug_valid)
    
    target = pd.read_hdf(target_hdf)
    
    cell_validation = target[target["CELL_LINE_NAME"].isin(cell_valid)]
    cell_validation = cell_validation[~cell_validation["DRUG_NAME"].isin(drug_valid)]
    
    drug_validation = target[target["DRUG_NAME"].isin(drug_valid)]
    drug_validation = drug_validation[~drug_validation["CELL_LINE_NAME"].isin(cell_valid)]

    cell_validation.to_csv(CELL_SPLIT, sep = ",", index = False)
    drug_validation.to_csv(DRUG_SPLIT, sep = ",", index = False)
    
    target = target[~target["CELL_LINE_NAME"].isin(cell_valid)]
    target = target[~target["DRUG_NAME"].isin(drug_valid)]
    
    x_train, x_test = train_test_split(target, test_size = 0.3, random_state = RANDOM_STATE)

    x_train.to_csv(TRAIN_SPLIT, sep = ",", index = False)
    x_test.to_csv(TEST_SPLIT, sep = ",", index = False)


# Perform validation and train-test splits
validation_splits(target_hdf = TARGET_LOC)
tt_splits(target_hdf = TARGET_LOC, cell_valid = LEAVE_CELL_LOC, drug_valid = LEAVE_DRUG_LOC)
