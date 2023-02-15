# -*- coding: utf-8 -*-

__authors__ = "Piochi, LF"
__email__ = "lpiochi@cnc.uc.pt"
__group__ = "Data-Driven Molecular Design"
__group_leader__ = "Irina S. Moreira"
__project__ = "DELFOS"

import numpy as np
import pandas as pd
import random
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from delfos_resources import DEFAULT_LOCATION, SPLITS_LOCATION, model_evaluation


np.random.seed(42)
random.seed(42)

def load_datasets_concat(mode = "train"):
    
    chrom = pd.read_csv(SPLITS_LOCATION + "bulk/chromatin_" + mode + "_features.csv")
    cn = pd.read_csv(SPLITS_LOCATION + "bulk/copynumber_" + mode + "_features.csv")
    expr = pd.read_csv(SPLITS_LOCATION + "bulk/expression_" + mode + "_features.csv")
    meth = pd.read_csv(SPLITS_LOCATION + "bulk/methylation_" + mode + "_features.csv")
    mirna = pd.read_csv(SPLITS_LOCATION + "bulk/mirna_" + mode + "_features.csv")
    mordred = pd.read_csv(SPLITS_LOCATION + "drug/mordred_" + mode + "_features.csv")
    drugtax = pd.read_csv(SPLITS_LOCATION + "drug/drugtax_" + mode + "_features.csv")

    featuring = [chrom, cn, expr, meth, mirna, mordred, drugtax]
    target = featuring[0]["LN_IC50"]
    for i in range(len(featuring)):
        featuring[i] = featuring[i].iloc[:, 4:]
    
    concat_data = pd.concat(featuring, axis = 1)
    scaler = StandardScaler()
    concat_data = scaler.fit_transform(concat_data)
    concat_data = np.asarray(concat_data).astype('float64')

    return concat_data, target

# Load datasets
x_train, target = load_datasets_concat(mode = "train")
x_test, target_test = load_datasets_concat(mode = "test")
x_val_cell, target_validation_cell = load_datasets_concat(mode = "valcell")
x_val_drug, target_validation_drug = load_datasets_concat(mode = "valdrug")

model = xgb.XGBRFRegressor()
model.fit(x_train, target)

score = model.score(x_train, target)  
print("Training score: ", score)

predicted_train = model.predict(x_train)
predicted_test = model.predict(x_test)
predicted_val_cells = model.predict(x_val_cell)
predicted_val_drugs = model.predict(x_val_drug)

model_evaluation(target, predicted_train, verbose = True, subset_type = "train", write_mode= True)
model_evaluation(target_test, predicted_test, verbose = True, subset_type = "test", write_mode= True)
model_evaluation(target_validation_cell, predicted_val_cells, verbose = True, subset_type = "val_cells", write_mode= True)
model_evaluation(target_validation_drug, predicted_val_drugs, verbose = True, subset_type = "val_drugs", write_mode= True)
