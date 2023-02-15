# -*- coding: utf-8 -*-

__authors__ = "Piochi, LF; Preto, AJ"
__email__ = "lpiochi@cnc.uc.pt", "martinsgomes.jose@gmail.com"
__group__ = "Data-Driven Molecular Design"
__group_leader__ = "Irina S. Moreira"
__project__ = "DELFOS"


import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
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


def hypertuning():
    class MyHyperModel(kt.HyperModel):
        def build(self, hp):
            layers_range = (8, 16)
            units_range = (200, 500, 25)
            lr_values = [1e-3, 1e-4]
            model = keras.Sequential()
            model.add(keras.layers.Dense(units=hp.Int("initial", min_value= 1000, max_value=3000, step=500), input_dim = np.shape(x_train)[1], activation = 'relu'))
            model.add(keras.layers.Dense(units=hp.Int("initial", min_value= 500, max_value=2000, step=500), activation = 'relu'))
            model.add(
                      keras.layers.Dropout(
                          hp.Float(
                              'dropout',
                              min_value=0.1,
                              max_value=0.5,
                              default=0.1,
                              step=0.1)
                      )
                  )
            for i in range(hp.Int('layers', layers_range[0], layers_range[1])):
              model.add(keras.layers.Dense(units=hp.Int('units_' + str(i),  
                                          min_value=units_range[0], max_value=units_range[1], 
                                          step=units_range[2]), activation='relu'))
              model.add(
                      keras.layers.Dropout(
                          hp.Float(
                              'dropout',
                              min_value=0.1,
                              max_value=0.5,
                              default=0.1,
                              step=0.1)
                      )
                  )
            model.add(keras.layers.Dense(1, activation = "linear"))

            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = hp.Choice('learning_rate', values = lr_values)), 
                          loss= "mse", metrics=[tf.keras.metrics.RootMeanSquaredError(), "mean_absolute_error"])
            return model
    
        def fit(self, hp, model, *args, **kwargs):
            return model.fit(
                *args,
                batch_size=hp.Choice("batch_size", [256, 512]),
                **kwargs,
            )
    
    tuner = kt.Hyperband(MyHyperModel(),
                          project_name= "prec_1", 
                          objective='val_loss',
                          max_epochs=200,
                          factor = 3,
                          seed = 42)
    
    keras_callbacks = [
          tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode = 'min', patience = 10, restore_best_weights = True),
          tf.keras.callbacks.ReduceLROnPlateau(factor = 0.1, min_lr = 1e-5, patience=5)
    ]
    
    tuner.search(x_train, target, epochs=200, callbacks = keras_callbacks, verbose = 2, validation_split = 0.1)
    
    best_hp = tuner.get_best_hyperparameters()[0]
       
    h_model = tuner.hypermodel.build(best_hp)
    
    h_model.fit(x_train, target, epochs=200, verbose = 1, batch_size = 256, validation_split = 0.1, callbacks = keras_callbacks)
    h_model.save_weights('results/prec_1')
    h_model.save('prec_1', save_format = 'tf')

    predicted_train = h_model.predict(x_train)
    predicted_test = h_model.predict(x_test)
    predicted_val_cells = h_model.predict(x_val_cell)
    predicted_val_drugs = h_model.predict(x_val_drug)

    model_evaluation(target, predicted_train, verbose = True, subset_type = "train")
    model_evaluation(target_test, predicted_test, verbose = True, subset_type = "test")
    model_evaluation(target_validation_cell, predicted_val_cells, verbose = True, subset_type = "cell_val")
    model_evaluation(target_validation_drug, predicted_val_drugs, verbose = True, subset_type = "drug_val")
    
hypertuning()