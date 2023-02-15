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
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, ReLU, BatchNormalization, Add, Concatenate, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from delfos_resources import SPLITS_LOCATION, model_evaluation

np.random.seed(42)
random.seed(42)

# Defining whether to use scRNA-seq data or not and, if using, the number of single cells per cell line that will be used
use_sc = True
single_cells = 10
    
# Load Training datasets
def load_datasets(mode = "train", use_sc_data = True):
    
    chrom = pd.read_csv(SPLITS_LOCATION + "bulk/chromatin_" + mode + "_features.csv")
    cn = pd.read_csv(SPLITS_LOCATION + "bulk/copynumber_" + mode + "_features.csv")
    expr = pd.read_csv(SPLITS_LOCATION + "bulk/expression_" + mode + "_features.csv")
    meth = pd.read_csv(SPLITS_LOCATION + "bulk/methylation_" + mode + "_features.csv")
    mirna = pd.read_csv(SPLITS_LOCATION + "bulk/mirna_" + mode + "_features.csv")
    mordred = pd.read_csv(SPLITS_LOCATION + "drug/mordred_" + mode + "_features.csv")
    drugtax = pd.read_csv(SPLITS_LOCATION + "drug/drugtax_" + mode + "_features.csv")

    featuring = [chrom, cn, expr, meth, mirna, mordred, drugtax]
    target = featuring[0]["LN_IC50"]

    # CREATE FUNCTION FOR LOAD_SC
    # Load single-cell data
    def load_sc(feature_list, mode = mode):
        cell_line_list = list(feature_list[0]["CELL_LINE_NAME"].unique())
        sc_list = {}
        for i in range(single_cells):
            sc_list['sc_' + str(i)] = []
        
        if mode == "valcell":
            for sel_cel in cell_line_list:
                df = pd.read_hdf(SPLITS_LOCATION + "single-cell/individual_valid/" + sel_cel + ".h5")
                i = 0
                for index, row in df.iterrows():
                    dam = pd.DataFrame(row).T
                    sc_list['sc_' + str(i)].append(dam)
                    i += 1

        else:
            for sel_cel in cell_line_list:
                df = pd.read_hdf(SPLITS_LOCATION + "single-cell/individual_tt/" + sel_cel + ".h5")
                i = 0
                for index, row in df.iterrows():
                    dam = pd.DataFrame(row).T
                    sc_list['sc_' + str(i)].append(dam)
                    i += 1
            
        ind_sing = []
        for i in sc_list:
            sing = pd.concat(sc_list[i], ignore_index = True)
            sing = feature_list[0].iloc[:, :4].merge(sing)
            ind_sing.append(sing)

        return ind_sing

    if use_sc_data == True:
        sc_data = load_sc(feature_list = featuring)
        featuring += sc_data

    for i in range(len(featuring)):
        featuring[i] = featuring[i].iloc[:, 4:]
        featuring[i] = np.asarray(featuring[i]).astype('float32')

    return featuring, target

### Defining DELFOS
class delfos(Model):

    ### Define layers in def__init__:

    def __init__(self, hidden_layer_number, hidden_layer_size_cells, hidden_layer_size_drugs, hidden_layer_size_single,
                 activation_function = "relu",
                 add_dropout_cells = False,
                 add_dropout_drugs = False,
                 dropout_rate_cells = 0.5,
                 dropout_rate_drugs = 0.3,
                 use_single_cell = True,
                 features = ['CCLE_chromatin', 'CCLE_copynumber', 'CCLE_expression','CCLE_methylation', 'CCLE_miRNA', 'Mordred', 'DrugTax']): 
        super(delfos, self).__init__()
        
        # Defining network architecture variables
        self.features = features
        self.hidden_layer_number = hidden_layer_number
        self.hidden_layer_size_cells = hidden_layer_size_cells
        self.hidden_layer_size_drugs = hidden_layer_size_drugs
        self.hidden_layer_size_single = hidden_layer_size_single
        self.add_dropout_cells = add_dropout_cells
        self.add_dropout_drugs = add_dropout_drugs
        self.dropout_rate_cells = dropout_rate_cells
        self.dropout_rate_drugs = dropout_rate_drugs
        self.use_single_cell = use_single_cell
        
        self.overall_model = {}
        
        # Defining feature blocks for CCLE features
        for feature_block in features[:5]:
            branch = Sequential()
            branch.add(Dense(units = hidden_layer_size_cells, input_shape = (input_features_size[feature_block], ), \
                                        activation = activation_function, \
                                        kernel_regularizer = regularizers.l1_l2(l1=1e-5, l2=1e-4), \
                                        bias_regularizer = regularizers.l2(1e-4), \
                                        activity_regularizer = regularizers.l2(1e-5)))
                                        
            for hidden_layer in range(hidden_layer_number - 1):
                if add_dropout_cells == True:
                    branch.add(Dropout(dropout_rate_cells))
                branch.add(Dense(units = hidden_layer_size_cells, input_shape = (input_features_size[feature_block], ), activation = activation_function, \
                                        kernel_regularizer = regularizers.l1_l2(l1=1e-5, l2=1e-4), \
                                        bias_regularizer = regularizers.l2(1e-4), \
                                        activity_regularizer = regularizers.l2(1e-5)))
            branch.add(Dense(1, activation = 'linear'))
            self.overall_model[feature_block] = branch
        
        # Defining feature blocks for drug features
        for feature_block in features[5:7]:
            branch = Sequential()
            branch.add(Dense(units = int(hidden_layer_size_drugs), input_shape = (input_features_size[feature_block], ), \
                                        activation = activation_function, \
                                        kernel_regularizer = regularizers.l1_l2(l1=1e-5, l2=1e-4), \
                                        bias_regularizer = regularizers.l2(1e-4), \
                                        activity_regularizer = regularizers.l2(1e-5)))
                                        
            for hidden_layer in range(hidden_layer_number - 1):
                if add_dropout_drugs == True:
                    branch.add(Dropout(dropout_rate_drugs))
                branch.add(Dense(units = int(hidden_layer_size_drugs), input_shape = (input_features_size[feature_block], ), activation = activation_function, \
                                        kernel_regularizer = regularizers.l1_l2(l1=1e-5, l2=1e-4), \
                                        bias_regularizer = regularizers.l2(1e-4), \
                                        activity_regularizer = regularizers.l2(1e-5)))
            branch.add(Dense(1, activation = 'linear'))
            self.overall_model[feature_block] = branch
            
        # Defining feature bloocks for scRNA-seq features
        if use_single_cell == True:
            for i in range(single_cells):
                self.features.append("sc" + str(i))
            for feature_block in features[7:]:
                branch = Sequential()
                branch.add(Dense(units = int(hidden_layer_size_single), input_shape = (input_features_size['sc0'], ), \
                                            activation = activation_function, \
                                            kernel_regularizer = regularizers.l1_l2(l1=1e-5, l2=1e-4), \
                                            bias_regularizer = regularizers.l2(1e-4), \
                                            activity_regularizer = regularizers.l2(1e-5)))
                                            
                for hidden_layer in range(hidden_layer_number - 1):
                    branch.add(Dense(units = int(hidden_layer_size_single), input_shape = (input_features_size['sc0'], ), activation = activation_function, \
                                            kernel_regularizer = regularizers.l1_l2(l1=1e-5, l2=1e-4), \
                                            bias_regularizer = regularizers.l2(1e-4), \
                                            activity_regularizer = regularizers.l2(1e-5)))
                branch.add(Dense(1, activation = 'linear'))
                self.overall_model[feature_block] = branch

        self.merge = Add()
        self.output_neuron = Dense(1, activation = 'linear')

    
    ### Defining model connections
    def call(self, inputs):
        branch = []
        for index, feature_block in enumerate(self.features):
            data = inputs[index]
            data = self.overall_model[feature_block](data)
            branch.append(data)
                
        merged = self.merge(branch)
        output = self.output_neuron(merged)
        return output



# Load datasets
featuring_train, target = load_datasets(mode = "train", use_sc_data = use_sc)
featuring_test, target_test = load_datasets(mode = "test", use_sc_data = use_sc)
featuring_validation_cell, target_validation_cell = load_datasets(mode = "valcell", use_sc_data = use_sc)
featuring_validation_drug, target_validation_drug = load_datasets(mode = "valdrug", use_sc_data = use_sc)

# Defining input shapes for implementation in the model
input_features_size = {
    'CCLE_chromatin': featuring_train[0].shape[1],
    'CCLE_copynumber': featuring_train[1].shape[1],
    'CCLE_expression': featuring_train[2].shape[1], 
    'CCLE_methylation': featuring_train[3].shape[1],  
    'CCLE_miRNA': featuring_train[4].shape[1],
    'Mordred': featuring_train[5].shape[1],
    'DrugTax': featuring_train[6].shape[1]}
if use_sc == True:
    input_features_size['sc0'] = featuring_train[7].shape[1]

def hypertuning():
    class MyHyperModel(kt.HyperModel):
        def build(self, hp):
            lr_values = [1e-3, 1e-4]
            model = delfos(hidden_layer_number = hp.Int('hl_number', 7, 14, step = 1), 
                            hidden_layer_size_cells = hp.Int('units_cells', min_value = 20, max_value = 250, step = 4), 
                            hidden_layer_size_drugs= hp.Int('units_drugs', min_value = 20, max_value = 250, step = 4),
                            hidden_layer_size_single= hp.Int('units_sc', min_value = 20, max_value = 250, step = 4), 
                            add_dropout_cells = False,
                            add_dropout_drugs = False,
                            use_single_cell = True)
            
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = hp.Choice('learning_rate', values = lr_values)), 
                          loss= "mse", metrics=[tf.keras.metrics.RootMeanSquaredError(), "mean_absolute_error"])
            return model
    
        def fit(self, hp, model, *args, **kwargs):
            return model.fit(
                *args,
                batch_size=hp.Choice("batch_size", [32, 64, 128, 256]),
                **kwargs,
            )
    
    
    tuner = kt.Hyperband(MyHyperModel(),
                    objective='val_loss',
                    max_epochs=100,
                    factor = 3,
                    seed = 42,
                    project_name= "delfos_1")
    
    keras_callbacks = [
      tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode = 'min', patience = 10, restore_best_weights = True),
      tf.keras.callbacks.ReduceLROnPlateau(factor = 0.1, min_lr = 1e-5, patience=5)]
    
    tuner.search(featuring_train, target, epochs = 200, callbacks = keras_callbacks, verbose = 2, validation_split = 0.1)
    
    best_hp = tuner.get_best_hyperparameters()[0]
       
    h_model = tuner.hypermodel.build(best_hp)
    h_model.fit(featuring_train, target, epochs = 200, verbose = 1)
    h_model.save_weights('results/delfos_1')
    h_model.save('delfos_1', save_format = 'tf')

hypertuning()