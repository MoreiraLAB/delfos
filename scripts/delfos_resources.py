# -*- coding: utf-8 -*-

__author__ = "Piochi, LF"
__email__ = "lpiochi@cnc.uc.pt"
__group__ = "Data-Driven Molecular Design"
__group_leader__ = "Irina S. Moreira"
__project__ = "DELFOS"

# Change this variable to your root directory
DEFAULT_LOCATION = "/path/to/root/"


TRAIN_SPLIT = DEFAULT_LOCATION + "data/splits/train_features.csv"
TEST_SPLIT = DEFAULT_LOCATION + "data/splits/test_features.csv"
CELL_SPLIT = DEFAULT_LOCATION + "data/splits/cell_validation.csv"
DRUG_SPLIT = DEFAULT_LOCATION + "data/splits/drug_validation.csv"

TARGET_LOC = DEFAULT_LOCATION + "data/mycellsGDSC.h5"
LEAVE_CELL_LOC = DEFAULT_LOCATION + "data/splits/leave_out/cell_validation.txt"
LEAVE_DRUG_LOC = DEFAULT_LOCATION + "data/splits/leave_out/drug_validation.txt"

DRUG_SMILES_LOC = DEFAULT_LOCATION + "data/drug_smiles.csv"

SPLITS_LOCATION = DEFAULT_LOCATION + "data/processed/"
ORIGINAL_DATA_LOC = DEFAULT_LOCATION + "data/original/"

RANDOM_STATE = 42

def open_txt(input_file):
	opened_file = open(input_file, "r").readlines()
	return [x.replace("\n", "") for x in opened_file]


def model_evaluation(input_class, input_predictions, subset_type = "test", verbose = False, write_mode = False):
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error, \
                                    r2_score
    import math
    from scipy.stats import pearsonr, spearmanr
    if verbose == True:
        print("Currently evaluating:",subset_type, "\n")

    try:
        list_input_class = list(input_class.iloc[:,0])
    except:
        list_input_class = list(input_class)
    list_input_predictions = list(input_predictions)
    try:
        RMSE = math.sqrt(mean_squared_error(input_class, input_predictions))
    except:
        RMSE = 10000
    try:
        MSE = mean_squared_error(input_class, input_predictions)
    except:
        MSE = 10000
    try:
        Pearson, pval_Pearson = pearsonr([float(x) for x in list_input_class], [float(x) for x in list_input_predictions])
    except:
        Pearson = -1.0
    try:
        r2 = r2_score(input_class, input_predictions)
    except:
        r2 = -1.0
    try:
        MAE = mean_absolute_error(input_class, input_predictions)
    except:
        MAE = 10000
    try:
        Spearman, pval_Spearman = spearmanr(list_input_class, list_input_predictions)
    except:
        Spearman = -1.0
    if verbose == True:
        print("RMSE:", round(RMSE, 2), "\n",
               "MSE:" , round(MSE, 2), "\n",
            "Pearson:", round(Pearson, 2), "\n",
            "r^2:", round(r2, 2), "\n",
            "MAE:", round(MAE, 2), "\n",
            "Spearman:", round(Spearman, 2), "\n")
        
    if write_mode == True:
        output_file_name = DEFAULT_LOCATION + subset_type + ".csv"
        with open(output_file_name, "w") as output_file:
            output_file.write("Metric,Value\n")
            output_file.write("RMSE," + str(RMSE) + "\n")
            output_file.write("MSE," + str(MSE) + "\n")
            output_file.write("Pearson," + str(Pearson) + "\n")
            output_file.write("r^2," + str(r2) + "\n")
            output_file.write("MAE," + str(MAE) + "\n")
            output_file.write("Spearman," + str(Spearman) + "\n")
    return [RMSE, MSE, Pearson, r2, MAE, Spearman]