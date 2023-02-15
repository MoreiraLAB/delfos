# -*- coding: utf-8 -*-

__author__ = "Piochi, LF"
__email__ = "lpiochi@cnc.uc.pt"
__group__ = "Data-Driven Molecular Design"
__group_leader__ = "Irina S. Moreira"
__project__ = "DELFOS"

import numpy as np
import pandas as pd
import pubchempy as pcp
from delfos_resources import DEFAULT_LOCATION, ORIGINAL_DATA_LOC, DRUG_SMILES_LOC


# The CCLE and GDSC datasets were filtered to contain only the cell lines that we had scRNA-seq data available.
single_cell_lines = pd.read_csv(ORIGINAL_DATA_LOC + "single_cell_lines.csv")
ccle_ref_table = pd.read_csv(ORIGINAL_DATA_LOC + "ccle_sample_info.csv")
gdsc2 = pd.read_csv(ORIGINAL_DATA_LOC + "GDSC2_24Jul22.csv")
gdsc2.CELL_LINE_NAME = gdsc2.CELL_LINE_NAME.replace("-", "", regex = True)

single_ccle_gdsc = ccle_ref_table.merge(single_cell_lines).merge(gdsc2, left_on = "Sanger_Model_ID", right_on = "SANGER_MODEL_ID")
single_ccle_gdsc.rename(columns = {"CELL_LINE_NAME_y":"CELL_LINE_NAME"}, inplace = True)

"""
# The following codde was added to address two errors that were identified in the GDSC2 dataset and make them uniform with the other data. 
# First, one in which the supposed cell line name is NCIH322M, # but the Sanger Model ID and COSMIC IDs of the sample are of the cell line NCIH322. 
# The other is where the cell line "786-O" is mistakingly named "786-0".
"""
single_ccle_gdsc.CELL_LINE_NAME = single_ccle_gdsc.CELL_LINE_NAME.replace("NCIH322M", "NCIH322", regex = True)
single_ccle_gdsc.CELL_LINE_NAME = single_ccle_gdsc.CELL_LINE_NAME.replace("786-0", "786-O", regex = True)
single_ccle_gdsc['CELL_LINE_NAME'] = single_ccle_gdsc['CELL_LINE_NAME'].str.upper()

# Dataset containing cell lines in common 
single_ccle_gdsc = single_ccle_gdsc[["CELL_LINE_NAME", "DepMap_ID", "DRUG_NAME", "LN_IC50"]]

# Fetch SMILES from drug names
drugnames = list(single_ccle_gdsc["DRUG_NAME"].unique())

def fetch_smiles(drug_list):
    smiles = np.array(['Name', 'iupac', 'Csmiles', 'Ismiles'])   
    for i in range(len(drug_list)):
        compound = drug_list[i]
        results = pcp.get_compounds(compound, 'name')
        try:
            smiles = np.vstack((smiles, [compound, results[0].iupac_name, results[0].canonical_smiles, results[0].isomeric_smiles]))
        except IndexError:
            smiles = np.vstack((smiles, [compound, 'NF', 'NF', 'NF']))
    
    smiles = pd.DataFrame(smiles[1:], columns = smiles[0])
    return smiles

smiles = fetch_smiles(drugnames)
    
# Removing drugs from which no SMILES were found
have_smiles = smiles[~(smiles["Ismiles"] == "NF")]
have_smiles.to_csv(DRUG_SMILES_LOC, index = False)
single_ccle_gdsc = single_ccle_gdsc[single_ccle_gdsc["DRUG_NAME"].isin(have_smiles["Name"])]

cells = single_ccle_gdsc.groupby(["CELL_LINE_NAME", "DepMap_ID"]).size().reset_index()
cells = cells.iloc[: , :2]
cells["CELL_LINE_NAME"].to_csv(DEFAULT_LOCATION + 'data/unique_cells.csv', index = False)
drugs = pd.DataFrame(single_ccle_gdsc["DRUG_NAME"].unique())
drugs.rename(columns = {0 : "DRUG_NAME"}, inplace = True)
drugs.to_csv(DEFAULT_LOCATION + 'data/unique_drugs.csv', index = False)

single_ccle_gdsc = single_ccle_gdsc.drop("DepMap_ID", axis = 1)
single_ccle_gdsc.to_hdf(DEFAULT_LOCATION + 'data/mycellsGDSC.h5', "w", index = False)

# Filtering and retrieving CCLE datasets:
def generate_CCLE(ccle_input, omics_type):
    ccle = pd.read_csv(ccle_input)
    if omics_type == "expression" or omics_type == "copynumber":
        ccle.rename(columns = {"Unnamed: 0":"DepMap_ID"}, inplace = True)
        ccle = ccle.merge(cells, on = "DepMap_ID")
        ccle.insert(1, 'CELL_LINE_NAME', ccle.pop("CELL_LINE_NAME"))
        ccle = ccle.iloc[:, 1:]
    else:
        ccle = ccle.merge(cells, on = "CELL_LINE_NAME")
        ccle = ccle.drop_duplicates()
        ccle = ccle.drop("DepMap_ID", axis = 1)
    ccle.to_hdf(DEFAULT_LOCATION + "/data/h5/CCLE/CCLE_" + omics_type + ".h5", "w", index = False)

ccle_original = {
    "chromatin" : ORIGINAL_DATA_LOC + "CCLE/CCLE_chromatin.csv",
    "copynumber" : ORIGINAL_DATA_LOC + "CCLE/CCLE_copynumber.csv",
    "expression" : ORIGINAL_DATA_LOC + "CCLE/CCLE_expression.csv",
    "methylation" : ORIGINAL_DATA_LOC + "CCLE/CCLE_methylation.csv",
    "mirna" : ORIGINAL_DATA_LOC + "CCLE/CCLE_mirna.csv"
}

for key in ccle_original:
    generate_CCLE(ccle_input = ccle_original[key], omics_type = key)
