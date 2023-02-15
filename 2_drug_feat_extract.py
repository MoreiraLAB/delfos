# -*- coding: utf-8 -*-

__author__ = "Piochi, LF"
__email__ = "lpiochi@cnc.uc.pt"
__group__ = "Data-Driven Molecular Design"
__group_leader__ = "Irina S. Moreira"
__project__ = "DELFOS"

import pandas as pd
import drugtax
from rdkit import Chem
from mordred import Calculator, descriptors
from delfos_resources import DEFAULT_LOCATION, DRUG_SMILES_LOC

def fetch_drug_descriptors(drug_smiles = DRUG_SMILES_LOC):
    drugs = pd.read_csv(drug_smiles)
    smile_list = drugs["Ismiles"]
    # Extraction of Mordred drug descriptors
    features_mordred = pd.DataFrame()
    for i in range(len(smile_list)):
        compound = smile_list[i]
        
        mols = [Chem.MolFromSmiles(compound)]
        calc = Calculator(descriptors, ignore_3D = True)

        features_mordred = features_mordred.append(calc.pandas(mols))

    features_mordred["DRUG_NAME"] = drugs["Name"].values
    features_mordred.to_hdf(DEFAULT_LOCATION + "data/h5/mordred_feat.h5", "w", index = False)


    # Extraction of DrugTax drug descriptors
    features_drugtax = pd.DataFrame()
    for i in range(len(smile_list)):
        
        compound = smile_list[i]
        molecule = drugtax.DrugTax(compound)
        features_drugtax = features_drugtax.append(molecule.features, ignore_index = True)

    features_drugtax["DRUG_NAME"] = drugs["Name"].values
    features_drugtax.to_hdf(DEFAULT_LOCATION + "data/h5/drugtax_feat.h5", "w", index = False)

fetch_drug_descriptors()
