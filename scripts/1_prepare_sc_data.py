# -*- coding: utf-8 -*-

__author__ = "Piochi, LF"
__email__ = "lpiochi@cnc.uc.pt"
__group__ = "Data-Driven Molecular Design"
__group_leader__ = "Irina S. Moreira"
__project__ = "DELFOS"

import pandas as pd
from delfos_resources import DEFAULT_LOCATION

unique_cells = pd.read_csv(DEFAULT_LOCATION + "data/unique_cells.csv")
unique_cells['CELL_LINE_NAME'] = unique_cells['CELL_LINE_NAME'].str.upper()

# Calling the merged single-cell data
ssc = pd.read_csv(DEFAULT_LOCATION + "data/single-cell/merged_sc.csv")
ssc.rename(columns = {"Unnamed: 0": "barcode"}, inplace = True)

# Adding cell line names to the data
mcfar_id = pd.read_csv(DEFAULT_LOCATION + "data/single-cell/prep/classifications.csv")
mcfar_id = mcfar_id.iloc[:, :2]
mcfar_id["CELL_LINE_NAME"], mcfar_id["TISSUE"] = mcfar_id["singlet_ID"].str.split("_", 1).str
mcfar_id['CELL_LINE_NAME'] = mcfar_id['CELL_LINE_NAME'].str.upper()
mcfar_id['barcode'] = "mcfarland_" + mcfar_id['barcode']
mcfar_id = mcfar_id[["barcode", "CELL_LINE_NAME"]]

schnepp_annot = pd.read_csv(DEFAULT_LOCATION + "data/single-cell/prep/Annotation.txt", sep = "\t", header = None)
schnepp_annot.rename(columns = {0: "barcode", 1: "CELL_LINE_NAME"}, inplace = True)
schnepp_annot = schnepp_annot.loc[schnepp_annot['CELL_LINE_NAME'].str.contains('Sen')]
schnepp_annot.loc[schnepp_annot['CELL_LINE_NAME'].str.contains('DU145'), 'CELL_LINE_NAME'] = 'DU145'
schnepp_annot.loc[schnepp_annot['CELL_LINE_NAME'].str.contains('PC3'), 'CELL_LINE_NAME'] = 'PC3'
schnepp_annot['barcode'] = "schnepp_" + schnepp_annot['barcode']
mcfar_id = pd.concat([mcfar_id, schnepp_annot], axis = 0)

merg = ssc.merge(mcfar_id, how = "left", on = "barcode")
merg.loc[merg['barcode'].str.contains('OVCAR3'), 'CELL_LINE_NAME'] = 'OVCAR3'
merg.loc[merg['barcode'].str.contains('SCC25'), 'CELL_LINE_NAME'] = 'SCC25'
merg.loc[merg['barcode'].str.contains('MCF7'), 'CELL_LINE_NAME'] = 'MCF7'
merg["barcode"] = merg["CELL_LINE_NAME"]
merg = merg.iloc[:, :-1]
merg.rename(columns = {"barcode": "CELL_LINE_NAME"}, inplace = True)
merg = merg[merg['CELL_LINE_NAME'].isin(unique_cells["CELL_LINE_NAME"])]

# Splitting each cell line into a separate file
dfs = dict(tuple(merg.groupby('CELL_LINE_NAME')))
for i, df in dfs.items():
    df.to_csv(DEFAULT_LOCATION + "data/single-cell/individual/" + i + ".csv")