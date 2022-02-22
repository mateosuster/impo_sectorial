

import pandas as pd
import os
os.chdir("D:/impo_sectorial/impo_sectorial/scripts/nivel_ncm_12d_6act")


xls = pd.ExcelFile('../data/Patentamientos.xlsx')
df1 = pd.read_excel(xls, 0)
df2 = pd.read_excel(xls, 1)
