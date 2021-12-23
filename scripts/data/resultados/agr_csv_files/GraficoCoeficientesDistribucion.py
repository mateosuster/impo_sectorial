#!/usr/bin/env python
# coding: utf-8

# In[35]:


#Librería
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Directorio
import os
from pathlib import Path
import platform
directory = "impo_sectorial"
if platform.system()=='Windows':
    Path("D:/impo_sectorial/impo_sectorial").mkdir(parents=True, exist_ok=True)
    os.chdir("D:/impo_sectorial/impo_sectorial")
else: 
    Path('/home/nachengue/Escritorio/CEP/'+directory).mkdir(parents=True, exist_ok=True)
    os.chdir('/home/nachengue/Escritorio/CEP/'+directory)
print(os.getcwd())


# In[37]:


data_folder = "./scripts/data/resultados"

coef_path = "/agr_csv_files/mectra_agr_2021-12-23.csv"

desc_letra_path = "/desc_letra_propio.csv"

coef_df = pd.read_csv(data_folder+coef_path)
coef_df.columns = ['letra','anio','coeficiente']
desc_df = pd.read_csv(data_folder+desc_letra_path)


# In[38]:


merged = coef_df.merge(desc_df, left_on="letra", right_on="letra", how="left")


# In[39]:


gr = merged.groupby('desc').agg({'coeficiente':'mean'}).sort_values(by='coeficiente', ascending=False).reset_index()


# In[40]:


def break_lines(text,max_length):
    text = str(text).split(' ')
    lines = []
    if len(text)== 1:
        return text[0]
    else: 
        line = text[0]
        for i in range(1,len(text)):
            if len(line+(" ")+text[i])<=max_length:
                line = line+(" ")+text[i]
                
            else: 
                lines.append(line)
                line = text[i]
        if i==len(text)-1:
            lines.append(line)
    return "\n".join(lines)


# In[41]:


import matplotlib.font_manager as fm
import matplotlib.ticker as mtick
import seaborn as sns
sns.set_theme(style="whitegrid")

from tempfile import NamedTemporaryFile
import urllib

TOPN = 10

#Font Roboto Bold

roboto_bold = "https://github.com/googlefonts/roboto/blob/main/src/hinted/Roboto-Bold.ttf?raw=true"
response = urllib.request.urlopen(roboto_bold)
f = NamedTemporaryFile(delete=False, suffix='.ttf')
f.write(response.read())
f.close()
prop1 = fm.FontProperties(fname=f.name, size="large")

#Font Roboto Regular
roboto_reg = "http://github.com/googlefonts/roboto/blob/main/src/hinted/Roboto-Regular.ttf?raw=true"
response = urllib.request.urlopen(roboto_reg)
f = NamedTemporaryFile(delete=False, suffix='.ttf')
f.write(response.read())
f.close()
prop2 = fm.FontProperties(fname=f.name)

YLabs = gr.desc.head(TOPN).values.tolist()
YLabs = [break_lines(x, 20) for x in YLabs]

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(6, 15))

plot_data = gr.sort_values(by='coeficiente', ascending=False).head(TOPN)

sns.set_color_codes("pastel")
a = sns.barplot(x="coeficiente", y="desc", data=plot_data,
            color="#7030a0")


#ax.legend(ncol = 1, loc='lower right', borderaxespad=1)
a.set_ylabel("")
a.set_xlabel("Coeficiente de Distribución", fontproperties=prop2)
a.set_yticklabels(YLabs, fontproperties=prop1)
ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
# ax.set(xlim=(0, 24), ylabel="",fontproperties=prop
#       xlabel="Participación Ventas Intrafirma")
sns.despine(left=True, bottom=True)

plt.tight_layout()

plt.savefig(data_folder+"/agr_csv_files/TOP10-CoeficientesDistribucion.png", dpi=200,facecolor="w")

