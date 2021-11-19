
# library(devtools)
# install_github("insongkim/concordance", dependencies=TRUE, force = T)

library(concordance)
library(readr)
library(tidyverse)

impo = read_csv("C:/Archivos/repos/impo_sectorial/scripts/data/resultados/importaciones_bk_pre_intro_matriz.csv")

impo_s = impo[sample(x = nrow(impo), size = 10000, replace = F), ] %>% 
  mutate(HS6 = as.character(HS6), 
         HS6 = case_when(length(HS6) == 5 ~ paste0("0", HS6),
                         T~ HS6) ,
         HS4 = substr(HS6, 1, 4) )


x = concord(impo_s$HS4, origin = "HS4", destination = "ISIC3.1", dest.digit = 4, all=T )
