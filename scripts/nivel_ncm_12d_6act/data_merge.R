rm(list =ls())
gc()

getwd()
setwd("C:/Users/rAccess/Documents/")

library(foreign)

#un solo archivo
# data_131 = read.dbf("data/impo_diaria_12d/M_13s1_d12.DBF")

# lista de archivos 
file_list = list.files(path ="data/impo_diaria_12d" , pattern="*.DBF")
data_list <- list()
str(data_list)

#guardo archivos en lista
for (i in seq_along(file_list)) {
  filename = file_list[[i]]
  
  # Read data in
  df <- read.dbf(paste0("data/impo_diaria_12d/", filename) )
 
  # Guardo DF en la lista
  # data_list[[date]] <- df
  name = paste0("data_", i)
  data_list[[name]] <- df
  cat("ya cargó", i, "datasets \n")
  rm(df)
}


# asigno archivos
for (i in names(data_list)){
  # i <<- data.frame(data_list[[i]])
  assign(x = i, 
         value = data.frame(data_list[[i]])#%>% 
         # seleccion_vars()
  )
}

dfs = sapply(.GlobalEnv, is.data.frame) 
super_data = do.call(rbind, mget(names(dfs)[dfs]))

rm(list=ls(pattern="^data_"))

str(super_data)

# write.csv(super_data, file ="impo_sectorial/scripts/data/M_2013a2017_d12.csv" ,row.names = F)

