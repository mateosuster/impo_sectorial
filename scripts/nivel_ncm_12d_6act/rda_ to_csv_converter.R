# Entre los datasets contamos con: 
#       * datasets gigantes de muchas filas y columnas (denominados históricos)
#       * datasets resumidos, que solo cuentan con una preselección de columnas (denominados resumidos)
# El script permite elegir el primer conjunto de datasets seteando la variable "historic" como TRUE. 
# Por default la variable "historic" es FALSE y entonces elige los datasets resumidos. 
# Es necesario crear previamente un sistema de ficheros en la ruta del repo
#       * historico_rda/
#       * bases_mectra_resumidas_rda_files/
# El resto de los ficheros los crea el mismo script. 

#Librerias
library(data.table)
library(tidyverse)

VMrepo.full.path <- "D:/impo_sectorial/impo_sectorial" #poner la ruta del repo en el VM
switch ( Sys.info()[['sysname']],
         Windows = {directory.root  <-  VMrepo.full.path}, 
         Linux   = {directory.root  <-  "~/Escritorio/CEP/mectra_desarrollo"} #ruta para Nacho
)

#Defining Working Directory
setwd( directory.root )

#rda folder
historic <- TRUE
rda_folder <- ifelse(historic,"./scripts/data/historico_rda","./scripts/data/bases_mectra_resumidas_rda_files")

#Listing rda files
files <- list.files(path = rda_folder,
                    pattern = "*.rda", full.names = T)

#Columns subset
cols <- c("cuit","clae6","cuil","act_trab","zona","codprov","remuneracion",
          "sueldo","sac","conss","conos","madconss","conrenatre","codobsoc",
          "cond_cuil","sit_cuil","apobliss","apobsoc","convencionado",
          "modalidad","mes")

#csv output folder
csv_folder <- "./scripts/data/bases_mectra_csv_files"
dir.create(file.path(directory.root, csv_folder), showWarnings = FALSE)

for(rda in files){
  name <- str_extract(rda,'[m][0-9]+')
  cat("Saving to csv...",name,'\n')
  load(rda)
  setDT(data)
  data <- data[, ..cols]
  
  fwrite(data,paste0(csv_folder,"/","mectra_",name,".csv"))
  cat('Already saved: ',name,'\n')
}
rm(list=ls())
gc()



