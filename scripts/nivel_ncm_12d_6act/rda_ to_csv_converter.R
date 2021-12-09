#Cambiar las rutas para correr en la nube correctamente. 


#Librerias
library(data.table)
library(dplyr)
library(stringr)


switch ( Sys.info()[['sysname']],
         Windows = {directory.root  <-  "C:/Asus/Desktop/CEP/mectra_desarrollo"},
         Linux   = {directory.root  <-  "~/Escritorio/CEP/mectra_desarrollo"} 
)

#Defining Working Directory
setwd( directory.root )

#rda folder
historic <- FALSE
rda_folder <- ifelse(historic,"./historico_rda","./bases_mectra_resumidas_rda_files")

#Listing rda files
files <- list.files(path = rda_folder,
                    pattern = "*.rda", full.names = T)

#Columns subset
cols <- c("cuit","clae6","cuil","act_trab","zona","codprov","remuneracion",
          "sueldo","sac","conss","conos","madconss","conrenatre","codobsoc",
          "cond_cuil","sit_cuil","apobliss","apobsoc","convencionado",
          "modalidad","mes")

#csv output folder
csv_folder <- ifelse(historic,"./historico_csv_files","./bases_mectra_resumidas_csv_files")
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



