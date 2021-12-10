

rm(list=ls())
gc()

#Librerias
library(data.table)
library(tidyverse)

#Setea ruta según el OS en el que se trabaje. 
VMrepo.full.path <- "D:/impo_sectorial/impo_sectorial"  #poner la ruta del repo en el VM
switch ( Sys.info()[['sysname']],
         Windows = {directory.root  <-  VMrepo.full.path}, #AWS
         Linux   = {directory.root  <-  "~/Escritorio/CEP/mectra_desarrollo"} #ruta para Nacho
)

#Defining Working Directory
setwd( directory.root )

#csv folder
csv_folder <- "./scripts/data/bases_mectra_csv_files"

#Listing rda files
files <- list.files(path = csv_folder,
                    pattern = "*.csv", full.names = T)

#Columns subset
cols <- c("clae6","cuil","zona","mes")

frames <- data.table()
for (f in files){
  name <- str_extract(f, '[m][0-9]+')
  cat("Processing...", name,"\n")
  df <- fread(f, select = cols)
  df <- df[, anio:= as.integer(str_sub(mes,1,4))]
  agr <- df[, .(trabajadores=uniqueN(cuil)), by= c("clae6","zona","anio")]
  agr <- agr[, clae6:= str_pad(clae6, 6, pad = "0")]
  frames <- setDT(rbind(frames, agr))
  cat("Done...", name,"\n")
}

#csv agr folder
csv_agr_folder <- "./scripts/data/resultados/agr_csv_files"
dir.create(file.path(directory.root, csv_agr_folder), showWarnings = FALSE)


fwrite(frames, paste0(csv_agr_folder,"/","mectra_agr_", Sys.Date(),".csv"))

rm(list=ls())
gc()
