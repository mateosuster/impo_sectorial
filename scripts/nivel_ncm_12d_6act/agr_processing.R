rm(list=ls())
gc()

#Librerias
library(data.table)
library(tidyverse)
library(readxl)

#Setea ruta según el OS en el que se trabaje. 
VMrepo.full.path <- "D:/impo_sectorial/impo_sectorial"  #poner la ruta del repo en el VM
switch ( Sys.info()[['sysname']],
         Windows = {directory.root  <-  VMrepo.full.path}, #AWS
         Linux   = {directory.root  <-  "~/Escritorio/CEP/mectra_desarrollo"} #Nacho
)

#Defining Working Directory
setwd( directory.root )

#csv folder
switch ( Sys.info()[['sysname']],
         Windows = {csv_folder <-  "./scripts/data/bases_mectra_csv_files"}, #AWS
         Linux   = {csv_folder  <-  "./bases_mectra_csv_files"} #Nacho
)


#Listing rda files
files <- list.files(path = csv_folder,
                    pattern = "*.csv", full.names = T)


#MetaData Folder
switch ( Sys.info()[['sysname']],
         Windows = {data.folder <-  "./scripts/data"}, #AWS
         Linux   = {data.folder <- "./data"} #Nacho
)


#CLAE to CIIU_LETRA
clae_ciiu_letra.propios.path <- paste0(data.folder,"/resultados/dic_clae_ciiu_propio.csv")
# clae_ciiu_letra <- fread(clae_ciiu_letra.propios.path, colClasses = c("character","character",
#                                                         "character","character",
#                                                         "character","character",
#                                                         "character"),
#                       select = c("clae6","propio_letra_2"))
clae_ciiu_letra <- fread(clae_ciiu_letra.propios.path, select = c("clae6","propio_letra_2"))
clae_ciiu_letra <- clae_ciiu_letra[ , clae6:=str_pad(clae6, 6, pad = "0")]
clae_ciiu_letra <- clae_ciiu_letra[clae6!="000000",]

#ZONA to PROVINCIA
zona.prov.path <- paste0(data.folder,"/zona_loc.csv")
zona.prov <- fread(zona.prov.path)
zona.prov <- zona.prov[, zona:= str_pad(zona, 2, pad = "0")]
zona.prov <- zona.prov[, zona_prov:= str_trim(zona_prov)]


#DENSIDAD POR PROVINCIA
densidad.path <- paste0(data.folder,"/densidad.csv")
densidad <- fread(densidad.path, sep=";", dec=",")
densidad <- densidad[ ,Provincia:=  str_trim(Provincia)]

#Columns subset
cols <- c("cuit","clae6","cuil","zona","mes")

Ntrab.prom.mens_by_anio.zona.clae <- function(file.list){
  
  frames <- data.table()
  
  #for (f in file.list){
  for (f in files){
    name <- str_extract(f, '[m][0-9]+')
    cat("Processing...", name,"\n")
    df <- fread(f, select = cols)
    df <- df[, anio:= as.integer(str_sub(mes,1,4))]
    agr <- df[, .(Ncuil = uniqueN(cuil)), by = c("cuit","clae6","zona","mes","anio")] #Cuantos empleados hay por empresa, clae y zona en un mes/anio determinado
    agr <- agr[, clae6:= str_pad(clae6, 6, pad = "0")]
    agr <- agr[clae6!="000000"] #saco los clae6==0 --> no tienen asignación de actividad
    
    #join con clae.ciiu
    agr <- inner_join(agr, clae_ciiu_letra, by= c("clae6"="clae6"))
    
    #Join con zona.prov
    agr <- inner_join(agr, zona.prov, by= c("zona"="zona"))
    agr <- agr[, .(SumMes = sum(Ncuil)), by= c("propio_letra_2","zona_prov","anio")] #Suma de los cuils únicos
    frames <- setDT(rbind(frames, agr))
    cat("Done...", name,"\n")
    }
  
  #Calculo pŕomedio mensual 
  frames <- frames[, .(Ntrab.prom_mens = sum(SumMes)/12), by= c("propio_letra_2","zona_prov","anio")] #Acá siempre vamos a contar con anios de 12 meses
  
  #Realizo join con densidad
  names(frames) <- c("propio_letra_2","provincia","anio","ntrab_pmens")
  frames <- left_join(frames, densidad, by= c("provincia"="Provincia")) %>% 
                   select(propio_letra_2, provincia, anio, ntrab_pmens, densidad)
  
  
  #Hago la agregación
  frames <- frames %>% group_by(propio_letra_2,anio) %>% mutate(prop = ntrab_pmens/sum(ntrab_pmens))
  setDT(frames)
  
  frames <- frames[, dens_prop:= densidad * prop]
  frames <- frames %>% group_by(propio_letra_2, anio) %>% summarise(dens_prop = sum(dens_prop))
  frames <- frames %>% mutate(inv.dens_prop = 1/dens_prop)
  sum.inv.by.year <- frames %>% group_by(anio) %>% summarise(sum.inv.dens_prop = sum(inv.dens_prop))
  frames <- inner_join(frames, sum.inv.by.year, by= c("anio"="anio"))
  
  frames <- frames %>% mutate(coeficiente = inv.dens_prop/sum.inv.dens_prop)
  
  #Me quedo con lo que me interesa nomás. 
  frames <- frames %>% select(propio_letra_2, anio, coeficiente)
  return(frames)
}

frames <- Ntrab.prom.mens_by_anio.zona.clae(files)

#csv output folder
switch ( Sys.info()[['sysname']],
         Windows = {csv_agr_folder <- "./scripts/data/resultados/agr_csv_files"}, #AWS
         Linux   = {csv_agr_folder <- "./agr_csv_files"} #Nacho
)

dir.create(file.path(directory.root, csv_agr_folder), showWarnings = FALSE)


fwrite(frames, paste0(csv_agr_folder,"/","mectra_agr_", Sys.Date(),".csv"))

rm(list=ls())
gc()



