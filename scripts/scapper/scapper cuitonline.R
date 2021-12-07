
install.packages("openxlsx")


#Librerias
library(lubridate)
library(tidyverse)
library(data.table)
library(rvest)
library(openxlsx)
#Datos 

# data <- fread("C:/Users/Administrator/Desktop/CEP/Codigos/actividades principales CUITs IMPO - Igal/segundo pedido/cuits_impor.csv")
data <- fread("D:/impo_sectorial/impo_sectorial/scripts/scapper/cuits_2013a2017.csv")

setnames(data,'x','V1')
#Cantidad de alquileres por tipo
URL <- 'https://www.cuitonline.com/detalle/'

#Iteracion sobre paginas para conseguir los alquileres con expensas
result_DF1 <- tibble()
for (i in 2:length(data$V1)){
  tryCatch(
    result <- map_df(data$V1[i], ~{
      paste(URL,.x,'/',sep='') %>%
        read_html() %>% 
        html_nodes('.p_info') -> tmp
      datos_persona <- tmp %>% 
        html_text(trim=TRUE)
      cuit_persona <- data$V1[i]
      data.frame(datos_persona,cuit_persona,
                 stringsAsFactors = F)
    }),
    error=function(e){cat('ERROR :',conditionMessage(e),'\n')}
  )
  result_DF1 <- rbind(result_DF1,result)
  print(i)
}
result_DF2 <- result_DF1 %>% distinct()
result_DF2 <- result_DF2 %>% mutate(Actividades = str_extract(datos_persona,'Actividades: (.*?) [a-z]|Actividades: (.*?)$|Actividad: (.*?)$'))
result_DF2 <- result_DF2 %>% mutate(Actividades_2 = str_extract_all(Actividades,'#[0-9]+ (.*?) #|[0-9]+ (.*?) #|[0-9]+ .*$'))
result_DF2 <- result_DF2 %>% unnest(Actividades_2)
result_DF2 <- result_DF2 %>% mutate(Fecha_actividad = str_extract(Actividades_2,'\\[[0-9]+\\/[0-9]+\\]'),
                                    Clae6 = str_extract(Actividades_2,'\\] [0-9]+ -'))
result_DF2 <- result_DF2 %>% 
  mutate(Fecha_actividad = str_remove(Fecha_actividad,'\\['),
         Fecha_actividad = str_remove(Fecha_actividad,'\\]'),
         Fecha_actividad = paste('01/',Fecha_actividad,sep=''),
         Fecha_actividad = dmy(Fecha_actividad))
result_DF2 <- result_DF2 %>% 
  mutate(Clae6 = str_extract(Clae6,'[0-9]+')) 
result_DF2 <- result_DF2 %>% 
  mutate(Clae6_desc = str_extract(Actividades_2,'[A-Za-z]+.*'))
result_DF2 <- result_DF2 %>% 
  mutate(Numero_actividad_cuit = str_extract(Actividades_2,'[0-9]+'))
result_DF2 <- result_DF2 %>% 
  mutate(Numero_actividad_cuit = as.double(Numero_actividad_cuit)) %>% 
  group_by(cuit_persona) %>% 
  mutate(Cantidad_actividades_cuit = max(Numero_actividad_cuit))

result_DF2_simplificado <- result_DF2 %>% select(CUIT = cuit_persona,Fecha_actividad,Clae6,Clae6_desc,Numero_actividad_cuit,Cantidad_actividades_cuit)
result_DF2_simplificado <- result_DF2_simplificado %>% mutate(CUIT = as.double(CUIT))
#library(googlesheets4)
#sheet_write(result_DF2_simplificado)
#fwrite(result_DF2,file = 'Datos scrapeados Igal - CuitOnline.csv',sep = ';')

write.csv(result_DF2_simplificado, file ="clae_cuit_2013 a 2017.csv", sep =";")
