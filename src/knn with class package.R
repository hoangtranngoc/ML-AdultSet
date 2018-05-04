# KNN k nearest neighbors, training on model with class package modelmo

library(caret)
library(class)

# Loading datasets
adult <- read.csv("~/GitHub/ML-AdultSet/data/adult.data", header=FALSE)
adult.test <- read.csv("~/GitHub/ML-AdultSet/data/adult.test", header=FALSE)

#tranforming data into numerical
adult[,c(1:15)]<-sapply(adult[,c(1:15)],as.numeric)
adult.test[,c(1:15)]<-sapply(adult.test[,c(1:15)],as.numeric)

#normalizing data
normalize <- function(newdataf, dataf){
  normalizeddataf <- newdataf 
  for (n in names(newdataf)){
    normalizeddataf[,n] <-  
      (newdataf[,n] - min(dataf[,n])) /  (max(dataf[,n]) -  min(dataf[,n]))
  } 
  return(normalizeddataf)
}
# applying noralization on data
adult_norm <- normalize(adult,adult)
adult.test_norm <- normalize(adult.test,adult)

