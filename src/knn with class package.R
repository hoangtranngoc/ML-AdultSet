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

#storing class labels in factor vectors, divided to the training and test datasets
adult_norm_labels <- adult_norm[, 15]
adult.test_norm_labels <- adult.test_norm[, 15]

#make output V15 variable as factor
adult_norm[, c(15)] <-sapply(adult_norm[,c(15)], as.factor) 
adult.test_norm[, c(15)] <-sapply(adult.test_norm[,c(15)], as.factor) 

#training a model on the data
# we can either store factors in labels as above or just use cl=adult_norm[,15]
knn_model <- knn3(V15 ~., adult_norm,k=21)

#evaluating model performance
knn_pred <- predict(knn_model, adult.test_norm, type="class")
confusionMatrix(knn_pred,adult.test_norm_labels)

#sources:
#Lantz, B. (2013). Machine Learning with R. Packt Publishing.
#https://daviddalpiaz.github.io/stat432sp18/supp/knn_class_r.html