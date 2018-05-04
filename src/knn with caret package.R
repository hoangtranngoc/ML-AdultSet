# KNN (k nearest neighbors) , training with caret package

library(caret)

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

#make output V15 variable as factor
adult_norm[, c(15)] <-sapply(adult_norm[,c(15)], as.factor) 
adult.test_norm[, c(15)] <-sapply(adult.test_norm[,c(15)], as.factor) 

# Train the model with preprocessing
#model_knn <- train(adult[, 1:14], adult_norm[, 15], method='knn', preProcess=c("center", "scale"))

# Predict values
#predictions<-predict.train(object=model_knn,adult.test_norm[,1:14], type="raw")

# Confusion matrix
#confusionMatrix(predictions,adult.test[,15])

#train the model with  crossvalidation and tuneLength
 trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
 set.seed(3333)
 model_knn <- train(adult_norm[, 1:14], adult_norm[, 15], method='knn',
 trControl=trctrl, tuneLength = 10) 

# Predict values
predictions<-predict.train(object=model_knn,adult.test_norm[,1:14], type="raw")

# Confusion matrix
confusionMatrix(predictions,adult.test_norm[,15])

#sources:
#https://www.datacamp.com/community/tutorials/machine-learning-in-r
#http://dataaspirant.com/2017/01/09/knn-implementation-r-using-caret-package/

