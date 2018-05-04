#SVM, kernlab package

library(caret)
library(kernlab)

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

#training with ksvm, rbfdot kernel, sigma and cross validation
ksvm_model <- ksvm( V15 ~., type="C-svc", data= adult_norm, kernel = "rbfdot",
                    C=2,kpar=list(sigma=0.6),cross=10,prob.model=TRUE)

#evaluating the model, confusion matrix
ksvm_pred <- predict(ksvm_model, adult.test_norm)
confusionMatrix(ksvm_pred,adult.test_norm$V15)  

#sources: 
#https://www.rdocumentation.org/packages/kernlab/versions/0.9-25/topics/ksvm 