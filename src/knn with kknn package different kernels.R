# KNN, kknn package, training with multiple kernels, CV leave one out method

library(caret)
library(kknn)

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


#CV Leave one out method - training with kknn on different kernels and with 100 neighbors 
kknn.fit <- train.kknn(formula = V15 ~ ., data = adult_norm, kmax = 100,
                       kernel = c("optimal","rectangular", "inv", "gaussian", "triangular"), scale = TRUE)

#ploting results
plot(kknn.fit)

#Confusion matrix
kknn_pred <- predict(kknn.fit, adult.test_norm)
confusionMatrix(kknn_pred,adult.test_norm$V15)

#sources:
#http://rstudio-pubs-static.s3.amazonaws.com/349520_6c62f724297f4084abb48493c6f703a5.html
#https://cran.r-project.org/web/packages/kknn/kknn.pdf

