
# Library & prepare training, testing data
library(C50)
colClasses=c("integer", "factor", "integer", "factor", "integer", "factor", "factor", "factor", "factor", "factor",
             "integer", "integer", "integer", "factor", "factor")

url_train <- "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
data_train <- read.table( file=url_train, header=FALSE, colClasses=colClasses, sep=",", strip.white=TRUE )

url_test <- "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
data_test <- read.table( file=url_test, skip=1, header=FALSE, colClasses=colClasses, sep=",", strip.white=TRUE)

#remove trailing dot
data_test[,15] <- factor(sub("\\.", "", data_test[,15]))



# Print to see sample data
head(data_train,5)
head(data_test,5)



# Train model
set.seed(1234)

library(caret)
fitControl <- trainControl(method = "cv", number = 5)
mdl<- train(V15 ~ ., data=data_train, trControl=fitControl,method="C5.0")
mdl



p <- predict( mdl, data_test)
confusionMatrix(p, data_test$V15)




