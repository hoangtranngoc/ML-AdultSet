
# Library & prepare training, testing data
library(caret)
library(xgboost)
colClasses=c("integer", "factor", "integer", "factor", "integer", "factor", "factor", "factor", "factor", "factor",
             "integer", "integer", "integer", "factor", "factor")

url_train <- "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
data_train <- read.table( file=url_train, header=FALSE, colClasses=colClasses, sep=",", strip.white=TRUE )

url_test <- "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
data_test <- read.table( file=url_test, skip=1, header=FALSE, colClasses=colClasses, sep=",", strip.white=TRUE)

#remove trailing dot
data_test[,15] <- factor(sub("\\.", "", data_test[,15]))

# inTrain <- createDataPartition(y=data_train$V15, p=0.7, list=FALSE)
# training <- data_train[inTrain,]
# validation <- data_train[-inTrain,]


set.seed(1234)
ctrl <- trainControl( method = "cv", number = 5 )
xgbFit <- train( V15 ~ ., data = data_train, method = 'xgbTree', trControl = ctrl )


xgbFit

# Predicting response variable
p <- predict(xgbFit , data_test)
confusionMatrix(p, data_test$V15)

plot(xgbFit)

p <- predict(xgbFit ,data_test)
t <- factor(sub(" ", "", data_test$V15))
levels(data_test$V15)
confusionMatrix(p, t)




