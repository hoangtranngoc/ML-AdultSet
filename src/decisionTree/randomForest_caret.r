
# Library & prepare training, testing data
library(caret)
library(randomForest)
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

# Map levels of training set to test set
common <- intersect(names(data_train), names(data_test)) 
for (p in common) { 
  if (class(data_train[[p]]) == "factor") { 
    levels(data_test[[p]]) <- levels(data_train[[p]]) 
  } 
}

head(data_train,5)
head(data_test,5)



# Train model
library(parallel)
library(doParallel)
set.seed(1234)

print(detectCores())
cluster <- makeCluster(detectCores()-1) 
registerDoParallel(cluster)

control <- trainControl(method="cv", number=5, allowParallel = TRUE)
system.time(modFit <- train(V15 ~ ., data=data_train, method="rf", trControl=control))
stopCluster(cluster)
modFit


p <- predict(modFit ,data_test)
confusionMatrix(p, data_test$V15)

modFit$finalModel


