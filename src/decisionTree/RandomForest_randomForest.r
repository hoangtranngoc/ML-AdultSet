
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


common <- intersect(names(data_train), names(data_test)) 
for (p in common) { 
  if (class(data_train[[p]]) == "factor") { 
    levels(data_test[[p]]) <- levels(data_train[[p]]) 
  } 
}
head(data_train,5)
head(data_test,5)


# Establish a list of possible values for mtry, nodesize and sampsize
mtry <- seq(2, ncol(data_train) * 0.8, 2)
nodesize <- seq(1, 8, 2)
sampsize <- nrow(data_train) * c(0.7, 0.8, 1)

# Create a data frame containing all combinations 
hyper_grid <- expand.grid(mtry = mtry, nodesize = nodesize, sampsize = sampsize)

# Create an empty vector to store OOB error values
oob_err <- c()

# Write a loop over the rows of hyper_grid to train the grid of models
for (i in 1:nrow(hyper_grid)) {

    # Train a Random Forest model
    model <- randomForest(x=data_train[,1:14],
                          y=data_train[,15],
                          mtry = hyper_grid$mtry[i],
                          nodesize = hyper_grid$nodesize[i],
                          sampsize = hyper_grid$sampsize[i])
                          
    # Store OOB error for the model                      
    oob_err[i] <- model$err.rate[nrow(model$err.rate), "OOB"]
}

# Identify optimal set of hyperparmeters based on OOB error
opt_i <- which.min(oob_err)
print(hyper_grid[opt_i,])



set.seed(1234)
model <- randomForest(x=data_train[,1:14],
                      y=data_train[,15],
                      mtry = 2,
                      nodesize = 1,
                      sampsize = 32561)
model
# Predicting response variable
p <- predict(model ,data_test)
confusionMatrix(p, data_test$V15)



model <- randomForest(x=data_train[,1:14],
                      y=data_train[,15],
                      mtry = 9,
                      nodesize = 5)
model

# Predicting response variable
p <- predict(model ,data_test)
confusionMatrix(p, data_test$V15)

library(ggplot2)
bestmtry <- tuneRF(data_train[-15],data_train$V15, ntreeTry=500, 
     stepFactor=1.5,improve=0.01, trace=TRUE, dobest=FALSE)


bestmtry
adult.rf <-randomForest(V15~.,data=data_train, mtry=2, ntree=1000, 
     keep.forest=TRUE, importance=TRUE,test=data_test)

adult.rf


