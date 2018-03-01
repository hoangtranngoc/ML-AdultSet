# Library & prepare training, testing data
colClasses=c("integer", "factor", "integer", "factor", "integer", "factor", "factor", "factor", "factor", "factor",
             "integer", "integer", "integer", "factor", "factor")
colNames <- c("age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "outcome")

url_train <- "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
data_train <- read.table( file=url_train, header=FALSE, colClasses=colClasses, col.names = colNames, sep=",", strip.white=TRUE )

url_test <- "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
data_test <- read.table( file=url_test, skip=1, header=FALSE, colClasses=colClasses, col.names = colNames, sep=",", strip.white=TRUE)

# remove trailing dot
data_test[,15] <- factor(sub("\\.", "", data_test[,15]))

# Scale the column 'fnlwgt'.
#data_test$fnlwgt <- scale(data_test$fnlwgt)
#data_train$fnlwgt <- scale(data_train$fnlwgt)

library(caret)

# Normalize
train_normalizationlist <- preProcess(data_train, method=c("range"))
test_normalizationlist <- preProcess(data_test, method=c("range"))
data_train_normalized <- predict(train_normalizationlist, data_train)
data_test_normalized <- predict(test_normalizationlist, data_test)
rm(train_normalizationlist, test_normalizationlist)

# Apply centering and scaling
train_centeringandscalinglist <- preProcess(data_train_normalized, method=c("center", "scale"))
test_centeringandscalinglist <- preProcess(data_test_normalized, method=c("center", "scale"))
data_train_processed <- predict(train_centeringandscalinglist, data_train_normalized)
data_test_processed <- predict(test_centeringandscalinglist, data_test_normalized)
rm(train_centeringandscalinglist, test_centeringandscalinglist)

data_train <- data_train_processed 
data_test <- data_test_processed

# Debugging output, nice to have
dtrm <- mean(data_train$fnlwgt)
dtrs <- sd(data_train$fnlwgt)
dtem <- mean(data_test$fnlwgt)
dtes <- sd(data_test$fnlwgt)
message("Training: Mean of fnlwgt is ", dtrm)
message("Training: Standard deviation of fnlwgt is ", dtrs)
message("Testing: Mean of fnlwgt is ", dtem)
message("Testing: Standard deviation of fnlwgt is ", dtes)
rm(dtrm,dtrs,dtem,dter)

# Copy the data, and clean up
data_train <- data_train_processed
data_test <- data_test_processed
rm(data_train_processed, data_test_processed)

# Create the validation set from the training set
smp_size = floor(0.8*nrow(data_train))
set.seed(788)
train_ind <- sample(seq_len(nrow(data_train)), size=smp_size)
data_train_smaller <- data_train[train_ind,]
data_validation <- data_train[-train_ind,]

library(onehot)

# merge train, validation and test set before encoding
data <- rbind(data_train_smaller, data_validation, data_test)
encoder <- onehot(data_train_smaller, max_levels=45)

ohenc_data_train <- predict(encoder, data_train_smaller)
ohenc_data_validation <- predict(encoder, data_validation)
ohenc_data_test <- predict(encoder, data_test)

# write to files
write.table(ohenc_data_train, file= "ohenc_data.train", sep= " ", row.names=FALSE, col.names=FALSE)
write.table(ohenc_data_test, file= "ohenc_data.test", sep= " ", row.names=FALSE, col.names=FALSE)

write.table(ohenc_data_validation, file= "ohenc_data.validation", sep= " ", row.names=FALSE, col.names=FALSE)
write.table(ohenc_data_validation, file= "ohenc_data_colNames.validation", sep= " ", row.names=FALSE, col.names=TRUE)

write.table(ohenc_data_train, file= "ohenc_data_colNames.train", sep= " ", row.names=FALSE, col.names=TRUE)
write.table(ohenc_data_test, file= "ohenc_data_colNames.test", sep= " ", row.names=FALSE, col.names=TRUE)
