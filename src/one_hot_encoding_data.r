
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

library(caret)

# Create the validation set from the training set
smp_size = floor(0.8*nrow(data_train))
set.seed(788)
train_ind <- sample(seq_len(nrow(data_train)), size=smp_size)
data_train_smaller <- data_train[train_ind,]
data_val <- data_train[-train_ind,]

# Apply centering and scaling
preProcessModel <- preProcess(data_train_smaller, method=c("center", "scale"))
data_train_processed <- predict(preProcessModel, data_train_smaller)
data_val_processed <- predict(preProcessModel, data_val)
data_test_processed <- predict(preProcessModel, data_test)

data_train <- data_train_processed 
data_val <- data_val_processed 
data_test <- data_test_processed
# Copy the data, and clean up
rm(data_train_processed, data_val_processed, data_test_processed)


# Debugging output, nice to have
dtrm <- mean(data_train$fnlwgt)
dtrs <- sd(data_train$fnlwgt)
dvam <- mean(data_val$fnlwgt)
dvas <- sd(data_val$fnlwgt)
dtem <- mean(data_test$fnlwgt)
dtes <- sd(data_test$fnlwgt)
cat("\nTraining: Mean of fnlwgt is ", dtrm)
cat("\nTraining: Standard deviation of fnlwgt is ", dtrs)
cat("\nValidation: Mean of fnlwgt is ", dvam)
cat("\nValidation: Standard deviation of fnlwgt is ", dvas)
cat("\nTesting: Mean of fnlwgt is ", dtem)
cat("\nTesting: Standard deviation of fnlwgt is ", dtes)

library(onehot)

# merge train, validation and test set before encoding
data <- rbind(data_train, data_val, data_test)
encoder <- onehot(data, max_levels=45)

ohenc_data_train <- predict(encoder, data_train)
ohenc_data_val <- predict(encoder, data_val)
ohenc_data_test <- predict(encoder, data_test)

# this is for Tensorflow Framework
# remove special characters in column names 
colnames(ohenc_data_train) <- gsub(x = colnames(ohenc_data_train), pattern = "\\?", replacement = "Missing") 
colnames(ohenc_data_train) <- gsub(x = colnames(ohenc_data_train), pattern = "\\=", replacement = "") 
colnames(ohenc_data_train) <- gsub(x = colnames(ohenc_data_train), pattern = "\\&", replacement = "-")
colnames(ohenc_data_train) <- gsub(x = colnames(ohenc_data_train), pattern = "\\.", replacement = "-")
colnames(ohenc_data_train) <- gsub(x = colnames(ohenc_data_train), pattern = "\\(", replacement = "-")
colnames(ohenc_data_train) <- gsub(x = colnames(ohenc_data_train), pattern = "\\)", replacement = "-")

colnames(ohenc_data_val) <- colnames(ohenc_data_train)
colnames(ohenc_data_test) <- colnames(ohenc_data_train)

# write to files
write.table(ohenc_data_train, file= "ohenc_data.train", sep= " ", row.names=FALSE, col.names=FALSE)
write.table(ohenc_data_val, file= "ohenc_data.val", sep= " ", row.names=FALSE, col.names=FALSE)
write.table(ohenc_data_test, file= "ohenc_data.test", sep= " ", row.names=FALSE, col.names=FALSE)

write.table(ohenc_data_train, file= "ohenc_data_colNames.train", sep= " ", row.names=FALSE, col.names=TRUE)
write.table(ohenc_data_val, file= "ohenc_data_colNames.val", sep= " ", row.names=FALSE, col.names=TRUE)
write.table(ohenc_data_test, file= "ohenc_data_colNames.test", sep= " ", row.names=FALSE, col.names=TRUE)


head(ohenc_data_train)
head(ohenc_data_val)
head(ohenc_data_test)


