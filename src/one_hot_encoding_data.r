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


library(onehot)

# merge train and test set before encoding
data <- rbind(data_train, data_test)
encoder <- onehot(data_train, max_levels=45)

ohenc_data_train <- predict(encoder, data_train)
ohenc_data_test <- predict(encoder, data_test)

# write to files
write.table(ohenc_data_train, file= "ohenc_data.train", sep= " ", row.names=FALSE, col.names=FALSE)
write.table(ohenc_data_test, file= "ohenc_data.test", sep= " ", row.names=FALSE, col.names=FALSE)

write.table(ohenc_data_train, file= "ohenc_data_colNames.train", sep= " ", row.names=FALSE, col.names=TRUE)
write.table(ohenc_data_test, file= "ohenc_data_colNames.test", sep= " ", row.names=FALSE, col.names=TRUE)