# Remember to save the resulting files. :)

smp_size = floor(0.7*nrow(train))
set.seed(788)
train_ind <- sample(seq_len(nrow(train)), size=smp_size)
trainNew <- train[train_ind,]
validation <- train[-train_ind,]

