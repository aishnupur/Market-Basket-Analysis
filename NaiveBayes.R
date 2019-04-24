train_data <- read.csv('train.csv')
test_data <- read.csv('test.csv')
sum(is.na(train_data$FinelineNumber))
row.has.na <- apply(train_data, 1, function(x){any(is.na(x))})
sum(row.has.na)
train_data <- train_data[!row.has.na,]

rownames(train_data) <- seq(length=nrow(train_data))
table(train_data$TripType)

for(i in 1:nrow(train_data)) {
  row <- train_data[i,]
  if(row$TripType==4 | row$TripType==5 | row$TripType==6)
  {
    train_data[i,1] = 3
  }
  else if(row$TripType==12 | row$TripType==14 | row$TripType==18 | row$TripType==19)
  {
    train_data[i,1] = 9
  }
  else if(row$TripType==20 | row$TripType==21 | row$TripType==22 | row$TripType==23 | row$TripType==26 | row$TripType==27)
  {
    train_data[i,1] = 15
  }
  else if(row$TripType==28 | row$TripType==29 | row$TripType==31 )
  {
    train_data[i,1] = 24
  }
  else if(row$TripType==30 | row$TripType==34 | row$TripType==41 )
  {
    train_data[i,1] = 35
  }
  else if(row$TripType==33)
  {
    train_data[i,1] = 32
  }
  else if(row$TripType==43)
  {
    train_data[i,1] = 999
  }
}

table(train_data$TripType)

train_data$TripType <- factor(train_data$TripType)


test_data <- test_data[,-c(4)]

variance <- sapply(train_data, var)
variance
train_data <- train_data[,-c(5)]

library(caTools)
set.seed(123)
split = sample.split(train_data$TripType, SplitRatio = 0.80)
training_set2 = subset(train_data, split == TRUE)
test_set1 = subset(train_data, split == FALSE)
test_set2 <- test_set1[,2:6]

rownames(training_set2) <- seq(length=nrow(training_set2))
rownames(test_set1) <- seq(length=nrow(test_set1))
rownames(test_set2) <- seq(length=nrow(test_set2))

library(e1071)
classifier_nb = naiveBayes(TripType~.,
                           data = training_set2)
pred_nb = predict(classifier_nb,test_set2)

cm_nb = table(test_set1[,1], pred_nb)
cm_nb

#install.packages('caret')
library(caret)
confusionMatrix(cm_nb)

RMSE <- mean((as.numeric(test_set1[,1])-as.numeric(pred_nb))^2)
log(RMSE)