train_data <- read.csv('train.csv')
test_data <- read.csv('test.csv')

train_data.factor <- factor(train_data$TripType)
train_data$TripType <- as.numeric(train_data.factor)



library(caTools)
set.seed(123)
split = sample.split(train_data$TripType, SplitRatio = 0.80)
training_set2 = subset(train_data, split == TRUE)
test_set1 = subset(train_data, split == FALSE)
test_set2 <- test_set1[,2:7]

####################################NAIVE BAYES#########################################
library(e1071)
classifier_nb = naiveBayes(TripType~.,
                           data = training_set2)
pred_nb = predict(classifier_nb,test_set2)

cm_nb = table(test_set1[,1], pred_nb)
cm_nb

#install.packages('caret')
library(caret)
confusionMatrix(cm_nb)

####################################DECISION TREE#########################################
library(rpart)
classifier_dt = rpart(formula = TripType~.,
                   data = training_set2)
pred_dt = predict(classifier_dt,test_set2)
cm_dt = table(test_set1[,1], pred_dt)
cm_dt

#install.packages('caret')
library(caret)
confusionMatrix(cm_dt)

####################################XGBOOST#########################################
#install.packages("xgboost")
library(xgboost)

training_set2[] <- lapply(training_set2, function(x) {
  if(is.factor(x)) as.numeric(as.character(x)) else x
})

dtrain <- xgb.DMatrix(data = as.matrix(training_set2), label = training_set2$TripType)

classifier_xg = xgboost(data = dtrain,
                        max.depth = 10, 
                        eta = 1, nthread = 4, nrounds = 4,verbose = 1)
pred_xg = predict(classifier_xg,data.matrix(test_set1[,2:7]))

xgbModel <- xgboost(data = data.matrix(training_set2[,2:7]), label = data.matrix(training_set2[,1]), max_depth = 10, eta = 1, nthread = 4, nrounds = 4,
                    objective = "multi:softprob",
                    eval_metric = "mlogloss",
                    num_class = 38)
#max_depth = 2, eta = 1, verbose = 2

predicted.labels_XGB <- predict(xgbModel, data.matrix(test_set2))
cm_xg = table(test_set1[,1], predicted.labels_XGB)

library(caret)
confusionMatrix(cm_xg)