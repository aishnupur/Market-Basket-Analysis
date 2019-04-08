##This is the project file for Data mining
##Authors: Shrey, Tapan, Aishwarya
train_data <- read.csv('train.csv')
test_data <- read.csv('test.csv')

train_data.factor <- factor(train_data$TripType)
train_data$TripType <- as.numeric(train_data.factor)
train_data$TripType <- train_data$TripType - 1
train_data$TripType <- factor(train_data$TripType)

#############################DATA PREPROCESSING#########################################

sum(is.na(train_data$FinelineNumber))
row.has.na <- apply(train_data, 1, function(x){any(is.na(x))})
sum(row.has.na)
train_data <- train_data[!row.has.na,]

rownames(train_data) <- seq(length=nrow(train_data))

summary(train_data)

#############################DATA SPLITTING#############################################

library(caTools)
set.seed(123)
split = sample.split(train_data$TripType, SplitRatio = 0.80)
training_set2 = subset(train_data, split == TRUE)
test_set1 = subset(train_data, split == FALSE)
test_set2 <- test_set1[,2:7]

rownames(training_set2) <- seq(length=nrow(training_set2))
rownames(test_set1) <- seq(length=nrow(test_set1))
rownames(test_set2) <- seq(length=nrow(test_set2))


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


##################################LOGISTIC REGRESSION#####################################

##Tapan

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
                        eta = 1, nthread = 4, nrounds = 4,verbose = 1,
                        objective = "multi:softprob",
                        eval_metric = "mlogloss",
                        num_class = 38)
pred_xg = predict(classifier_xg,data.matrix(test_set1[,2:7]))

xgbModel <- xgbnoost(data = data.matrix(training_set2[,2:7]), label = data.matrix(training_set2[,1]),
                    max_depth = 80,
                    eta = 1, nthread = 8, nrounds = 1,
                    objective = "multi:softmax",
                    eval_metric = "mlogloss",
                    num_class = 38)

predicted.labels_XGB <- predict(xgbModel, data.matrix(test_set2))

cm_xg = table(test_set1[,1], predicted.labels_XGB)

library(caret)
confusionMatrix(cm_xg)



###############################################################################
#install.packages('R6')
library(R6)
xgb_trainer <- XGBTrainer$new()
gst <-GridSearchCV$new(trainer = xgb_trainer,
                       parameters = list(n_estimators = c(100),
                                         max_depth = c(5,2,10,50,60,80,100),
                                         eta = 1, nthread = 8, nrounds = c(1,2,3)),
                       n_folds = 3,
                       scoring = c('accuracy','rmse'))
data("iris")
gst$fit(iris, "Species")
gst$best_iteration()

xgb.grid <- expand.grid(max_depth = c(5,2,10,50,60,80,100),
                        eta = 1, nthread = 8, nrounds = c(1,2,3),
                        scoring = c('accuracy','rmse')
)