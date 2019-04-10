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

train_data <- train_data[,-c(5,7)]
test_data <- test_data[,-c(4,6)]

###########################ONE HOT ENCODING###############################################

library(mltools)
library(data.table)

trip_type_1h <- one_hot(as.data.table(train_data$TripType))
as.data.frame(trip_type_1h)

train_data_1h <- data.frame(train_data[,2:5], trip_type_1h[,1:38])

###########################DATA ANALYSIS###############################################

library(ggplot2)

###BAR PLOTS#####

##Visit Number
ggplot(train_data, aes(VisitNumber)) +
  geom_bar(fill = "#0073C2FF")

##Trip Type
barplot(table(train_data$TripType), las=2)

##Department Description
barplot(table(train_data$DepartmentDescription), las=2)

#Scan count
barplot(table(train_data$ScanCount), las=2)

###PIE CHART#####
library(plotrix)
table_weekday <- table(train_data$Weekday)
labels_weekday <- paste(names(table_weekday), "\n", table_weekday, sep="")
pie3D(table_weekday, labels = labels_weekday, main="Pie Chart of Departments")

neg_rows <- subset(train_data, ScanCount  < 0)
barplot(table(neg_rows$DepartmentDescription), las=2)

#############################DATA SPLITTING#############################################

library(caTools)
set.seed(123)
split = sample.split(train_data$TripType, SplitRatio = 0.80)
training_set2 = subset(train_data, split == TRUE)
test_set1 = subset(train_data, split == FALSE)
test_set2 <- test_set1[,2:5]

rownames(training_set2) <- seq(length=nrow(training_set2))
rownames(test_set1) <- seq(length=nrow(test_set1))
rownames(test_set2) <- seq(length=nrow(test_set2))

#############################DATA SPLITTING WITH 1 H#############################################

library(caTools)
set.seed(123)
split_1h = sample.split(train_data_1h[5:42], SplitRatio = 0.80)
dplyr::filter(train_data_1h)
training_set2_1h = subset(train_data_1h, split == TRUE)
test_set1_1h = subset(train_data_1h, split == FALSE)

rownames(training_set2) <- seq(length=nrow(training_set2))
rownames(test_set1) <- seq(length=nrow(test_set1))
rownames(test_set2) <- seq(length=nrow(test_set2))

#############################XGBOOST WITH 1 H#############################################

xgbModel_1h <- xgboost(data = data.matrix(training_set2[,2:5]), 
                       label = data.matrix(training_set2[,1]),
                       max_depth = 80,
                       eta = 1, nthread = 8, nrounds = 1,
                       objective = "multi:softmax",
                       eval_metric = "mlogloss",
                       num_class = 38)

predicted.labels_XGB <- predict(xgbModel, data.matrix(test_set2))

cm_xg = table(test_set1[,1], predicted.labels_XGB)

library(caret)
confusionMatrix(cm_xg)


####################################NAIVE BAYES#########################################

training_set2 <- training_set2[,-c(5,7)]
 

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

# training_set2[] <- lapply(training_set2, function(x) {
#   if(is.factor(x)) as.numeric(as.character(x)) else x
# })
# 
# dtrain <- xgb.DMatrix(data = as.matrix(training_set2), label = training_set2$TripType)
# 
# classifier_xg = xgboost(data = dtrain,
#                         max.depth = 10, 
#                         eta = 1, nthread = 4, nrounds = 4,verbose = 1,
#                         objective = "multi:softprob",
#                         eval_metric = "mlogloss",
#                         num_class = 38)
# pred_xg = predict(classifier_xg,data.matrix(test_set1[,2:6]))

xgbModel <- xgboost(data = data.matrix(training_set2[,2:5]), 
                    label = data.matrix(training_set2[,1]),
                    max_depth = 12,
                    eta = 1, nthread = 8, nrounds = 300,
                    objective = "multi:softmax",
                    eval_metric = "mlogloss",
                    num_class = 38,
                    eta = 0.1,
                    early_stopping_rounds = 10,
                    min_child_weight = 3)

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