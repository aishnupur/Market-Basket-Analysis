#install.packages('arules')
library(arules)
library(data.table)
library(dplyr)
library(ggplot2)
library(knitr)
library(stringr)
library(DT)
library(plotly)
library(arulesViz)
library(visNetwork)
library(igraph)
library(kableExtra)
library(RColorBrewer)

train_data <- read.csv('train.csv')
test_data <- read.csv('test.csv')

upc_dept <- data.frame(train_data$Upc, train_data$DepartmentDescription)
dataset2 <- data.frame(train_data$VisitNumber, train_data$Upc)
names(dataset2) <- c("transaction", "product")

library(plyr)
ordered_data <- ddply(dataset2,c("transaction"),
                      function(df1)paste(df1$product,collapse = ","))
write.csv(ordered_data,"ordered_data.csv",quote = FALSE,row.names = TRUE)

txn = read.transactions(file = "ordered_data.csv",rm.duplicates = TRUE,format = "basket",sep = ",",cols=1)
basket_rules <- apriori(txn,parameter = list(sup = 0.0006,conf = 0.5,target = "rules"))
inspect(basket_rules)
#inspect(sort(basket_rules,by='lift')[1:100])
summary(basket_rules)


arules::itemFrequencyPlot(txn,
                          topN=20,
                          col=brewer.pal(8,'Pastel2'),
                          main='Relative Item Frequency Plot',
                          type="relative",
                          ylab="Item Frequency (Relative)")

basket_rules <- basket_rules[!is.redundant(basket_rules)]
rules_dt <- data.table( lhs = labels( lhs(basket_rules) ), 
                        rhs = labels( rhs(basket_rules) ), 
                        quality(basket_rules) )[ order(-lift), ]

plotly_arules(basket_rules)

sel <- plot(basket_rules, measure=c("support", "lift"), 
            shading = "confidence",
            interactive = TRUE)

subrules2 <- head(sort(basket_rules, by="lift"))
ig <- plot( subrules2, method="graph", control=list(type="items") )

######################################################################################

train_data$TripType <- as.factor(train_data$TripType)

half_train <- train_data[1:5000,]
half_train2 <- train_data[5001:8000,]

half_test <- train_data[5001:8000,2:7]

table(half_train2$TripType)

classificator_strong <- CBA(
  TripType ~ ., data = half_train, supp = 0.0000006, conf=0.2, verbose = FALSE
)

predicted_strong <- predict(classificator_strong, half_test)
cross_tab_strong<- table(predicted = predicted_strong, true = train_data[5001:8000,]$TripType)
accuracy_strong <- (cross_tab_strong[1,1]+cross_tab_strong[2,2])/sum(cross_tab_strong)

# dataset3 <- aggregate(dataset2$product ~ dataset2$transaction, dataset2, c)
# names(dataset3) <- c("transaction", "products")
# 
# 
# rules <- apriori(dataset3, parameter = list(supp = 0.2, conf = 0.2)) # Min Support as 0.001, confidence as 0.8.
# rules_conf <- sort (rules, by="confidence", decreasing=TRUE)
# 
# inspect(head(rules_conf))