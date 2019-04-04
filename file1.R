#install.packages('arules')
library(arules)

dataset1 <- read.csv('train.csv')
dataset2 <- data.frame(dataset1$VisitNumber, dataset1$Upc)
names(dataset2) <- c("transaction", "product")

library(plyr)
ordered_data <- ddply(dataset2,c("transaction"),
                      function(df1)paste(df1$product,collapse = ","))
write.csv(ordered_data,"ordered_data.csv",quote = FALSE,row.names = TRUE)

txn = read.transactions(file = "ordered_data.csv",rm.duplicates = TRUE,format = "basket",sep = ",",cols=1)
basket_rules <- apriori(txn,parameter = list(sup = 0.005,conf = 0.5,target = "rules"))
inspect(basket_rules)
inspect(sort(basket_rules,by='lift')[1:100])
summary(basket_rules)

dataset3 <- aggregate(dataset2$product ~ dataset2$transaction, dataset2, c)
names(dataset3) <- c("transaction", "products")


rules <- apriori(dataset3, parameter = list(supp = 0.2, conf = 0.2)) # Min Support as 0.001, confidence as 0.8.
rules_conf <- sort (rules, by="confidence", decreasing=TRUE)

inspect(head(rules_conf))