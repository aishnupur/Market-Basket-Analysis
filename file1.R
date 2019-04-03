#install.packages('arules')
library(arules)

dataset1 <- read.csv('train.csv')
dataset2 <- data.frame(dataset1$VisitNumber, dataset1$Upc)
names(dataset2) <- c("transaction", "product")


dataset3 <- aggregate(dataset2$product ~ dataset2$transaction, dataset2, c)
names(dataset3) <- c("transaction", "products")

#library(data.table)
#dataset3 <- as.data.table(dataset2)[,list(list(dataset2$product)), by = dataset2$transaction]

data("Groceries")
head(dataset3, 2)

rules <- apriori(dataset3, parameter = list(supp = 0.2, conf = 0.2)) # Min Support as 0.001, confidence as 0.8.
rules_conf <- sort (rules, by="confidence", decreasing=TRUE)

inspect(head(rules_conf))