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

cat("N/A values in TripType column",sum(is.na(train_data$TripType)))
cat("N/A values in VisitNumber column",sum(is.na(train_data$VisitNumber)))
cat("N/A values in Weekday column",sum(is.na(train_data$Weekday)))
cat("N/A values in Upc column",sum(is.na(train_data$Upc)))
cat("N/A values in ScanCount column",sum(is.na(train_data$ScanCount)))
cat("N/A values in DepartmentDescription column",sum(is.na(train_data$DepartmentDescription)))
cat("N/A values in FineLineNumber column",sum(is.na(train_data$FinelineNumber)))

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

subrules2 <- head(sort(basket_rules, by="support"))
ig <- plot( subrules2, method="graph", control=list(type="items") )

######################################################################################