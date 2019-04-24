train_data <- read.csv('train.csv')
test_data <- read.csv('test.csv')

train_data$TripType <- factor(train_data$TripType)
train_data$TripType <- as.numeric(train_data$TripType)

# train_data$TripType <- train_data$TripType - 1
# train_data$TripType <- factor(train_data$TripType)

sum(is.na(train_data$FinelineNumber))
row.has.na <- apply(train_data, 1, function(x){any(is.na(x))})
sum(row.has.na)
train_data <- train_data[!row.has.na,]

rownames(train_data) <- seq(length=nrow(train_data))

summary(train_data)
table(train_data$TripType)
freq <- table(train_data$TripType)
freq[4]

row <- train_data[2,]
print(row$TripType)

for(i in 1:nrow(train_data)) {
  row <- train_data[i,]
  if(row$TripType==2 | row$TripType==3 | row$TripType==4)
  {
    train_data[i,1] = 1
  }
  else if(row$TripType==8 | row$TripType==9 | row$TripType==11 | row$TripType==12)
  {
    train_data[i,1] = 7
  }
  else if(row$TripType==13 | row$TripType==14 | row$TripType==15 | row$TripType==16 | row$TripType==19 | row$TripType==20)
  {
    train_data[i,1] = 10
  }
  else if(row$TripType==21 | row$TripType==22 | row$TripType==24 )
  {
    train_data[i,1] = 17
  }
  else if(row$TripType==23 | row$TripType==27 | row$TripType==34 )
  {
    train_data[i,1] = 28
  }
  else if(row$TripType==26)
  {
    train_data[i,1] = 25
  }
  else if(row$TripType==36)
  {
    train_data[i,1] = 38
  }
}
train_data$TripType <- factor(train_data$TripType)
train_data$TripType <- as.numeric(train_data$TripType)
table(train_data$TripType)
