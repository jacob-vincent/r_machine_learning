library(data.table)
library(mlr)
library(xgboost)

#set variable names
setcol <- c("age", "workclass", "fnlwgt", "education", "education-num",
            "marital-status", "occupation", "relationship", "race", "sex",
            "capital-gain", "capital-loss", "hours-per-week", "native-country", "target")

#load data
train <- read.table("adultdata.txt", header = F, sep = ",", 
                      col.names = setcol, na.strings = c(" ?"), stringsAsFactors = F)
test <- read.table("adulttest.txt",header = F,sep = ",",
                     col.names = setcol,skip = 1, na.strings = c(" ?"),stringsAsFactors = F)

#convert data frame to data table
setDT(train) 
setDT(test)

#check missing values 
table(is.na(train))
sapply(train, function(x) sum(is.na(x))/length(x))*100

table(is.na(test))
sapply(test, function(x) sum(is.na(x))/length(x))*100

#quick data cleaning
#remove extra character from target variable
library(stringr)
test [,target := substr(target,start = 1,stop = nchar(target)-1)]

#remove leading whitespaces
char_col <- colnames(train)[ sapply (test,is.character)]
for(i in char_col) set(train,j=i,value = str_trim(train[[i]],side = "left"))

for(i in char_col) set(test,j=i,value = str_trim(test[[i]],side = "left"))

#set all missing value as "Missing" 
train[is.na(train)] <- "Missing" 
test[is.na(test)] <- "Missing"


#using one hot encoding 
labels <- train$target 
ts_label <- test$target
new_tr <- model.matrix(~.+0,data = train[,-c("target"),with=F]) 
new_ts <- model.matrix(~.+0,data = test[,-c("target"),with=F])

#convert factor to numeric 
labels <- as.numeric(as.factor(labels))-1
ts_label <- as.numeric(as.factor(ts_label))-1

#preparing matrix 
dtrain <- xgb.DMatrix(data = new_tr,label = labels) 
dtest <- xgb.DMatrix(data = new_ts,label=ts_label)

#default parameters
params <- list(booster = "gbtree", objective = "binary:logistic", 
               eta=0.3, gamma=0, max_depth=6, min_child_weight=1, 
               subsample=1, colsample_bytree=1)

xgbcv <- xgb.cv( params = params, data = dtrain, 
                 nrounds = 100, nfold = 5, showsd = T,
                 stratified = T, print_every_n = 10, early_stopping_rounds = 20, maximize = F)
##best iteration = 50

min(xgbcv$evaluation_log$test_error_mean)

#first default - model training
xgb1 <- xgb.train(params = params, data = dtrain, 
                  nrounds = 50, watchlist = list(val=dtest,train=dtrain), 
                  print_every_n = 10, early_stopping_rounds = 10, maximize = F , 
                  eval_metric = "error")
#model prediction
xgbpred <- predict(xgb1,dtest)
xgbpred <- ifelse(xgbpred > 0.5,1,0)


