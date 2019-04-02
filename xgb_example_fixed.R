library(tidyverse)
library(mlr)
library(sjmisc)
library(xgboost)
library(data.table)
library(caret)

#set variable names
setcol <- c("age", "workclass", "fnlwgt", "education", "education-num",
            "marital-status", "occupation", "relationship", "race", "sex",
            "capital-gain", "capital-loss", "hours-per-week", "native-country", "target")

train_data <- read_csv('adultdata.txt', col_names=setcol, na=c("?"," ","","NA"))
test_data <- read_csv('adulttest.txt', col_names = setcol, skip=1, na=c("?"))

train_data %>% 
  select_if(function(x) any(is.na(x))) %>%
  summarise_all(funs(sum(is.na(.))))

train_data %>%
  summarise(missing_workclass=(n()-sum(!is.na(workclass)))/n()*100,
            missing_occupation=(n()-sum(!is.na(occupation)))/n()*100,
            missing_nativeCountry=(n()-sum(!is.na(`native-country`)))/n()*100)

test_data %>% 
  select_if(function(x) any(is.na(x))) %>%
  summarise_all(funs(sum(is.na(.))))

test_data %>%
  summarise(missing_workclass=(n()-sum(!is.na(workclass)))/n()*100,
            missing_occupation=(n()-sum(!is.na(occupation)))/n()*100,
            missing_nativeCountry=(n()-sum(!is.na(`native-country`)))/n()*100)

#One-hot encoding of train and test data
labels <- train_data %>% to_dummy(target, suffix = 'label') %>% select(last_col())
test_labels <- test_data %>% to_dummy(target, suffix = 'label') %>% select(last_col()) %>% rename("target_>50K"="target_>50K.")

cat_vars <- train_data %>% select(-c('target')) %>% select_if(function(x) is.character(x)) %>% names()
new_cols <- train_data %>% select(-c('target')) %>% to_dummy(cat_vars, suffix='label')
df <- train_data %>% select_if(function(x) !is.character(x))
train_df <- bind_cols(df,new_cols)

cat_vars <- test_data %>% select(-c('target')) %>% select_if(function(x) is.character(x)) %>% names()
new_cols <- test_data %>% select(-c('target')) %>% to_dummy(cat_vars, suffix='label')
df <- test_data %>% select_if(function(x) !is.character(x))
test_df <- bind_cols(df,new_cols)

#Need to add "native-country_Holand-Netherlands" column to test data
test_df$`native-country_Holand-Netherlands` <- 0
test_df <- test_df %>% select(names(train_df))

dtrain <- xgb.DMatrix(data = as.matrix(train_df),label = as.matrix(labels))
dtest <- xgb.DMatrix(data = as.matrix(test_df), label = as.matrix(test_labels))

#default parameters
params <- list(booster = "gbtree", objective = "binary:logistic", 
               eta=0.3, gamma=0, max_depth=6, min_child_weight=1, 
               subsample=1, colsample_bytree=1)

xgbcv <- xgb.cv( params = params, data = dtrain, 
                 nrounds = 100, nfold = 5, showsd = T,
                 stratified = T, print_every_n = 10, early_stopping_rounds = 20, maximize = F)
##best iteration = 58

min(xgbcv$evaluation_log$test_error_mean)

#first default - model training
xgb1 <- xgb.train(params = params, data = dtrain, 
                  nrounds = 58, watchlist = list(val=dtest,train=dtrain), 
                  print_every_n = 10, early_stopping_rounds = 10, maximize = F, 
                  eval_metric = "error")

#model prediction
xgbpred <- predict(xgb1,dtest)
xgbpred <- ifelse(xgbpred > 0.5,1,0)


#confusion matrix

confusionMatrix(as.factor(xgbpred), as.factor(test_labels$`target_>50K`))
#Accuracy - 86.54%` 

#view variable importance plot
mat <- xgb.importance (feature_names = names(train_df),model = xgb1)
xgb.plot.importance (importance_matrix = mat[1:20]) 
