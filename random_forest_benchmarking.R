###Benchmarking RandomForest Algorithms
library(tidyverse)
library(mlr)
library(sjmisc)
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
labels <- labels$`target_>50K`
test_labels <- test_data %>% to_dummy(target, suffix = 'label') %>% select(last_col()) %>% rename("target_>50K"="target_>50K.")
test_labels <- test_labels$`target_>50K`

cat_vars <- train_data %>% select(-c('target')) %>% select_if(function(x) is.character(x)) %>% names()
new_cols <- train_data %>% select(-c('target')) %>% to_dummy(cat_vars, suffix='label')
df <- train_data %>% select_if(function(x) !is.character(x))
train_df <- bind_cols(df,new_cols)
remove(new_cols)
remove(df)
remove(train_data)

cat_vars <- test_data %>% select(-c('target')) %>% select_if(function(x) is.character(x)) %>% names()
new_cols <- test_data %>% select(-c('target')) %>% to_dummy(cat_vars, suffix='label')
df <- test_data %>% select_if(function(x) !is.character(x))
test_df <- bind_cols(df,new_cols)
remove(new_cols)
remove(df)
remove(test_data)

#Need to add "native-country_Holand-Netherlands" column to test data
test_df$`native-country_Holand-Netherlands` <- 0
test_df <- test_df %>% select(names(train_df))

#fill na with 0
for(i in names(train_df)){
  train_df[[i]] <- train_df[[i]] %>% tidyr::replace_na(list(i=0))
  test_df[[i]] <- test_df[[i]] %>% tidyr::replace_na(list(i=0))
}



#Define randomForest::randomForest model
adult_random_forest <- randomForest::randomForest(x=train_df,
                                                  y=as.factor(labels), 
                                                  proximity=TRUE, 
                                                  importance=TRUE)
#Get predictions from randomForest::randomForest model

#Evaluate performance of randomForest::randomForest model




