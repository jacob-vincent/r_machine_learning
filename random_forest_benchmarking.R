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

for(i in names(train_df)){
  train_df[[i]] <- as.integer(train_df[[i]])
  test_df[[i]] <- as.integer(test_df[[i]])
}


#Define randomForest::randomForest model
system.time({adult_random_forest <- randomForest::randomForest(x=train_df,
                                                  y=as.factor(labels), 
                                                  proximity=TRUE, 
                                                  importance=TRUE)})
# user       system     elapsed 
# 2944.410   1818.999   5706.952

#Get predictions from randomForest::randomForest model
arf_preds <- predict(adult_random_forest, test_df)

#Evaluate performance of randomForest::randomForest model
confusionMatrix(as.factor(arf_preds),as.factor(test_labels))

#                Reference
#   Prediction   0      1
#        0      11716  1466
#        1      719    2380

#Accuracy : 0.8658
#Sensitivity : 0.9422         
#Specificity : 0.6188         
#Pos Pred Value : 0.8888         
#Neg Pred Value : 0.7680         
#Prevalence : 0.7638 

### ranger (Fast RandomForest Implementation) ###
ranger_train_df <- as.data.frame(train_df %>% mutate('target_label'= labels))
ranger_train_df$target_label <- as.factor(ranger_train_df$target_label)
system.time({
  ranger_rf <- ranger::ranger(dependent.variable.name = 'target_label', data=ranger_train_df)
  })
# user      system   elapsed 
# 129.279   0.661    36.568

# Predict with ranger_rf
ranger_preds <- predict(ranger_rf, test_df)

# Evaluate ranger_preds
confusionMatrix(as.factor(ranger_preds$predictions), as.factor(test_labels))

#               Reference
# Prediction     0     1
#          0  11704   1461
#          1    731   2385

# Accuracy : 0.8654
# Sensitivity : 0.9412        
# Specificity : 0.6201        
# Pos Pred Value : 0.8890        
# Neg Pred Value : 0.7654        
# Prevalence : 0.7638

### h2o RandomForest Implementation ###
library(h2o)
h2o.init(nthreads = -1)

train_hx <- as.data.frame(train_df %>% mutate('target_label'= labels))
train_hx$target_label <- as.factor(train_hx$target_label)
train_df.hex <- as.h2o(train_hx, destination_frame = 'train_df.hex')

test_hx <- as.data.frame(test_df %>% mutate('target_label' = test_labels))
test_hx$target_label <- as.factor(test_hx$target_label)
test_df.hex <- as.h2o(test_hx, destination_frame = 'test_df.hex')

system.time({
  h2o_rf <- h2o.randomForest(y='target_label', training_frame = train_df.hex, ntrees = 500)
  })
# user    system    elapsed 
# 0.781    0.172     64.493 

# Get h2o_rf predictions
h2o_preds1 <- h2o.predict(h2o_rf, test_df.hex)$p1>=0.5

# Evaluate h2o_preds
confusionMatrix(as.factor(as.vector(h2o_preds1)), as.factor(test_labels))

#                 Reference
# Prediction       0     1
#          0     11737  1537
#          1       698  2309

# Accuracy : 0.8627
# Sensitivity : 0.9439         
# Specificity : 0.6004         
# Pos Pred Value : 0.8842         
# Neg Pred Value : 0.7679         
# Prevalence : 0.7638 

h2o.shutdown(prompt = FALSE)


