# Theerarun Tubnonghee
# Jonathan Ko
# CS699
# Logistic Regression

# -------------------------------------------------------------------------


data <- read.csv("/Users/jonathanko/Documents/School/CS699/Project/project_dataset.csv", 
                 header = TRUE)
head(data)
dim(data)

# Load necessary libraries
library(caret) # for nearZeroVar
library(car) # for vif
library(rsample)
#install.packages("pRoc")
library(pROC)
#install.packages("ROSE")
library(ROSE)
#install.packages("RWeka")
#library("RWeka")
#install.packages("rJava")
#library(rJava)
# Identify near-zero variance predictors
nzv <- nearZeroVar(data, saveMetrics = TRUE,freqCut = 95/5)
-nzv$nzv

# Print out the predictors with near-zero variance
print(nzv[nzv$nzv,])

# Remove near-zero variance predictors from the data
data_clean <- data[, !(nzv$nzv | nzv$zeroVar)]
dim(data_clean)
head(data_clean)

# Check for multicollinearity using VIF
# We may need to fit a model first, depending on what you are trying to predict. Assuming a linear model:
model <- glm(o_bullied ~ ., data = data_clean)

# Calculate VIF
vif_values <- vif(model)

# Print out the VIF values
print(vif_values)

# It's often useful to consider a threshold for VIF, such as 5 or 10, to determine which variables might be causing multicollinearity.
high_vif <- vif_values[vif_values > 5] # or you can use 10 as your threshold
print(high_vif)

names(high_vif)
names(data_clean)
sum(!(names(data_clean) %in% names(high_vif)))

data_clean <- data_clean[, !(names(data_clean) %in% names(high_vif)) ]
head(data_clean)
dim(data_clean)
summary(data_clean)

# Use findCorrelation to identify highly correlated variables
correlation_matrix <- cor(data_clean)
correlated_vars <- findCorrelation(correlation_matrix, cutoff = 0.8, verbose = TRUE)
correlated_vars

1:ncol(data_clean) %in% correlated_vars

data_clean <- data_clean[,!(1:ncol(data_clean) %in% correlated_vars)]
dim(data_clean)
# information gain
#df <- data_clean
#df$o_bullied <- factor(df$o_bullied)
#bullied.infogain <- InfoGainAttributeEval(o_bullied ~ . , data = df)
#sorted.features <- sort(bullied.infogain, decreasing = TRUE)
# sorted.features
# names(sorted.features[sorted.features>0])
# (names(data_clean) %in% names(sorted.features[sorted.features>0]))
#df <- data_clean[,(names(data_clean) %in% names(sorted.features[sorted.features>0]))]
#df <- cbind(df,"o_bullied"=data_clean$o_bullied)
#data_clean <- df

#-----------------------------
# Initialize variables to store results
eval_class0_metrics <- data.frame("TPR"=numeric(0),
                                 "FPR"=numeric(0),
                                 "Precision"=numeric(0),
                                 "Recall"=numeric(0),
                                 "F-Measure"=numeric(0),
                                 "ROC"=numeric(0),
                                 "MCC"=numeric(0),
                                 "Kappa"=numeric(0))

eval_class1_metrics <- data.frame("TPR"=numeric(0),
                                 "FPR"=numeric(0),
                                 "Precision"=numeric(0),
                                 "Recall"=numeric(0),
                                 "F-Measure"=numeric(0),
                                 "ROC"=numeric(0),
                                 "MCC"=numeric(0),
                                 "Kappa"=numeric(0))

eval_weight_metrics <- data.frame("TPR"=numeric(0),
                                  "FPR"=numeric(0),
                                  "Precision"=numeric(0),
                                  "Recall"=numeric(0),
                                  "F-Measure"=numeric(0),
                                  "ROC"=numeric(0),
                                  "MCC"=numeric(0),
                                  "Kappa"=numeric(0))

set.seed(31)  # Set a seed for reproducibility
apply(data_clean,2,class)
data <- read.csv("/Users/jonathanko/Downloads/preprocess_data-2.csv", 
                 header = TRUE)
data_clean <- data

df_regression <- data_clean
#df_regression$o_bullied <- as.factor(df_regression$o_bullied)
#sapply(df_regression,class)

# Create an initial split for your data
split_data <- initial_split(df_regression, prop = 2/3, strata = o_bullied)  # 70% training, 30% testing

# Extract the training and testing sets
data_train <- training(split_data)
data_train <- data_train[sample(1:nrow(data_train)), ]
data_test <- testing(split_data)
data_test <- data_test[sample(1:nrow(data_test)), ]

data_train_rose <- ovun.sample(o_bullied ~ ., data = data_train, method = "under")$data
data_train <- data_train_rose

# Apply Labels to training and testing set
train_data_labels <- lapply(data_train$o_bullied, as.numeric)
train_data_numeric <- data_train %>%
  select(-o_bullied)
train_data_df <- data.matrix(train_data_numeric)

test_data_labels <- lapply(data_test$o_bullied, as.numeric)
test_data_numeric <- data_test %>%
  select(-o_bullied)
test_data_df <- data.matrix(test_data_numeric)

# training data
data_train <- train_data_df
train_labels <- train_data_labels
test_labels_factor <- unlist(test_labels)
test_labels_factor <- factor(test_labels_factor)

# testing data
data_test <- test_data_df
test_labels <- test_data_labels

# convert data to matrix for xgboost
dtrain <- xgb.DMatrix(data = data_train, label= train_labels)
dtest <- xgb.DMatrix(data = data_test, label= test_labels)

for (i in 1:10) {
  # split_data <- initial_split(df_regression, prop = 2/3, strata = o_bullied)  # 70% training, 30% testing
  # data_train <- training(split_data)
  # data_test <- testing(split_data)
  # 
  # data_train_rose <- ovun.sample(o_bullied ~ ., data = data_train, method = "under")$data
  # data_train <- data_train_rose
  # get the number of negative & positive cases in our data
  negative_cases <- sum(train_labels == FALSE)
  postive_cases <- sum(train_labels == TRUE)
  # build a Naïve Bayes model from training dataset
  model <- xgboost(data = dtrain, # the data           
                         max.depth = 10, # the maximum depth of each decision tree; max.depth = 4; max.depth = 7; 8
                         nround = 100, # number of boosting rounds; nround = 20; nround = 25
                         colsample_bytree = 1, subsample = 1,
                         early_stopping_rounds = 3, # if we dont see an improvement in this many rounds, stop
                         objective = "binary:logistic", # the objective function
                         scale_pos_weight = negative_cases/postive_cases, # control for imbalanced classes
                   verbose = FALSE,
                   gamma = 0,
                         min_child_weight = 10, #min_child_weight = 10; min_child_weight = 15; 25
                         learning_rate = 0.25) #learning_rate = 0.25; learning_rate 0.3
  
  
  #with weight average
  class_counts <- table(test_labels_factor)
  class_weights <- 1 / (class_counts / sum(class_counts))
  class_weights
  class_w <- class_weights[[2]] / class_weights[[1]]
  class_w
  #class_weights <- ifelse(test_labels_factor == 0, 1, class_w)
  class_weights <- ifelse(test_labels_factor == 1, class_weights[[2]], class_weights[[1]])  # Adjust weights as needed
  class_weights
  model_weight <- xgboost(data = dtrain, # the data           
                   max.depth = 10, # the maximum depth of each decision tree; max.depth = 4; max.depth = 7
                   nround = 100, # number of boosting rounds; nround = 20; nround = 25
                   colsample_bytree = 1, subsample = 1,
                   early_stopping_rounds = 3, # if we dont see an improvement in this many rounds, stop
                   objective = "binary:logistic", # the objective function
                   scale_pos_weight = negative_cases/postive_cases, # control for imbalanced classes
                   min_child_weight = 10, #min_child_weight = 10; min_child_weight = 15
                   verbose = FALSE,
                   gamma = 0,
                   weight = class_weights,
                   learning_rate = 0.25) #learning_rate = 0.25; learning_rate 0.3
  # Make predictions on the test sets
  
  predictions <- predict(model, newdata = dtest)
  #lambda = 2,
  #alpha = 1,
  #gamma = 1) # add a regularization term
  
  #pred <- predict(model_tuned, dtest)
  #pred <- as.factor(pred > 0.5)
  #pred <- factor(as.numeric(pred)-1)
  
  predictions_weight <- predict(model_weight, newdata = dtest)
  
  # Evaluate the models
  confusion_matrix <- table(Actual = test_labels_factor, Prediction = ifelse(as.numeric(predictions) > 0.5, 1, 0))
  
  # confusion_matrix[1,1] #TP
  # confusion_matrix[1,2] #FN
  # confusion_matrix[2,1] #FP
  # confusion_matrix[2,2] #TN
  
  # Class O
  #TPR FPR
  tp <- confusion_matrix[1,1]
  fn <- confusion_matrix[1,2]
  fp <- confusion_matrix[2,1]
  tn<- confusion_matrix[2,2]
  
  tpr <- tp / (tp+fn)
  fpr <- fp / (fp+tn)
  
  #Precision
  precision <- tp / (tp+fp)
  
  #Recall
  recall <- tp/(tp+fn)
  
  #F-measure
  f_measure <- 2*precision*recall / (precision+recall)

  #roc
  y_true <- test_labels_factor
  y_scores <- as.numeric(predictions)
  roc_obj <- roc(y_true, y_scores)
  auc_value <- auc(roc_obj)
  
  #mcc
  numerator <- tp*tn - fp*fn
  denominator <- sqrt((tp+fp)*(tp+fn))*sqrt((tn+fp)*(tn+fn))
  # numerator <- confusion_matrix[1,1]*confusion_matrix[2,2] - confusion_matrix[2,1]*confusion_matrix[1,2]
  # denominator <- sqrt((confusion_matrix[1,1]+confusion_matrix[2,1])*(confusion_matrix[1,1]+confusion_matrix[1,2]))*sqrt((confusion_matrix[2,2]+confusion_matrix[2,1])*(confusion_matrix[2,2]+confusion_matrix[1,2]))
  mcc <- numerator / denominator
  
  #kappa
  po <- (tp+tn)/(tp+tn+fn+fp)
  pe1 <- (tp+fn)/(tp+tn+fn+fp)*(tp+fp)/(tp+tn+fn+fp)
  pe2 <- (fp+tn)/(tp+tn+fn+fp)*(fn+tn)/(tp+tn+fn+fp)
  
  pe <- pe1+pe2
  
  Kappa <- (po - pe)/(1-pe)
  
  
  new_eval0 <- data.frame("TPR"=tpr
                         ,"FPR"=fpr
                         ,"Precision"=precision
                         ,"Recall"=recall
                         ,"F-Measure"=f_measure
                         ,"ROC"=auc_value
                         ,"MCC"=mcc
                         ,"Kappa"=Kappa)
  # Class 1
  # confusion_matrix[1,1] #TN
  # confusion_matrix[1,2] #FP
  # confusion_matrix[2,1] #FN
  # confusion_matrix[2,2] #TP
  #TPR FPR
  tn <- confusion_matrix[1,1]
  fp <- confusion_matrix[1,2]
  fn <- confusion_matrix[2,1]
  tp<- confusion_matrix[2,2]
  
  tpr <- tp / (tp+fn)
  fpr <- fp / (fp+tn)
  
  #Precision
  precision <- tp / (tp+fp)
  
  #Recall
  recall <- tp/(tp+fn)
  
  #F-measure
  f_measure <- 2*precision*recall / (precision+recall)
  
  #roc
  y_true <- test_labels_factor
  y_scores <- as.numeric(predictions)
  roc_obj <- roc(y_true, y_scores)
  auc_value <- auc(roc_obj)
  
  #mcc
  numerator <- tp*tn - fp*fn
  denominator <- sqrt((tp+fp)*(tp+fn))*sqrt((tn+fp)*(tn+fn))
  
  #numerator <- confusion_matrix[1,1]*confusion_matrix[2,2] - confusion_matrix[2,1]*confusion_matrix[1,2]
  #denominator <- sqrt((confusion_matrix[1,1]+confusion_matrix[2,1])*(confusion_matrix[1,1]+confusion_matrix[1,2]))*sqrt((confusion_matrix[2,2]+confusion_matrix[2,1])*(confusion_matrix[2,2]+confusion_matrix[1,2]))
  mcc <- numerator / denominator
  
  #kappa
  po <- (tp+tn)/(tp+tn+fn+fp)
  pe1 <- (tp+fn)/(tp+tn+fn+fp)*(tp+fp)/(tp+tn+fn+fp)
  pe2 <- (fp+tn)/(tp+tn+fn+fp)*(fn+tn)/(tp+tn+fn+fp)
  
  pe <- pe1+pe2
  
  Kappa <- (po - pe)/(1-pe)
  
  
  new_eval1 <- data.frame("TPR"=tpr
                          ,"FPR"=fpr
                          ,"Precision"=precision
                          ,"Recall"=recall
                          ,"F-Measure"=f_measure
                          ,"ROC"=auc_value
                          ,"MCC"=mcc
                          ,"Kappa"=Kappa)
  print(new_eval0)
  print(new_eval1)
  print("Normal Model")
  print(confusion_matrix)
  confusion_matrix_weight <- table(Actual = test_labels_factor, Prediction = ifelse(as.numeric(predictions) > 0.5 , 1, 0))
  confusion_matrix <- confusion_matrix_weight
  
  # Wt. Average
  # confusion_matrix[1,1] #TN
  # confusion_matrix[1,2] #FP
  # confusion_matrix[2,1] #FN
  # confusion_matrix[2,2] #TP
  #TPR FPR
  tn <- confusion_matrix[1,1]
  fp <- confusion_matrix[1,2]
  fn <- confusion_matrix[2,1]
  tp<- confusion_matrix[2,2]
  
  tpr <- tp / (tp+fn)
  fpr <- fp / (fp+tn)
  
  #Precision
  precision <- tp / (tp+fp)
  
  #Recall
  recall <- tp/(tp+fn)
  
  #F-measure
  f_measure <- 2*precision*recall / (precision+recall)
  
  #roc
  y_true <- test_labels_factor
  y_scores <- as.numeric(predictions_weight)
  roc_obj <- roc(y_true, y_scores)
  auc_value <- auc(roc_obj)
  
  #mcc
  numerator <- tp*tn - fp*fn
  denominator <- sqrt((tp+fp)*(tp+fn))*sqrt((tn+fp)*(tn+fn))
  
  #numerator <- confusion_matrix[1,1]*confusion_matrix[2,2] - confusion_matrix[2,1]*confusion_matrix[1,2]
  #denominator <- sqrt((confusion_matrix[1,1]+confusion_matrix[2,1])*(confusion_matrix[1,1]+confusion_matrix[1,2]))*sqrt((confusion_matrix[2,2]+confusion_matrix[2,1])*(confusion_matrix[2,2]+confusion_matrix[1,2]))
  mcc <- numerator / denominator
  
  #kappa
  po <- (tp+tn)/(tp+tn+fn+fp)
  pe1 <- (tp+fn)/(tp+tn+fn+fp)*(tp+fp)/(tp+tn+fn+fp)
  pe2 <- (fp+tn)/(tp+tn+fn+fp)*(fn+tn)/(tp+tn+fn+fp)
  
  pe <- pe1+pe2
  
  Kappa <- (po - pe)/(1-pe)
  
  
  new_eval_weight <- data.frame("TPR"=tpr
                          ,"FPR"=fpr
                          ,"Precision"=precision
                          ,"Recall"=recall
                          ,"F-Measure"=f_measure
                          ,"ROC"=auc_value
                          ,"MCC"=mcc
                          ,"Kappa"=Kappa)
  print("Weight Model")
  print(confusion_matrix)
  # Calculate the evaluation metric of interest
  eval_class0_metrics <- rbind(eval_class0_metrics,new_eval0)
  eval_class1_metrics <- rbind(eval_class1_metrics,new_eval1)
  eval_weight_metrics <- rbind(eval_weight_metrics,new_eval_weight)
}

#print(eval_class_metrics)
eval_class0_metrics
eval_class1_metrics
eval_weight_metrics
# Calculate the mean and standard deviation of the evaluation metric
mean_eval_class0_metric <- apply(eval_class0_metrics,2,mean)
mean_eval_class1_metric <- apply(eval_class1_metrics,2,mean)
mean_eval_weight_metric <- apply(eval_weight_metrics,2,mean)

# Print the results
mean_eval_class <- rbind("Class_0"=mean_eval_class0_metric,
                         "Class_1"=mean_eval_class1_metric,
                         "Wt.Average"=mean_eval_weight_metric)
print("Evaluation for Logistic Regression")
mean_eval_class

#-----------------------------