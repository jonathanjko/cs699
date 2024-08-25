# Theerarun Tubnonghee
# Jonathan Ko
# CS699
# Logistic Regression

# Load necessary libraries
#install.packages("keras")
#install.packages("pRoc")
#install.packages("ROSE")
library(caret) # for nearZeroVar
library(car) # for vif
library(rsample)
library(pROC)
library(ROSE)
library(FSelector)
library(RWeka)
library(nnet)
library(keras)

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
data <- read.csv("preprocess_data.csv")
# data_train <- read.csv("initial_train.csv")
# data_test <- read.csv("initial_split.csv")

# Create an initial split for your data
split_data <- initial_split(data, prop = 2/3, strata = o_bullied)  # 70% training, 30% testing

# Extract the training and testing sets
data_train <- training(split_data)
data_test <- testing(split_data)

# rebalancing classes using ROSE
data_train_rose <- ovun.sample(o_bullied ~ ., data = data_train, method = "under")$data
data_train <- data_train_rose
data_train$o_bullied <- as.factor(data_train$o_bullied)

# for (i in 1:10) {

  #no weight average
  # o_bullied.cfs
  # model <- glm(o_bullied.cfs, data = data_train, family = "binomial")
  model <- glm(o_bullied ~ ., data = data_train, family = "binomial")
  # model
  
  #with weight average
  # calculate weights to average classes
  class_counts <- table(data_train$o_bullied)
  class_weights <- 1 / (class_counts / sum(class_counts))
  class_weights
  class_w <- class_weights[[2]] / class_weights[[1]]
  # class_w
  #class_weights <- ifelse(data_train$o_bullied == 0, 1, class_w)
  class_weights <- ifelse(data_train$o_bullied == 1, class_weights[[2]], class_weights[[1]])  # Adjust weights as needed
  table(class_weights)
  model_weight <- glm(o_bullied ~ ., data = data_train, family = "binomial", weights = class_weights)
  
  # Make predictions on the test sets
  predictions <- predict(model, newdata = data_test, type = "response")
  predictions_weight <- predict(model_weight, newdata = data_test, type = "response")
  
  # Evaluate the models
  confusion_matrix <- table(Actual = data_test$o_bullied, Prediction = ifelse(predictions > 0.5, 1, 0))
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
  y_true <- data_test$o_bullied
  y_scores <- as.vector(predictions)
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
  y_true <- data_test$o_bullied
  y_scores <- as.vector(predictions)
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
  # print(new_eval0)
  # print(new_eval1)
  # print("Normal Model")
  # print(confusion_matrix)
  confusion_matrix_weight <- table(Actual = data_test$o_bullied, Prediction = ifelse(predictions_weight > 0.5 , 1, 0))
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
  y_true <- data_test$o_bullied
  y_scores <- as.vector(predictions_weight)
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
  # print("Weight Model")
  # print(confusion_matrix)
  # Calculate the evaluation metric of interest
  eval_class0_metrics <- rbind(eval_class0_metrics,new_eval0)
  eval_class1_metrics <- rbind(eval_class1_metrics,new_eval1)
  eval_weight_metrics <- rbind(eval_weight_metrics,new_eval_weight)
# }

#print(eval_class_metrics)
# eval_class0_metrics
# eval_class1_metrics
# eval_weight_metrics
# Calculate the mean and standard deviation of the evaluation metric
mean_eval_class0_metric <- apply(eval_class0_metrics,2,mean)
mean_eval_class1_metric <- apply(eval_class1_metrics,2,mean)
mean_eval_weight_metric <- apply(eval_weight_metrics,2,mean)

# Print the results
mean_eval_class <- rbind("Class_0"=mean_eval_class0_metric,
                         "Class_1"=mean_eval_class1_metric,
                         "Wt.Average"=mean_eval_weight_metric)
print("Evaluation for Logistic Regression")
glm_eval <- mean_eval_class
glm_eval

#-----------------------------