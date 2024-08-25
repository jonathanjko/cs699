# Theerarun Tubnonghee
# Jonathan Ko
# CS699
# Preprocessing

data <- read.csv("project_dataset.csv")
head(data)
dim(data)

# Load necessary libraries
library(caret) # for nearZeroVar
library(car) # for vif
library(rsample)

# Identify near-zero variance predictors
nzv <- nearZeroVar(data, saveMetrics = TRUE,freqCut = 95/5)
-nzv$nzv

# Print out the predictors with near-zero variance
# print(nzv[nzv$nzv,])

# Remove near-zero variance predictors from the data
data_clean <- data[, !(nzv$nzv | nzv$zeroVar)]
# dim(data_clean)
# head(data_clean)

# Check for multicollinearity using VIF
# We may need to fit a model first, depending on what you are trying to predict. Assuming a linear model:
model <- glm(o_bullied ~ ., data = data_clean)

# Calculate VIF
vif_values <- vif(model)

# Print out the VIF values
# print(vif_values)

# It's often useful to consider a threshold for VIF, such as 5 or 10, to determine which variables might be causing multicollinearity.
high_vif <- vif_values[vif_values > 5] # or you can use 10 as your threshold
# print(high_vif)

names(high_vif)
# names(data_clean)
# sum(!(names(data_clean) %in% names(high_vif)))

data_clean <- data_clean[, !(names(data_clean) %in% names(high_vif)) ]
# head(data_clean)
# dim(data_clean)
# summary(data_clean)

# Use findCorrelation to identify highly correlated variables
correlation_matrix <- cor(data_clean)
correlated_vars <- findCorrelation(correlation_matrix, cutoff = 0.8, verbose = TRUE)
# correlated_vars

# 1:ncol(data_clean) %in% correlated_vars

data_clean <- data_clean[,!(1:ncol(data_clean) %in% correlated_vars)]
# dim(data_clean)

# information gain
df <- data_clean
df$o_bullied <- factor(df$o_bullied)
bullied.infogain <- InfoGainAttributeEval(o_bullied ~ . , data = df)
sorted.features <- sort(bullied.infogain, decreasing = TRUE)
# sorted.features
# names(sorted.features[sorted.features>0])
# (names(data_clean) %in% names(sorted.features[sorted.features>0]))
df <- data_clean[,(names(data_clean) %in% names(sorted.features[sorted.features>0]))]
df <- cbind(df,"o_bullied"=data_clean$o_bullied)
names(df)
# sapply(df,class)
#-----------------------------
set.seed(31)  # Set a seed for reproducibility
apply(data_clean,2,class)

df_regression <- df
# df_regression$o_bullied <- ifelse(df_regression$o_bullied == 1, "Yes", "No")
df_regression$o_bullied <- as.factor(df_regression$o_bullied)
sapply(df_regression,class)

# Create an initial split for your data
split_data <- initial_split(df_regression, prop = 2/3, strata = o_bullied)  # 70% training, 30% testing

# Extract the training and testing sets
data_train <- training(split_data)
data_test <- testing(split_data)

# Export dataset to a CSV file
write.csv(df_regression, "preprocess_data.csv", row.names = FALSE)
write.csv(data_train, "initial_train.csv", row.names = FALSE)
write.csv(data_test, "initial_split.csv", row.names = FALSE)