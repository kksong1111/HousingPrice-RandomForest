library(here)
library(ranger)

# Set working directory to file root folder and load the data
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
f <- read.csv(file = 'train.csv', header = TRUE)

# Shuffle the data
set.seed(42)
f <- f[sample(nrow(f)), ]

# Split into training set and test set
train_data <- f[1:1200, ]
test_data <- f[1201:nrow(f), ]

# Remove the headers
train_data_rf <- train_data[, -1]
test_ids <- test_data$Id
test_features_rf <- test_data[, -1]



# The test results for our models
results <- data.frame(Model = character(), num_trees = integer(), rmse = numeric(), stringsAsFactors = FALSE)

# Simple prediction using mean
mean_prediction <- mean(train_data$SalePrice)
predictions_mean <- rep(mean_prediction, length(test_data$SalePrice))
rmse_mean <- sqrt(sum((predictions_mean - test_data$SalePrice)^2) / length(test_data$SalePrice))

results <- rbind(results, data.frame(Model = "Mean Prediction", num_trees = NA, rmse = rmse_mean))

cat("Mean Prediction RMSE:", rmse_mean, "\n\n")

# Linear Regression on LotArea
train_data_lm <- train_data[, c("LotArea", "SalePrice")]
test_feature <- test_data[, "LotArea", drop = FALSE]
lm_model <- lm(SalePrice ~ LotArea, data = train_data_lm)

predictions_lm <- predict(lm_model, newdata = test_feature)

# Evaluate model performance using RMSE
rmse_lm <- sqrt(mean((predictions_lm - test_data$SalePrice)^2))

cat("Linear Regression Model RMSE:", rmse_lm, "\n")

results <- rbind(results, data.frame(Model = "Linear Regression", num_trees = NA, rmse = rmse_lm))

# Different num.trees for Random Forest
num_trees_values <- c(50, 80, 100, 200)


# Handle missing values in training data
train_data_rf[sapply(train_data_rf, is.numeric)] <- lapply(train_data_rf[sapply(train_data_rf, is.numeric)], function(x) {
  ifelse(is.na(x), median(x, na.rm = TRUE), x)
})

train_data_rf[sapply(train_data_rf, is.character)] <- lapply(train_data_rf[sapply(train_data_rf, is.character)], function(x) {
  x[is.na(x)] <- "Missing"
  return(as.factor(x))
})

# Handle missing values in test data
test_features_rf[sapply(test_features_rf, is.numeric)] <- lapply(test_features_rf[sapply(test_features_rf, is.numeric)], function(x) {
  ifelse(is.na(x), median(x, na.rm = TRUE), x)
})

test_features_rf[sapply(test_features_rf, is.character)] <- lapply(test_features_rf[sapply(test_features_rf, is.character)], function(x) {
  x[is.na(x)] <- "Missing"
  return(as.factor(x))
})



for (num_trees in num_trees_values) {
  # Random forest model
  bag.y <- ranger(SalePrice ~ ., data = train_data_rf, mtry = ncol(train_data_rf) - 1, num.trees = num_trees, respect.unordered.factors = "order")
  
  # Random forest predictions
  predictions_rf <- predict(bag.y, data = test_features_rf)$predictions
  
  # Calculate RMSE for RF
  actual <- test_data$SalePrice
  rmse_rf <- sqrt(sum((predictions_rf - actual)^2) / length(actual))
  
  cat("num.trees:", num_trees, "RMSE:", rmse_rf, "\n")
  results <- rbind(results, data.frame(Model = "Random Forest", num_trees = num_trees, rmse = rmse_rf))
}

# Arrange results
results <- results[order(match(results$Model, c("Mean Prediction", "Linear Regression", "Random Forest"))), ]

cat("\nModel Performance Results:\n")
print(results)