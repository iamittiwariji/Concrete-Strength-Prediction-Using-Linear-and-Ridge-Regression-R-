# Note: Required packages are listed in README.md

# 1. Setup: Load libraries and data ----------------------------------------


# Load required libraries (assuming they are already installed)
library(tidyverse)    # Includes dplyr, ggplot2, etc., for data manipulation and visualization
library(readxl)       # To read Excel files
library(caret)        # For data splitting and model evaluation
library(glmnet)       # For ridge regression modeling
library(corrplot)     # For correlation matrix visualization

# Read the dataset (Excel file) from the data directory
data <- read_excel("data/concrete compressive strength.xlsx")

# (Optional) Rename columns for convenience
colnames(data) <- c("Cement", "BlastFurnaceSlag", "FlyAsh", "Water", 
                    "Superplasticizer", "CoarseAggregate", "FineAggregate", "Age", 
                    "ConcreteCategory", "ContainsFlyAsh", "CompressiveStrength")

# Convert appropriate columns to factors
data$ConcreteCategory <- as.factor(data$ConcreteCategory)
# (ContainsFlyAsh is 0/1 numeric; we leave it as numeric for modeling)

# Get a quick overview of the data
cat("Dataset contains", nrow(data), "observations and", ncol(data), "variables.\n")
str(data)    # Structure of the dataset (data types and sample)
head(data)   # First few rows of the dataset
summary(data) # Summary statistics (min, median, mean, etc. for numeric columns)

# 2. Exploratory Data Analysis (EDA) ---------------------------------------

# Calculate summary statistics (mean and standard deviation) for select key variables
data %>% 
  summarise(
    Cement_mean = mean(Cement, na.rm = TRUE),
    Cement_sd   = sd(Cement, na.rm = TRUE),
    Water_mean  = mean(Water, na.rm = TRUE),
    Water_sd    = sd(Water, na.rm = TRUE),
    Strength_mean = mean(CompressiveStrength, na.rm = TRUE),
    Strength_sd   = sd(CompressiveStrength, na.rm = TRUE)
  )

# Interpretation:
# The above provides the average and variability of cement, water, and strength.
# For instance, CompressiveStrength_mean and _sd give an idea of typical strength values and spread.

# 3. Data Visualization ---------------------------------------------------

# Histogram of the target variable (Compressive Strength distribution)
ggplot(data, aes(x = CompressiveStrength)) +
  geom_histogram(binwidth = 5, fill = "blue", color = "black", alpha = 0.7) +
  labs(title = "Distribution of Concrete Compressive Strength",
       x = "Compressive Strength (MPa)",
       y = "Frequency")
# Optional: Save the plot to a file
#ggsave("strength_distribution.png", width = 6, height = 4)

# Boxplot of Compressive Strength by Concrete Category (e.g., Coarse vs Fine aggregate mix)
ggplot(data, aes(x = ConcreteCategory, y = CompressiveStrength)) +
  geom_boxplot(fill = "lightblue") +
  labs(title = "Compressive Strength by Concrete Category",
       x = "Concrete Category",
       y = "Compressive Strength (MPa)")
# Optional: Save the plot
# ggsave("strength_by_category.png", width = 6, height = 4)

# Scatterplot: Cement vs. Compressive Strength (to see correlation between cement content and strength)
ggplot(data, aes(x = Cement, y = CompressiveStrength)) +
  geom_point(color = "darkblue", alpha = 0.6) +
  labs(title = "Cement Content vs. Compressive Strength",
       x = "Cement (kg in mixture)",
       y = "Compressive Strength (MPa)")
# Optional: Save the plot
# ggsave("cement_vs_strength.png", width = 6, height = 4)

# Scatterplot: Water vs. Compressive Strength (to examine relationship between water content and strength)
ggplot(data, aes(x = Water, y = CompressiveStrength)) +
  geom_point(color = "darkgreen", alpha = 0.6) +
  labs(title = "Water Content vs. Compressive Strength",
       x = "Water (kg in mixture)",
       y = "Compressive Strength (MPa)")
# Optional: Save the plot
# ggsave("water_vs_strength.png", width = 6, height = 4)

# Boxplot: Effect of Fly Ash presence on Compressive Strength
ggplot(data, aes(x = as.factor(ContainsFlyAsh), y = CompressiveStrength)) +
  geom_boxplot(fill = "lightgreen") +
  labs(title = "Impact of Fly Ash on Compressive Strength",
       x = "Contains Fly Ash (0 = No, 1 = Yes)",
       y = "Compressive Strength (MPa)")
# Optional: Save the plot
# ggsave("flyash_effect.png", width = 6, height = 4)

# 4. Correlation Analysis -------------------------------------------------

# Select only numeric columns for correlation analysis (includes the target as well)
numeric_data <- data %>% select_if(is.numeric)

# Compute the correlation matrix for numeric variables (using complete observations)
cor_matrix <- cor(numeric_data, use = "complete.obs")

# Examine the correlation matrix values
print(cor_matrix)

# Visualize the correlation matrix
corrplot(cor_matrix, method = "color", type = "upper", diag = FALSE,
         tl.cex = 0.8, addCoef.col = "black",
         title = "Correlation Matrix of Concrete Variables", mar = c(0,0,2,0))
# (Above, 'diag = FALSE' ensures self-correlations are not displayed)
 png("correlation_matrix.png", width = 6, height = 5, units = "in", res = 300); 
  corrplot(
    cor_matrix,
    method = "color",
    type = "upper",
    diag = FALSE,
    tl.cex = 0.8,
    addCoef.col = "black",
    title = "Correlation Matrix of Concrete Variables",
    mar = c(0, 0, 2, 0)
  )
  dev.off()

# Identify strongly correlated pairs (absolute correlation > 0.7, excluding self-correlation)
cat("Highly correlated pairs (corr > 0.7):\n")
for(i in 1:(ncol(cor_matrix)-1)) {
  for(j in (i+1):ncol(cor_matrix)) {
    if(abs(cor_matrix[i, j]) > 0.7) {
      cat(colnames(cor_matrix)[i], "and", colnames(cor_matrix)[j], 
          "have correlation", round(cor_matrix[i, j], 3), "\n")
    }
  }
}

# 5. Modeling and Prediction ----------------------------------------------

# Split the data into training and testing sets (80% training, 20% testing)
set.seed(123)  # for reproducibility
train_index <- createDataPartition(data$CompressiveStrength, p = 0.8, list = FALSE)
train_data  <- data[train_index, ]
test_data   <- data[-train_index, ]

# Ensure factor levels in test set match training (especially for ConcreteCategory)
test_data$ConcreteCategory <- factor(test_data$ConcreteCategory, levels = levels(train_data$ConcreteCategory))

## 5.1 Baseline Linear Regression (Ordinary Least Squares)

# Train a linear regression model using all predictors on the training set
lin_model <- lm(CompressiveStrength ~ ., data = train_data)

# Summarize the linear model (optional)
summary(lin_model)  # This will output coefficients, R-squared, etc.
# Note: In a script for analysis, one might examine the summary to check model coefficients and significance.

# Predict on the test set using the linear model
pred_lm <- predict(lin_model, newdata = test_data)

## 5.2 Ridge Regression (L2 Regularization)

# Prepare matrix of features (predictor variables) for ridge regression
# We use model.matrix to create a numeric matrix (including dummy variables for factors)
x_train <- model.matrix(CompressiveStrength ~ ., data = train_data)
x_train <- x_train[, -1]  # drop the intercept column
y_train <- train_data$CompressiveStrength

x_test <- model.matrix(CompressiveStrength ~ ., data = test_data)
x_test <- x_test[, -1]    # drop intercept to match the training matrix
y_test <- test_data$CompressiveStrength

# Perform cross-validation to find the optimal lambda for ridge
set.seed(123)  # for reproducibility of CV
cv_ridge <- cv.glmnet(x_train, y_train, alpha = 0)  # 10-fold CV by default

# Plot cross-validation results (optional visualization of CV errors vs lambda)
plot(cv_ridge)
# Optional: Save the CV plot
  ggsave("ridge_cv_plot.png", width = 5, height = 5)

# Optimal lambda value that minimizes CV error
opt_lambda <- cv_ridge$lambda.min
cat("Optimal lambda for ridge:", opt_lambda, "\n")

# Train the final Ridge Regression model on the full training data using the optimal lambda
ridge_model <- glmnet(x_train, y_train, alpha = 0, lambda = opt_lambda)

# Examine the ridge model coefficients (at the optimal lambda)
coef(ridge_model)

# Plot the coefficient paths for different lambda values (from the original glmnet fit on train data)
# (This requires fitting across a range of lambdas; we can use the cv.glmnet object for illustration)
plot(cv_ridge$glmnet.fit, xvar = "lambda", label = TRUE)
title("Ridge Coefficient Paths", line = 2.5)
# Optional: Save the plot
# ggsave("ridge_coefficient_paths.png", width = 6, height = 4)

# Predict on the test set using the ridge model
pred_ridge <- predict(ridge_model, newx = x_test, s = opt_lambda)
pred_ridge <- as.numeric(pred_ridge)  # ensure prediction is a numeric vector

# 6. Model Evaluation and Comparison --------------------------------------

# Define a function to compute Root Mean Squared Error (optional utility)
rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2))
}

# Calculate RMSE for both models on the test set
lm_rmse   <- rmse(y_test, pred_lm)
ridge_rmse <- rmse(y_test, pred_ridge)

# Print out the RMSE results for comparison
cat(sprintf("Linear Regression RMSE (Test Set): %.3f MPa\n", lm_rmse))
cat(sprintf("Ridge Regression RMSE (Test Set): %.3f MPa\n", ridge_rmse))

# If desired, one can compare which model performs better
if (ridge_rmse < lm_rmse) {
  cat("Ridge Regression has a lower test RMSE, indicating better predictive performance on the test set.\n")
} else {
  cat("Linear Regression has a lower or equal test RMSE on the test set (Ridge did not show improvement in this case).\n")
}
