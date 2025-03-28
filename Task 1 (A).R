# Ensure required packages are installed and loaded
packages <- c("caret", "dplyr", "tidyr", "rpart", "ggplot2")

# Loop through each package and install if not already installed
for (pkg in packages) {
    if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
        install.packages(pkg, dependencies = TRUE)  # Install missing package
        library(pkg, character.only = TRUE)  # Load package after installation
    }
}

# Load the built-in iris dataset
data(iris)

# Convert the 'Species' column to a factor and remove any missing values
iris <- iris %>%
  mutate(Species = as.factor(Species)) %>%
  drop_na()

# Display the structure of the dataset (column types and sample data)
glimpse(iris)

# Set a random seed for reproducibility
set.seed(53)

# Create a training index: 80% of the data is used for training
trainIndex <- createDataPartition(iris$Species, p = 0.8, list = FALSE)

# Split the dataset into training (80%) and testing (20%) sets
trainData <- iris %>% slice(trainIndex)
testData <- iris %>% slice(-trainIndex)

# Train a decision tree model using cross-validation (10-fold)
model <- train(Species ~ .,  # Predict Species using all other features
               data = trainData,
               method = "rpart",  # Use the decision tree method
               trControl = trainControl(method = "cv", number = 10))  # 10-fold cross-validation

# Print the trained model details
print(model)

# Make predictions on the test set
predictions <- predict(model, testData)

# Compute the confusion matrix to evaluate the model performance
confMatrix <- confusionMatrix(predictions, testData$Species)
print(confMatrix)  # Display classification results

# Add predictions as a new column to the test dataset for visualization
testData <- testData %>%
  mutate(Predicted = as.factor(predictions))

# Create a scatter plot of actual vs. predicted species classifications
ggplot(testData, aes(x = Species, y = Predicted, color = Predicted)) +
  geom_jitter(width = 0.2, size = 4, alpha = 0.8) +  # Add slight jitter to separate overlapping points
  labs(title = "Actual vs Predicted Species",  # Set plot title
       x = "Actual Species",  # Label x-axis
       y = "Predicted Species",  # Label y-axis
       color = "Predicted") +  # Label legend
  theme_minimal()  # Use a clean, minimalistic theme

