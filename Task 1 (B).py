# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Import scikit-learn modules
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer

# Load the Iris dataset into a Pandas DataFrame
df = pd.DataFrame(data=load_iris().data, columns=load_iris().feature_names)
df["Target"] = load_iris().target  # Add target labels

# Handle missing values (though Iris has none)
imputer = SimpleImputer(strategy="mean")
df[df.columns] = imputer.fit_transform(df)

# Standardize feature values
scaler = StandardScaler()
features = load_iris().feature_names  # Feature column names
df[features] = scaler.fit_transform(df[features])  # Scale features

# Split dataset into training (80%) and testing (20%) sets with stratification
train_df, test_df = train_test_split(df, test_size=0.2, random_state=53, stratify=df["Target"])

# Train a Random Forest model
rfc = RandomForestClassifier(n_estimators=100, random_state=53)
rfc.fit(train_df.drop("Target", axis=1), train_df["Target"])

# Perform 7-fold cross-validation
cv_score = cross_val_score(rfc, train_df.drop("Target", axis=1), train_df["Target"], cv=7, scoring="accuracy")

# Print model evaluation results
print("Cross-validation scores:", cv_score)
print("Mean accuracy:", np.mean(cv_score))
print("Standard deviation:", np.std(cv_score))

# Make predictions
y_preds = rfc.predict(test_df.drop("Target", axis=1))

# Create DataFrame for plotting
plot_df = pd.DataFrame({"True": test_df["Target"], "Predicted": y_preds})

# Scatter plot of true vs. predicted labels
sns.scatterplot(x="True", y="Predicted", data=plot_df)
plt.plot(plot_df["True"], plot_df["True"], color="red", linestyle="dashed")  # Diagonal reference line
plt.xlabel("True Labels")
plt.ylabel("Predicted Labels")
plt.title("True vs Predicted Labels")
plt.show()
