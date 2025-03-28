
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

# Handle missing values
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

################################# TASK 2 #######################################

# Import ONNX-related libraries
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt

# Convert trained Random Forest model to ONNX format
initial_type = [('float_input', FloatTensorType([None, len(features)]))]
onnx_model = convert_sklearn(rfc, initial_types=initial_type)

# Save the ONNX model
with open("random_forest.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("Model saved as 'random_forest.onnx'")

# Load ONNX model for inference
session = rt.InferenceSession("random_forest.onnx")

# Prepare test data for ONNX inference
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
input_data = {input_name: test_df.drop("Target", axis=1).to_numpy().astype(np.float32)}

# Run inference using ONNX
pred_onnx = session.run([output_name], input_data)[0]

# Print first 5 predictions from the ONNX model
print("Predictions from ONNX model:", pred_onnx[:5])
