import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load the dataset
file_path =r'c:\Users\hp\Downloads\Telegram Desktop\Eartquakes-1990-2023.csv'
data = pd.read_csv(file_path)

print("Dataset Overview:")
print(data.head())
print(data.info())

# Handle missing values
data.fillna(data.mean(numeric_only=True), inplace=True)

# Categorize magnitudes
bins = [0, 4.0, 6.0, 10.0]
labels = ['Low', 'Moderate', 'High']
if 'magnitudo' in data.columns:
    data['magnitude_category'] = pd.cut(data['magnitudo'], bins=bins, labels=labels)
else:
    raise KeyError("The target column 'magnitudo' does not exist in the dataset.")

# Convert magnitude categories to numerical codes
data['magnitude_category'] = data['magnitude_category'].cat.codes

# Define features and target
features = ['longitude', 'latitude', 'depth', 'tsunami', 'significance']
target = 'magnitude_category'

X = data[features]
y = data[target]

# One-hot encode the target variable
y = to_categorical(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the ANN model
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(y_train.shape[1], activation='softmax')  # Output layer with softmax for multi-class classification
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Predict on test data
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Confusion Matrix
print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test_classes, y_pred_classes, target_names=labels))

# Example Prediction
example = [[78.1, 19.1, 10.0, 0, 300]]
prediction = model.predict(np.array(example))
predicted_category = labels[np.argmax(prediction)]
print(f"\nPrediction for {example}: {predicted_category}")
