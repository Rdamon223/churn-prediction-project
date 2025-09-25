import pandas as pd  # For loading and manipulating the dataset
from sklearn.model_selection import train_test_split  # To split data into training and testing sets for evaluation
from sklearn.preprocessing import StandardScaler  # To scale numerical features for better model performance
from sklearn.ensemble import RandomForestClassifier  # The ML model for classification (random forest is robust for this data)
from sklearn.metrics import classification_report, confusion_matrix  # To evaluate model accuracy, recall, etc.
import joblib  # To save the trained model and scaler for later use in deployment

# Load the dataset - why? To get the raw data for processing and training
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Preprocess data - why? Raw data has strings and missing values; models need clean numbers
# Handle missing values in TotalCharges (some are blank spaces)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')  # Convert to number, set invalid to NaN
df = df.dropna(subset=['TotalCharges'])  # Drop rows with NaN in TotalCharges (small number)

# Encode categorical features to numbers - why? Models like random forest can't handle strings directly
categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 
                    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
                    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)  # One-hot encode (creates binary columns)

# Features (X) and target (y) - why? X is inputs (e.g., tenure), y is output (Churn: Yes/No)
X = df.drop(['customerID', 'Churn'], axis=1)  # Drop ID (irrelevant) and target
y = df['Churn'].map({'Yes': 1, 'No': 0})  # Convert Churn to 0/1

# Split data - why? Train on one part, test on unseen to measure real performance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features - why? Normalizes numbers (e.g., tenure vs charges) for fair model weighting
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model with class weighting - why? Handles imbalance by penalizing errors on minority class (churn=1) more
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')  # 'balanced' auto-adjusts weights
model.fit(X_train, y_train)

# Evaluate - why? Checks if the model is good (aim for >80% accuracy, high recall for churn)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Save model and scaler - why? For reuse in the API without retraining every time
joblib.dump(model, 'churn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model trained and saved!")