import joblib

model = joblib.load("regression.joblib")  # Linear regression model

features = [[20.0, 4.0, 1.0]]
print(model.predict(features))