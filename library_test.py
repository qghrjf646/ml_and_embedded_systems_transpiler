import joblib
from transpile_simple_model import SimpleTranspiler

model = joblib.load("regression.joblib") # Linear regression model

thetas = model.coef_.tolist()
print("Model coefficients:", thetas)
thetas.insert(0, 0.0)  # Null bias
n_coeffs = len(thetas)
transpiler = SimpleTranspiler(thetas=thetas, n_coeffs=n_coeffs, type="linear_regression")
transpiler.transpile()