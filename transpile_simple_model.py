import joblib
import os
from jinja2 import Template

model = joblib.load("regression.joblib") # Linear regression model

thetas = model.coef_.tolist()
print("Model coefficients:", thetas)
thetas.insert(0, 0.0)  # Null bias
n_coeffs = len(thetas)

template = Template("""
#include <stdio.h>
float thetas[{{ n_coeffs }}] = { {% for theta in thetas %}{{ theta }}{% if not loop.last %}, {% endif %}{% endfor %} };

float prediction(float *features, int n_feature)
{
    if (n_feature != {{ n_coeffs }} - 1)
    {
        printf("Error: expected %d features, got %d\\n", {{ n_coeffs }} - 1, n_feature);
        return -1;
    }
    int i = 0;
    float r = thetas[0];
    while (i < n_feature)
    {
        r += thetas[i + 1] * features[i];
        i++;
    }
    return r;
}
                    
int main(void)
{
    float features[{{ n_coeffs }} - 1] = {20.0, 4.0, 1.0}; // Example features
    float result = prediction(features, {{ n_coeffs }} - 1);
    if (result == -1) {
        return 1;
    }
    printf("Prediction: %f\\n", result);
    return 0;
}
""")

code = template.render(thetas=thetas, n_coeffs=n_coeffs)
with open("regression_model.c", "w") as f:
    f.write(code)

exit_status = os.system("gcc regression_model.c -o regression_model")
if exit_status != 0:
    print("Compilation failed.")
else:
    print("Compiled regression_model.c to regression_model executable.")