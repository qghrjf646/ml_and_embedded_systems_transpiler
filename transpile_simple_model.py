import joblib
import os
from jinja2 import Template


logistic_regression_template = Template("""
#include <stdio.h>
float thetas[{{ n_coeffs }}] = { {% for theta in thetas %}{{ theta }}{% if not loop.last %}, {% endif %}{% endfor %} };

float exp_approx(float x, int n_term)
{
    if (x < -1e5)
        return -1e10; // Prevent overflow
    float r = 1;
    int i = 1;
    float p = 1;
    int fact = 1;
    while (i <= n_term)
    {
        p *= x;
        fact *= i;
        r += p / fact;
        i++;
    }
    return r;
}

float sigmoid(float x)
{
    return 1 / (1 + exp_approx(-x, 10));
}                    

float linear_regression(float *features, int n_feature)
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
                    
float prediction(float* features, int n_parameter)
{
    float logits = linear_regression(features, n_parameter);
    return sigmoid(logits);
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

linear_regression_template = Template("""
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

decision_tree_template = Template("""
#include <stdio.h>

float simple_tree(float *features, int n_features)
{
    if (n_features != 2)
    {
        printf("Error: expected 2 features, got %d\\n", n_features);
        return -1;
    }
    return features[0] <= 0 && features[1] <= 0;
}
                                  
int main(void)
{
    float features[2] = {0.0, 0.0}; // Example features
    float result = simple_tree(features, 2);
    if (result == -1) {
        return 1;
    }
    printf("Prediction: %f\\n", result);
    return 0;
}
""")

class SimpleTranspiler:
    def __init__(self, thetas, n_coeffs, type="logistic_regression"):
        self.thetas = thetas
        self.n_coeffs = n_coeffs
        self.type = type

    def transpile(self):
        if self.type == "logistic_regression":
            code = logistic_regression_template.render(thetas=self.thetas, n_coeffs=self.n_coeffs)
        elif self.type == "linear_regression":
            code = linear_regression_template.render(thetas=self.thetas, n_coeffs=self.n_coeffs)
        elif self.type == "decision_tree":
            code = decision_tree_template.render()
        else:
            raise ValueError("Unknown model type")
        
        with open("regression_model.c", "w") as f:
            f.write(code)

        exit_status = os.system("gcc regression_model.c -o regression_model")
        if exit_status != 0:
            print("Compilation failed.")
        else:
            print("Compiled regression_model.c to regression_model executable.")