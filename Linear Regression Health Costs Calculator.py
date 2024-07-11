import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
insurance_data = pd.read_csv('insurance.csv')

# Explore the dataset
print(insurance_data.head())

# Data preprocessing
# Convert categorical variables to numerical using one-hot encoding
insurance_data = pd.get_dummies(insurance_data, columns=['sex', 'smoker', 'region'])

# Separate features (X) and target variable (y)
X = insurance_data.drop(['charges'], axis=1)
y = insurance_data['charges']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Function to predict insurance cost based on user input
def predict_insurance_cost(age, bmi, children, smoker_yes, smoker_no, sex_female, sex_male, region_northeast, region_northwest, region_southeast, region_southwest):
    input_data = pd.DataFrame({
        'age': [age],
        'bmi': [bmi],
        'children': [children],
        'smoker_yes': [smoker_yes],
        'smoker_no': [smoker_no],
        'sex_female': [sex_female],
        'sex_male': [sex_male],
        'region_northeast': [region_northeast],
        'region_northwest': [region_northwest],
        'region_southeast': [region_southeast],
        'region_southwest': [region_southwest]
    })
    return model.predict(input_data)[0]

# Example usage
age = 35
bmi = 30
children = 2
smoker_yes = 0
smoker_no = 1
sex_female = 1
sex_male = 0
region_northeast = 1
region_northwest = 0
region_southeast = 0
region_southwest = 0

predicted_cost = predict_insurance_cost(age, bmi, children, smoker_yes, smoker_no, sex_female, sex_male, region_northeast, region_northwest, region_southeast, region_southwest)
print(f'Predicted insurance cost: ${predicted_cost:.2f}')
