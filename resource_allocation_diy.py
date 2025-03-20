import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Read the dataset
df = pd.read_csv(r'C:\Users\vgvan\Downloads\Data_DIY - Sheet1 (1).csv')
# Drop rows where 'Population' column has a value of 0
df_filtered = df[df['Population'] != 0]


X = df_filtered[["Population"]]  # feature variable (independent variable)


# Input data from the sheet
pop = df_filtered['Population']
b = 0.5  # Transmission rate (beta)
y = 0.2  # Recovery rate (gamma)


s = pop
r = 0
l = 0


l = (pop * b) / 100  # Initial infected population
s = pop - l  # Initial susceptible population
r = 0  # Initial recovered population


s_list = []
l_list = []
r_list = []
time = 45  # Number of days


for t in range(time):
    s_list.append(s)
    l_list.append(l)
    r_list.append(r)


    ds = -b * s * l / pop
    s = s + ds
    dl = b * s * l / pop - y * l
    l = l + dl
    dr = y * l
    r = r + dr


Avg_Susceptible_population = sum(s_list) / len(s_list)
Avg_Infected_population = sum(l_list) / len(l_list)
Avg_Recovered_population = sum(r_list) / len(r_list)


# Convert average infected population to a NumPy array
A_I_pop = np.array(Avg_Infected_population)  # Assuming Avg_Infected_population is a single value


y_Hospitals = df_filtered['Hospitals']  # target variable (dependent variable)
y_Doctors = df_filtered['Doctors']
y_Nurses = df_filtered['Nurses']


# Split your data into training and testing sets (80% for training and 20% for testing)
X_train, X_test, y_train_Hospitals, y_test_Hospitals = train_test_split(X, y_Hospitals, test_size=0.2, random_state=42)
X_train, X_test, y_train_Doctors, y_test_Doctors = train_test_split(X, y_Doctors, test_size=0.2, random_state=42)
X_train, X_test, y_train_Nurses, y_test_Nurses = train_test_split(X, y_Nurses, test_size=0.2, random_state=42)


# Create a Linear Regression model
model_Hospitals = LinearRegression()
model_Doctors = LinearRegression()
model_Nurses = LinearRegression()


# Train the model using the training data
model_Hospitals.fit(X_train, y_train_Hospitals)
model_Doctors.fit(X_train, y_train_Doctors)
model_Nurses.fit(X_train, y_train_Nurses)


# Make predictions on the testing data
y_pred_Hospitals = model_Hospitals.predict(X_test)
y_pred_Doctors = model_Doctors.predict(X_test)
y_pred_Nurses = model_Nurses.predict(X_test)


# Evaluate the model using Mean Squared Error (MSE)
mse_Hospitals = mean_squared_error(y_test_Hospitals, y_pred_Hospitals)
mse_Doctors = mean_squared_error(y_test_Doctors, y_pred_Doctors)
mse_Nurses = mean_squared_error(y_test_Nurses, y_pred_Nurses)


# Define a function to predict the number of hospitals, doctors, and nurses required for a given population
def predict_healthcare(population):
    popul = np.array([[population]])  # Reshape to 2D array with shape (1, 1)
    predicted_Hospitals = model_Hospitals.predict(popul)[0]
    predicted_Doctors = model_Doctors.predict(popul)[0]
    predicted_Nurses = model_Nurses.predict(popul)[0]
    return predicted_Hospitals, predicted_Doctors, predicted_Nurses


# Initialize the dictionary to store the results
resources_required = []


# Store the results in the list
for infected_population in A_I_pop:
    required_Hospitals, required_Doctors, required_Nurses = predict_healthcare(infected_population)
    resources_required.append({
        'Infected Population': infected_population,
        'Hospitals': required_Hospitals,
        'Doctors': required_Doctors,
        'Nurses': required_Nurses
    })


# Convert the list of dictionaries to a DataFrame
df_resources_required = pd.DataFrame(resources_required)


# Save the DataFrame to a CSV file
output_file = r'C:\Users\vgvan\Downloads\Resources_Required.csv'
df_resources_required.to_csv(output_file, index=False)


print(f"Results have been saved to {output_file}")



