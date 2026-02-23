# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries.
2. Load the dataset.
3. Separate independent variable (X) and dependent variable (y).
4. Split the dataset into training and testing sets.
5. Create the Decision Tree Regressor model.
6. Train the model using training data.
7. Predict the salary using test data.
8. Evaluate the model performance.
   
## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Abinaya R
RegisterNumber:  212225320004
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

dataset = pd.read_csv(r'C:\Users\acer\Downloads\Salary.csv')

print("Dataset:\n", dataset.head())

X = dataset[['Level']].values
y = dataset['Salary'].values

model = DecisionTreeRegressor(random_state=0)

model.fit(X, y)

y_pred = model.predict([[6]])
print("\nPredicted Salary for Level 6:", y_pred)

# Visualization
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y)
plt.plot(X_grid, model.predict(X_grid))
plt.title('Decision Tree Regression')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()
```

## Output:
![Decision Tree Regressor Model for Predicting the Salary of the Employee](sam.png)

<img width="878" height="772" alt="Screenshot 2026-02-23 194655" src="https://github.com/user-attachments/assets/2f4d666b-c238-4005-ad76-acf3720a8135" />



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
