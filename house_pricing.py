print("final project based on machine learning and python for predicting the price of the house")
#import all the Required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Step 1: Generate Dataset
data_size = 500
data = {
    'Area_sqft': np.random.randint(500, 5000, data_size),
    'Bedrooms': np.random.randint(1, 6, data_size),
    'Price': np.random.randint(50000, 500000, data_size)
}
df = pd.DataFrame(data)
print(df)

#Check the missing value
print(df.isnull().sum())
# Introduce Missing Values
print("Data containing missing values")
missing_indices = np.random.choice(df.index, size=50, replace=False)
df.loc[missing_indices, 'Bedrooms'] = np.nan
print(df)
print(df.isnull().sum())

# Step 2: Handle Missing Values
print("Data after handling the missing values")
imputer = SimpleImputer(strategy='median')
df[['Bedrooms']] = imputer.fit_transform(df[['Bedrooms']])

print(df)

# Step 3: Feature Scaling
scaler = StandardScaler()
df[['Area_sqft', 'Bedrooms']] = scaler.fit_transform(df[['Area_sqft', 'Bedrooms']])
print(df)

# Step 4: Train-Test Split
X = df[['Area_sqft', 'Bedrooms']]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Step 5: Model Training & Evaluation
models = {
    'Decision Tree': DecisionTreeRegressor(),
    'K-Nearest Neighbors': KNeighborsRegressor()
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'R2 Score': r2_score(y_test, y_pred)
    }

# Step 6: Print Results
for model, metrics in results.items():
    print(f"{model} Performance:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    print("\n")

# Step 7: Data Visualization
plt.figure(figsize=(10, 5))
sns.scatterplot(x=df['Area_sqft'], y=df['Price'], hue=df['Bedrooms'], palette='viridis')
plt.title('House Prices Based on Area and Bedrooms')
plt.xlabel('Area (sq ft)')
plt.ylabel('Price')
plt.show()
# Step 8: Decision Tree Visualization
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)
plt.figure(figsize=(20,12))
plot_tree(dt_model, feature_names=['Area_sqft', 'Bedrooms'], filled=True, rounded=True)
plt.title("Decision Tree Visualization")
plt.show()

# Step 9: Feature Importance
feature_importance = dt_model.feature_importances_
features = ['Area_sqft', 'Bedrooms']
plt.figure(figsize=(8, 5))
sns.barplot(x=feature_importance, y=features)
plt.title("Feature Importance in Decision Tree")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()


print("In this project, we analyzed house price data based on area and number of bedrooms. We used Python and machine learning models like Decision Tree and K-Nearest Neighbors to predict house prices. Our model performance was evaluated using MAE, MSE, and R² score. We improved accuracy by handling missing values, applying feature scaling, and visualizing feature importance.")
