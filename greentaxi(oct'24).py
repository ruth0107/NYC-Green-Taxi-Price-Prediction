# -*- coding: utf-8 -*-
"""GreenTaxi(Oct'24).ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1zYcI_iS4y4-l9YeMPd8IQPC1SQf57oWO
"""



from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import statsmodels.api as sm
from statsmodels.formula.api import ols
import warnings
warnings.filterwarnings('ignore')

# Load data with the specified path
file_path = r"/content/green_tripdata_2024-10.parquet"
df = pd.read_parquet(file_path)
print("Dataset overview:")
print(f"Number of records: {len(df)}")
print(f"Number of columns: {len(df.columns)}")
print(df.info())

# Data preparation
# Drop unused column and calculate trip duration
df = df.drop("ehail_fee", axis=1)
df['trip_duration'] = (df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']).dt.total_seconds() / 60

# Extract date features
df['weekday'] = df['lpep_dropoff_datetime'].dt.day_name()
df['hourofday'] = df['lpep_dropoff_datetime'].dt.hour
df['dayofmonth'] = df['lpep_dropoff_datetime'].dt.day
df['weekend'] = df['weekday'].isin(['Saturday', 'Sunday']).astype(int)

# Display some basic distributions
print("\nWeekday trip distribution:")
weekday_counts = df['weekday'].value_counts()
print(weekday_counts)

print("\nHour of day distribution:")
hour_counts = df['hourofday'].value_counts().sort_index()
print(hour_counts)

# Missing values handling
print("\nMissing values before imputation:")
print(df.isnull().sum())

# Numeric columns - impute with median
num_cols = ['trip_distance', 'fare_amount', 'extra', 'mta_tax', 'tip_amount',
            'tolls_amount', 'improvement_surcharge', 'congestion_surcharge',
            'trip_duration', 'passenger_count']

# Object columns - impute with mode
obj_cols = ['store_and_fwd_flag', 'RatecodeID', 'payment_type', 'trip_type']

# Impute missing values
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

for col in obj_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing values after imputation:")
print(df.isnull().sum())

# Convert payment_type and trip_type to categorical if they aren't already
# Note: Based on the data info, these are actually stored as float64, so convert to int first then category
df['payment_type'] = df['payment_type'].astype(int).astype('category')
df['trip_type'] = df['trip_type'].astype(int).astype('category')
df['RatecodeID'] = df['RatecodeID'].astype(int).astype('category')

# Basic descriptive statistics
print("\nBasic statistics for key metrics:")
print(df[['trip_distance', 'fare_amount', 'tip_amount', 'total_amount', 'trip_duration']].describe())

# Visualizations
plt.figure(figsize=(15, 10))

# Payment Type and Trip Type distribution
plt.subplot(2, 2, 1)
payment_counts = df['payment_type'].value_counts()
plt.pie(payment_counts, labels=payment_counts.index, autopct='%1.1f%%')
plt.title('Payment Type Distribution')

plt.subplot(2, 2, 2)
trip_counts = df['trip_type'].value_counts()
plt.pie(trip_counts, labels=trip_counts.index, autopct='%1.1f%%')
plt.title('Trip Type Distribution')

# Trip distance distribution
plt.subplot(2, 2, 3)
sns.histplot(df['trip_distance'].clip(upper=20), bins=30, kde=True)
plt.title('Trip Distance Distribution (capped at 20 miles)')

# Trip duration distribution
plt.subplot(2, 2, 4)
sns.histplot(df['trip_duration'].clip(upper=60), bins=30, kde=True)
plt.title('Trip Duration Distribution (capped at 60 minutes)')

plt.tight_layout()
plt.savefig('taxi_distribution_plots.png')
plt.close()

# Temporal analysis
# Average fare by hour and weekday
hourly_amounts = df.groupby('hourofday')['total_amount'].mean().reset_index()
weekday_amounts = df.groupby('weekday')['total_amount'].mean().reset_index()

# Create custom weekday order
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekday_amounts['weekday'] = pd.Categorical(weekday_amounts['weekday'], categories=weekday_order, ordered=True)
weekday_amounts = weekday_amounts.sort_values('weekday')

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
sns.barplot(x='hourofday', y='total_amount', data=hourly_amounts)
plt.title('Average Fare by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Average Total Amount ($)')

plt.subplot(1, 2, 2)
sns.barplot(x='weekday', y='total_amount', data=weekday_amounts)
plt.title('Average Fare by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Average Total Amount ($)')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('temporal_analysis.png')
plt.close()

# Tip analysis
# Create tip percentage feature
df['tip_percentage'] = (df['tip_amount'] / df['fare_amount']) * 100
df.loc[df['tip_percentage'].isin([np.inf, -np.inf]), 'tip_percentage'] = 0
df['tip_percentage'] = df['tip_percentage'].fillna(0).clip(upper=100)  # Cap at 100%

print("\nAverage tip percentage by payment type:")
print(df.groupby('payment_type')['tip_percentage'].mean().sort_values(ascending=False))

print("\nAverage tip percentage by weekday:")
print(df.groupby('weekday')['tip_percentage'].mean().sort_values(ascending=False))

print("\nAverage tip percentage by hour of day:")
hour_tips = df.groupby('hourofday')['tip_percentage'].mean().reset_index()
print(hour_tips.sort_values('tip_percentage', ascending=False).head(5))

# Visualize tip patterns
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
sns.barplot(x='payment_type', y='tip_percentage', data=df)
plt.title('Average Tip Percentage by Payment Type')
plt.xlabel('Payment Type')
plt.ylabel('Average Tip (%)')

plt.subplot(1, 2, 2)
sns.boxplot(x='weekday', y='tip_percentage', data=df[df['tip_percentage'] > 0])
plt.title('Tip Percentage Distribution by Weekday')
plt.xlabel('Day of Week')
plt.ylabel('Tip Percentage (%)')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('tip_analysis.png')
plt.close()

# Statistical tests
# ANOVA test for total_amount by trip_type
print("\nANOVA test for total_amount by trip_type:")
model = ols('total_amount ~ C(trip_type)', data=df).fit()
anova_result = sm.stats.anova_lm(model, typ=2)
print(anova_result)

# ANOVA test for total_amount by weekday
print("\nANOVA test for total_amount by weekday:")
model = ols('total_amount ~ C(weekday)', data=df).fit()
anova_result = sm.stats.anova_lm(model, typ=2)
print(anova_result)

# Chi-square test for association between trip_type and payment_type
contingency_table = pd.crosstab(df['trip_type'], df['payment_type'])
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print(f"\nChi-square test for association between trip_type and payment_type:")
print(f"Chi2: {chi2:.4f}, p-value: {p:.8f}")

# Correlation analysis
# Select numerical columns for correlation
correlation_cols = ['trip_distance', 'trip_duration', 'fare_amount', 'tip_amount',
                    'total_amount', 'passenger_count', 'weekend']

corr_matrix = df[correlation_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# Prepare data for modeling
# Create dummy variables
df_encoded = pd.get_dummies(df, columns=['store_and_fwd_flag', 'RatecodeID',
                                         'payment_type', 'trip_type',
                                         'weekday'], drop_first=True)

# Select features and target
X = df_encoded.drop(['lpep_pickup_datetime', 'lpep_dropoff_datetime',
                    'PULocationID', 'DOLocationID', 'total_amount',
                    'tip_percentage'], axis=1, errors='ignore')
y = df_encoded['total_amount']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Build and evaluate models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append({
        'Model': name,
        'RMSE': rmse,
        'MAE': mae,
        'R² Score': r2
    })

    print(f"{name}:")
    print(f"  RMSE: ${rmse:.2f}")
    print(f"  MAE: ${mae:.2f}")
    print(f"  R² Score: {r2:.4f}")

# Feature importance for best model (Random Forest)
best_model = models['Random Forest']
features = X.columns
importances = best_model.feature_importances_
indices = np.argsort(importances)[-10:]
# Get top 10 features

plt.figure(figsize=(12, 8))
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.title('Top 10 Important Features for Predicting Fare Amount')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Additional analysis - trip distance vs. fare amount
plt.figure(figsize=(10, 6))
sns.scatterplot(x='trip_distance', y='fare_amount', data=df.sample(1000), alpha=0.6, s=50)
plt.title('Trip Distance vs. Fare Amount')
plt.xlabel('Trip Distance (miles)')
plt.ylabel('Fare Amount ($)')
plt.grid(True, alpha=0.3)
plt.savefig('distance_vs_fare.png')
plt.close()

# Create a summary DataFrame of model results
results_df = pd.DataFrame(results)
print("\nModel Comparison:")
print(results_df.to_string(index=False))

# Create a new visualization showing VendorID comparison
plt.figure(figsize=(15, 6))

# Compare trip distance by vendor
plt.subplot(1, 3, 1)
sns.boxplot(x='VendorID', y='trip_distance', data=df)
plt.title('Trip Distance by Vendor')
plt.xlabel('Vendor ID')
plt.ylabel('Trip Distance (miles)')

# Compare fare by vendor
plt.subplot(1, 3, 2)
sns.boxplot(x='VendorID', y='fare_amount', data=df)
plt.title('Fare Amount by Vendor')
plt.xlabel('Vendor ID')
plt.ylabel('Fare Amount ($)')

# Compare tip by vendor
plt.subplot(1, 3, 3)
sns.boxplot(x='VendorID', y='tip_percentage', data=df[df['tip_percentage'] > 0])
plt.title('Tip Percentage by Vendor')
plt.xlabel('Vendor ID')
plt.ylabel('Tip Percentage (%)')

plt.tight_layout()
plt.savefig('vendor_comparison.png')
plt.close()

# Add map visualization - location-based analysis (mock visualization as we don't have actual location data)
# This would typically use actual location data, but we're creating a mock visualization
plt.figure(figsize=(10, 8))
unique_locations = df['PULocationID'].unique()
location_counts = df['PULocationID'].value_counts().sort_index()

plt.bar(range(len(location_counts)), location_counts.values)
plt.title('Trip Count by Pickup Location ID')
plt.xlabel('Pickup Location ID')
plt.ylabel('Number of Trips')
plt.xticks([])  # Hide x-axis ticks as there would be too many
plt.savefig('location_analysis.png')
plt.close()

print("\nAnalysis complete (October 2024 by Ruthvik Akula - Sap id : 70572200028 !")

