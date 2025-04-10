# 🚖 NYC Green Taxi Trip Analysis - October 2024

This Streamlit web application provides a comprehensive analysis of NYC Green Taxi trips for October 2024. It supports data exploration, cleaning, visualization, statistical testing, and lays the groundwork for fare prediction and model building (under construction in future tabs).

## 📌 Features

### ✅ Tab 1: Data Analysis

- Upload `.parquet` file of NYC Green Taxi data
- Automatic data info and summary
- Handles missing values (numeric: median, categorical: mode)
- Extracts temporal features (hour, day, weekday, weekend)
- Visualizes:
  - Payment type and trip type distribution (pie charts)
  - Trip distance and duration (histograms)
  - Average fare by hour of day and weekday (bar charts)
  - Tip percentage stats by payment type and weekday
- Statistical Tests:
  - ANOVA for fare differences by trip type and weekday
  - Chi-square test for relationship between trip type and payment type
- Correlation matrix of numerical features
- VendorID-based fare, tip, and distance comparison (boxplots)

### 🛠️ Tab 2: Model Building (Coming Soon)
- Will allow selection and training of ML models (Linear Regression, Random Forest, Gradient Boosting, etc.)
- Evaluation using RMSE, MAE, and R²

### 🔮 Tab 3: Fare Prediction (Coming Soon)
- User inputs ride features to predict fare

## 📂 File Structure

