import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pickle
import io
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import statsmodels.api as sm
from statsmodels.formula.api import ols
import warnings
import datetime
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="NYC Green Taxi Analysis (Oct 2024)",
    layout="wide",
    page_icon="🚖"
)

# Create tabs
tab1, tab2, tab3 = st.tabs(["Data Analysis", "Model Building", "Fare Prediction"])

with tab1:
    st.title('NYC Green Taxi Trip Data Analysis - October 2024')
    st.markdown("**By Ruthvik Akula (SAP ID: 70572200028)**")

    # File uploader
    uploaded_file = st.file_uploader("Upload NYC Green Taxi Trip data (Parquet)", type=["parquet"])

    # Load data
    @st.cache_data
    def load_data(file):
        if file.name.endswith('.parquet'):
            return pd.read_parquet(file)
        else:
            return None

    if uploaded_file is not None:
        df = load_data(uploaded_file)

        if df is not None:
            # Display data info
            st.subheader('1. Data Overview')
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Number of records:** {len(df):,}")
                st.write(f"**Number of columns:** {len(df.columns)}")

            with col2:
                buffer = io.StringIO()
                df.info(buf=buffer)
                info_str = buffer.getvalue()
                st.download_button(
                    label="Download Data Info",
                    data=info_str,
                    file_name="data_info.txt",
                    mime="text/plain"
                )

            # Data Preparation
            st.subheader('2. Data Preparation')

            # a) Drop ehail_fee column
            if 'ehail_fee' in df.columns:
                df = df.drop('ehail_fee', axis=1)
                st.success("Dropped 'ehail_fee' column as it's unused")

            # b) Calculate trip duration
            df['trip_duration'] = (df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']).dt.total_seconds() / 60
            st.write("Trip duration statistics (minutes):")
            st.write(df['trip_duration'].describe().to_frame().T)

            # c) Extract temporal features
            df['weekday'] = df['lpep_dropoff_datetime'].dt.day_name()
            df['hourofday'] = df['lpep_dropoff_datetime'].dt.hour
            df['dayofmonth'] = df['lpep_dropoff_datetime'].dt.day
            df['weekend'] = df['weekday'].isin(['Saturday', 'Sunday']).astype(int)

            st.success("Added temporal features: weekday, hourofday, dayofmonth, weekend flag")

            # Missing Values Handling
            st.subheader('3. Missing Values Handling')

            st.write("Missing values before imputation:")
            missing_before = df.isnull().sum()
            st.write(missing_before[missing_before > 0])

            # Numeric columns imputation
            num_cols = ['trip_distance', 'fare_amount', 'extra', 'mta_tax', 'tip_amount',
                        'tolls_amount', 'improvement_surcharge', 'congestion_surcharge',
                        'trip_duration', 'passenger_count']

            for col in num_cols:
                if col in df.columns:
                    df[col] = df[col].fillna(df[col].median())

            # Categorical columns imputation
            cat_cols = ['store_and_fwd_flag', 'RatecodeID', 'payment_type', 'trip_type']
            for col in cat_cols:
                if col in df.columns:
                    df[col] = df[col].fillna(df[col].mode()[0])

            st.write("Missing values after imputation:")
            missing_after = df.isnull().sum()
            st.write(missing_after[missing_after > 0])

            # Convert to categorical
            df['payment_type'] = df['payment_type'].astype(int).astype('category')
            df['trip_type'] = df['trip_type'].astype(int).astype('category')
            df['RatecodeID'] = df['RatecodeID'].astype(int).astype('category')

            # Data Distribution Visualizations
            st.subheader('4. Data Distributions')

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Payment Type Distribution**")
                fig, ax = plt.subplots(figsize=(8, 6))
                payment_counts = df['payment_type'].value_counts()
                ax.pie(payment_counts, labels=payment_counts.index, autopct='%1.1f%%')
                st.pyplot(fig)

            with col2:
                st.write("**Trip Type Distribution**")
                fig, ax = plt.subplots(figsize=(8, 6))
                trip_counts = df['trip_type'].value_counts()
                ax.pie(trip_counts, labels=trip_counts.index, autopct='%1.1f%%')
                st.pyplot(fig)

            col3, col4 = st.columns(2)

            with col3:
                st.write("**Trip Distance Distribution (capped at 20 miles)**")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.histplot(df['trip_distance'].clip(upper=20), bins=30, kde=True, ax=ax)
                st.pyplot(fig)

            with col4:
                st.write("**Trip Duration Distribution (capped at 60 minutes)**")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.histplot(df['trip_duration'].clip(upper=60), bins=30, kde=True, ax=ax)
                st.pyplot(fig)

            # Temporal Analysis
            st.subheader('5. Temporal Analysis')

            # Average fare by hour and weekday
            hourly_amounts = df.groupby('hourofday')['total_amount'].mean().reset_index()
            weekday_amounts = df.groupby('weekday')['total_amount'].mean().reset_index()

            # Create custom weekday order
            weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekday_amounts['weekday'] = pd.Categorical(weekday_amounts['weekday'], categories=weekday_order, ordered=True)
            weekday_amounts = weekday_amounts.sort_values('weekday')

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            sns.barplot(x='hourofday', y='total_amount', data=hourly_amounts, ax=ax1)
            ax1.set_title('Average Fare by Hour of Day')
            ax1.set_xlabel('Hour of Day')
            ax1.set_ylabel('Average Total Amount ($)')

            sns.barplot(x='weekday', y='total_amount', data=weekday_amounts, ax=ax2)
            ax2.set_title('Average Fare by Day of Week')
            ax2.set_xlabel('Day of Week')
            ax2.set_ylabel('Average Total Amount ($)')
            ax2.tick_params(axis='x', rotation=45)

            st.pyplot(fig)

            # Tip Analysis
            st.subheader('6. Tip Analysis')

            # Create tip percentage feature
            df['tip_percentage'] = (df['tip_amount'] / df['fare_amount']) * 100
            df.loc[df['tip_percentage'].isin([np.inf, -np.inf]), 'tip_percentage'] = 0
            df['tip_percentage'] = df['tip_percentage'].fillna(0).clip(upper=100)

            st.write("**Average Tip Percentage by Payment Type:**")
            st.write(df.groupby('payment_type')['tip_percentage'].mean().sort_values(ascending=False).to_frame())

            st.write("**Average Tip Percentage by Weekday:**")
            st.write(df.groupby('weekday')['tip_percentage'].mean().sort_values(ascending=False).to_frame())

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            sns.barplot(x='payment_type', y='tip_percentage', data=df, ax=ax1)
            ax1.set_title('Average Tip Percentage by Payment Type')
            ax1.set_xlabel('Payment Type')
            ax1.set_ylabel('Average Tip (%)')

            sns.boxplot(x='weekday', y='tip_percentage', data=df[df['tip_percentage'] > 0], ax=ax2)
            ax2.set_title('Tip Percentage Distribution by Weekday')
            ax2.set_xlabel('Day of Week')
            ax2.set_ylabel('Tip Percentage (%)')
            ax2.tick_params(axis='x', rotation=45)

            st.pyplot(fig)

            # Statistical Tests
            st.subheader('7. Statistical Tests')

            # ANOVA for total_amount by trip_type
            st.write("**ANOVA for total_amount by trip_type:**")
            model = ols('total_amount ~ C(trip_type)', data=df).fit()
            anova_result = sm.stats.anova_lm(model, typ=2)
            st.write(anova_result)

            # ANOVA for total_amount by weekday
            st.write("**ANOVA for total_amount by weekday:**")
            model = ols('total_amount ~ C(weekday)', data=df).fit()
            anova_result = sm.stats.anova_lm(model, typ=2)
            st.write(anova_result)

            # Chi-square test
            st.write("**Chi-square test for association between trip_type and payment_type:**")
            contingency_table = pd.crosstab(df['trip_type'], df['payment_type'])
            chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
            st.write(f"Chi2: {chi2:.4f}, p-value: {p:.8f}")

            # Correlation Analysis
            st.subheader('8. Correlation Analysis')

            correlation_cols = ['trip_distance', 'trip_duration', 'fare_amount', 'tip_amount',
                                'total_amount', 'passenger_count', 'weekend']

            corr_matrix = df[correlation_cols].corr()

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', ax=ax)
            ax.set_title('Correlation Matrix')
            st.pyplot(fig)

            # Vendor Comparison
            st.subheader('9. Vendor Comparison')

            if 'VendorID' in df.columns:
                st.write("**Distributions by Vendor ID:**")
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

                sns.boxplot(x='VendorID', y='trip_distance', data=df, ax=ax1)
                ax1.set_title('Trip Distance by Vendor')
                ax1.set_xlabel('Vendor ID')
                ax1.set_ylabel('Trip Distance (miles)')

                sns.boxplot(x='VendorID', y='fare_amount', data=df, ax=ax2)
                ax2.set_title('Fare Amount by Vendor')
                ax2.set_xlabel('Vendor ID')
                ax2.set_ylabel('Fare Amount ($)')

                sns.boxplot(x='VendorID', y='tip_percentage', data=df[df['tip_percentage'] > 0], ax=ax3)
                ax3.set_title('Tip Percentage by Vendor')
                ax3.set_xlabel('Vendor ID')
                ax3.set_ylabel('Tip Percentage (%)')

                st.pyplot(fig)
            else:
                st.warning("VendorID column not found in dataset")

            # Store the processed dataframe in session state
            st.session_state.processed_df = df
            st.success("Data processing complete! Proceed to Model Building tab.")

        else:
            st.error("Error loading data file!")
    else:
        st.info("Please upload a Parquet file to begin analysis")

with tab2:
    st.title('NYC Green Taxi Model Building')

    if 'processed_df' in st.session_state:
        df = st.session_state.processed_df

        st.subheader('1. Data Preparation for Modeling')

        # Create dummy variables
        df_encoded = pd.get_dummies(df, columns=['store_and_fwd_flag', 'RatecodeID',
                                                 'payment_type', 'trip_type',
                                                 'weekday'], drop_first=True)

        # Select features and target
        X = df_encoded.drop(['lpep_pickup_datetime', 'lpep_dropoff_datetime',
                            'PULocationID', 'DOLocationID', 'total_amount',
                            'tip_percentage'], axis=1, errors='ignore')
        y = df_encoded['total_amount']

        # Save feature names
        feature_names = list(X.columns)
        st.session_state.feature_names = feature_names
        st.success("Features extracted and stored in session state")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        st.write(f"**Training set:** {X_train.shape[0]:,} records")
        st.write(f"**Test set:** {X_test.shape[0]:,} records")
        st.write(f"**Number of features:** {X_train.shape[1]}")

        # Model Building
        st.subheader('2. Model Training')

        models = {
            'Linear Regression': LinearRegression(),
            'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        }

        results = []

        for name, model in models.items():
            with st.spinner(f"Training {name}..."):
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

                # Save the trained model
                pickle.dump(model, open(f'{name.lower().replace(" ", "_")}_model.pkl', 'wb'))

        # Display results
        results_df = pd.DataFrame(results)
        st.subheader("Model Performance Comparison")
        st.dataframe(results_df.style.format({
            'RMSE': '{:.2f}',
            'MAE': '{:.2f}',
            'R² Score': '{:.4f}'
        }))

        # Visual comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        results_df.set_index('Model').plot(kind='bar', ax=ax)
        ax.set_title('Model Performance Comparison')
        ax.set_ylabel('Score')
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)

        # Feature Importance
        st.subheader('3. Feature Importance')

        # Get best model
        best_model_name = results_df.loc[results_df['R² Score'].idxmax(), 'Model']
        best_model = models[best_model_name]

        st.write(f"**Best Model:** {best_model_name} (R²: {results_df.loc[results_df['R² Score'].idxmax(), 'R² Score']:.4f})")

        if hasattr(best_model, 'feature_importances_'):
            # For tree-based models
            features = X.columns
            importances = best_model.feature_importances_

            feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
            feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

            st.write("**Feature Importances:**")
            st.dataframe(feature_importance_df)

            # Plot Feature Importances
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=ax)
            ax.set_title('Feature Importance')
            ax.set_xlabel('Importance')
            ax.set_ylabel('Feature')
            st.pyplot(fig)
        else:
            st.warning("Feature importances not available for this model.")
    else:
        st.info("Please process data and train models in the Data Analysis and Model Building tabs first.")

with tab3:
    st.title('NYC Green Taxi Fare Prediction')

    # Model Selection
    model_option = st.selectbox('Select a Model',
                                ('Linear Regression', 'Decision Tree', 'Random Forest',
                                 'Gradient Boosting'))

    # Input Form
    st.subheader('Enter Trip Details')

    col1, col2 = st.columns(2)

    with col1:
        pickup_datetime = st.date_input("Pickup Date", datetime.date(2024, 10, 15))  # Default date
        pickup_time = st.time_input("Pickup Time", datetime.time(10, 0))  # Default time
        passenger_count = st.number_input('Passenger Count', min_value=1, max_value=6, value=1)
        trip_distance = st.number_input('Trip Distance (miles)', min_value=0.1, max_value=50.0, value=1.0)
        store_and_fwd_flag = st.selectbox('Store and Fwd Flag', ('Y', 'N'))

    with col2:
        rate_code_id = st.selectbox('Rate Code ID', (1, 2, 3, 4, 5, 6))
        payment_type = st.selectbox('Payment Type', (1, 2, 3, 4))
        fare_amount = st.number_input('Fare Amount', min_value=2.5, value=10.0)
        extra = st.number_input('Extra Charges', min_value=0.0, value=0.5)
        mta_tax = st.number_input('MTA Tax', value=0.5)
        tip_amount = st.number_input('Tip Amount', min_value=0.0, value=2.0)
        tolls_amount = st.number_input('Tolls Amount', min_value=0.0, value=0.0)
        improvement_surcharge = st.number_input('Improvement Surcharge', value=0.3)
        congestion_surcharge = st.number_input('Congestion Surcharge', min_value=0.0, value=0.0)
        trip_type = st.selectbox('Trip Type', (1, 2))

    # Prepare input data
    input_data = {
        'passenger_count': passenger_count,
        'trip_distance': trip_distance,
        'store_and_fwd_flag': store_and_fwd_flag,
        'RatecodeID': rate_code_id,
        'payment_type': payment_type,
        'fare_amount': fare_amount,
        'extra': extra,
        'mta_tax': mta_tax,
        'tip_amount': tip_amount,
        'tolls_amount': tolls_amount,
        'improvement_surcharge': improvement_surcharge,
        'congestion_surcharge': congestion_surcharge,
        'trip_type': trip_type,
    }

    # Add temporal features (weekday, hourofday)
    pickup_datetime_combined = datetime.datetime.combine(pickup_datetime, pickup_time)
    input_data['weekday'] = pickup_datetime_combined.strftime('%A')  # Full weekday name
    input_data['hourofday'] = pickup_datetime_combined.hour

    # Feature names (CRITICAL: Ensure consistency with training)
    if 'feature_names' in st.session_state:
        feature_names = st.session_state.feature_names
    else:
        st.error("Feature names not found. Train models first!")
        feature_names = [] # Prevent errors if features not available

    if st.button('Predict Total Amount'):
        try:
            # Load the selected model
            model = pickle.load(open(f'{model_option.lower().replace(" ", "_")}_model.pkl', 'rb'))

            # Prepare input data (same preprocessing as during training)
            input_df = pd.DataFrame(input_data, index=[0])

            # One-hot encode categorical variables
            input_encoded = pd.get_dummies(input_df, columns=['store_and_fwd_flag', 'RatecodeID',
                                                                 'payment_type', 'trip_type',
                                                                 'weekday'], drop_first=True)

            # Ensure all training features are present
            missing_cols = set(feature_names) - set(input_encoded.columns)
            for col in missing_cols:
                input_encoded[col] = 0

            # Reorder columns to match training data
            input_encoded = input_encoded[feature_names]

            # Make prediction
            prediction = model.predict(input_encoded)

            # Display results
            st.subheader('Prediction Result')
            st.success(f"**Predicted Total Amount:** ${prediction[0]:.2f}")

            # Show fare breakdown
            st.subheader('Fare Breakdown')

            breakdown_data = {
                'Component': [
                    'Base Fare', 'Extra Charges', 'MTA Tax', 'Tip Amount',
                    'Tolls', 'Improvement Surcharge', 'Congestion Surcharge',
                    'Other Charges'
                ],
                'Amount ($)': [
                    fare_amount, extra, mta_tax, tip_amount,
                    tolls_amount, improvement_surcharge, congestion_surcharge,
                    max(0, prediction[0] - (fare_amount + extra + mta_tax + tip_amount +
                                               tolls_amount + improvement_surcharge + congestion_surcharge))
                ]
            }

            st.dataframe(pd.DataFrame(breakdown_data))

            # Visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            breakdown_df = pd.DataFrame(breakdown_data)
            breakdown_df = breakdown_df[breakdown_df['Amount ($)'] > 0]

            ax.bar(breakdown_df['Component'], breakdown_df['Amount ($)'])
            ax.set_title('Fare Components')
            ax.set_ylabel('Amount ($)')
            ax.tick_params(axis='x', rotation=45)

            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
    else:
        st.info("Click the button to make a prediction")

