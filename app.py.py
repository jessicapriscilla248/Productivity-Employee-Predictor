# streamlit_app.py

import streamlit as st
import pandas as pd
import pickle
import numpy as np
from catboost import CatBoostClassifier

# --- 1. Set Streamlit Page Configuration ---
st.set_page_config(layout="wide", page_title="Garment Worker Productivity Predictor")

# --- 2. Load Model and Preprocessing Assets ---

@st.cache_resource
def load_model():
    """Loads the trained CatBoost model."""
    try:
        with open('./model/trained_model.pkl', 'rb') as file:
            model = pickle.load(file)
        st.success("Model loaded successfully!")
        return model
    except FileNotFoundError:
        st.error("Error: 'trained_model.pkl' not found. Please ensure the trained model file is in the correct path ('./model/').")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_model()

# Median 'wip' from the training data for imputation
try:
    _temp_df_for_median = pd.read_csv('./dataset/garments_worker_productivity.csv')
    WIP_MEDIAN = _temp_df_for_median['wip'].median()
    del _temp_df_for_median 
except FileNotFoundError:
    st.error("Error: 'garments_worker_productivity.csv' not found. Cannot determine WIP median for preprocessing. Please ensure the dataset file is in the correct path ('./dataset/').")
    st.stop()
except Exception as e:
    st.error(f"Error determining WIP median: {e}")
    st.stop()

# SMV category bins and labels
SMV_BINS = [0, 20, 30, 50]
SMV_LABELS = ['low', 'medium', 'high']

# List of categorical features 
CATEGORICAL_COLS_FOR_STR_CONVERSION = ['day', 'department', 'quarter', 'smv_category']

# Column order for prediction - MUST match the order of X_train features
FEATURE_COLUMNS_ORDER = [
    'quarter', 'department', 'day', 'team', 'targeted_productivity', 'smv',
    'wip', 'over_time', 'incentive', 'idle_time', 'idle_men',
    'no_of_style_change', 'no_of_workers', 'smv_category'
]


# --- 3. Preprocessing Function ---

def preprocess_input(input_df: pd.DataFrame) -> pd.DataFrame:
    """Applies the same preprocessing steps as used during model training."""

    df_processed = input_df.copy()

    # Fill missing 'wip' with the pre-calculated median
    df_processed['wip'] = df_processed['wip'].fillna(WIP_MEDIAN)

    # Clean 'department' column: lowercase and strip spaces
    if 'department' in df_processed.columns:
        df_processed['department'] = df_processed['department'].astype(str).str.lower().str.strip()
    else:
        st.warning("Department column not found in input data. Please ensure it is present.")

    # Ensure 'smv' column is numeric, convert errors to NaN if any
    if 'smv' in df_processed.columns:
        df_processed['smv'] = pd.to_numeric(df_processed['smv'], errors='coerce')
        df_processed['smv_category'] = pd.cut(
            df_processed['smv'],
            bins=SMV_BINS,
            labels=SMV_LABELS,
            right=True,
            include_lowest=True
        )
    else:
        st.warning("SMV column not found in input data. 'smv_category' will be set to 'nan'.")
        df_processed['smv_category'] = np.nan # Ensure it's NaN before conversion to str "nan"


    # Convert specific categorical columns to string type for CatBoost
    for col in CATEGORICAL_COLS_FOR_STR_CONVERSION:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype(str)
        else:
            pass


    # Drop 'date' column if present
    if 'date' in df_processed.columns:
        df_processed = df_processed.drop('date', axis=1)


    # Ensure all required feature columns are present and in the correct order
    for col in FEATURE_COLUMNS_ORDER:
        if col not in df_processed.columns:
            default_val = "nan" if col in CATEGORICAL_COLS_FOR_STR_CONVERSION else 0.0
            df_processed[col] = default_val
            st.warning(f"Missing column '{col}' detected in input data. Filled with default value '{default_val}'.")


    final_features_df = df_processed[FEATURE_COLUMNS_ORDER]

    return final_features_df

# --- 4. Streamlit UI Sections ---

# Initialize session state for navigation and prediction results
if 'page' not in st.session_state:
    st.session_state.page = 'Application Introduction'
if 'single_prediction_result' not in st.session_state:
    st.session_state.single_prediction_result = None
if 'batch_prediction_df' not in st.session_state:
    st.session_state.batch_prediction_df = None


# --- Sidebar Navigation ---
with st.sidebar:
    st.header("Productivity App")

    if st.button("➕ New Prediction"):
        st.session_state.single_prediction_result = None
        st.session_state.batch_prediction_df = None
        st.session_state.page = 'Input Data'
        st.rerun()

    st.markdown("---") # Separator

    page_selection = st.radio(
        "Navigate Sections:",
        ('Application Introduction', 'Input Data', 'View Results & Analysis'),
        index=['Application Introduction', 'Input Data', 'View Results & Analysis'].index(st.session_state.page), 
        key="main_navigation_radio" 
    )

    if page_selection != st.session_state.page:
        st.session_state.page = page_selection
        st.rerun() 


# --- Main Page ---

if st.session_state.page == 'Application Introduction':
    st.title("👕 Garment Worker Productivity Predictor")
    st.markdown("""
    Welcome to the Garment Worker Productivity Predictor!
    This application uses a trained CatBoost Classification model to predict whether a team's
    `actual_productivity` will be **Productive** or **Not Productive**,
    based on various operational factors.

    Use the sidebar to navigate to the 'Input Data' section to make a prediction, 
    'View Results & Analysis' to see past prediction outputs, or '➕ New Prediction' to make a new prediction.

    ---
    **About the Prediction:**
    The model classifies productivity into two categories:
    -   **Productive (1)**: The team's actual productivity is expected to be at or above the historical median productivity.
    -   **Not Productive (0)**: The team's actual productivity is expected to be below the historical median productivity.

    This can help in identifying teams that might need support or recognizing high-performing teams.
    """)
    st.info("Dataset Source: [Garment Worker Productivity Dataset](https://archive.ics.uci.edu/dataset/597/productivity+prediction+of+garment+employees)")



elif st.session_state.page == 'Input Data':
    st.title("📥 Input Data for Prediction")
    st.markdown("""
    Please provide the necessary data points below to get a productivity prediction.
    You can either enter data manually for a single prediction or upload a CSV file for batch predictions.
    """)
    st.subheader("Dataset Column Information:")
    st.markdown("""
    Here's a breakdown of the columns in the dataset used for this prediction model:

    * **`date`**: Date of the record for each observation (e.g., 'MM/DD/YYYY').
    * **`quarter`**: The quarter of the year (e.g., 'Quarter1', 'Quarter2').
    * **`department`**: The department responsible for production (e.g., 'sweing', 'finishing').
    * **`day`**: The day of the week (e.g., 'Monday', 'Tuesday').
    * **`team`**: Unique identifier for a production team (integer).
    * **`targeted_productivity`**: Targeted productivity set by the Authority for each team for each day.
    * **`smv` (Standard Minute Value)**: Standard Minute Value, it is the allocated time for a task.
    * **`wip` (Work in Progress)**: Work in progress. Includes the number of unfinished items for products.
    * **`over_time`**: Represents the amount of overtime by each team in minutes.
    * **`incentive`**: Represents the amount of financial incentive (in BDT) that enables or motivates a particular course of action.
    * **`idle_time`**: The amount of time (in minutes) when the production line or workers were idle (float).
    * **`idle_men`**: The number of workers who were idle due to production interruption.
    * **`no_of_style_change`**: Number of changes in the style of a particular product (integer: 0, 1, or 2).
    * **`no_of_workers`**: Number of workers in each team.
    * **`actual_productivity`**: The actual productivity achieved by the team (float, 0.0 to 1.0, this is the original target used to derive binary 'target').
    """)

    input_method = st.radio("Choose input method:", ("Enter Data Manually", "Upload CSV File"), key="input_method_radio")

    if input_method == "Enter Data Manually":
        st.subheader("Enter Individual Data Point")

        col1, col2, col3 = st.columns(3)

        with col1:
            date_val = st.text_input("Date (MM/DD/YYYY)", "01/01/2015", key="date_input")
            quarter_val = st.selectbox("Quarter (1-5)", ['Quarter1', 'Quarter2', 'Quarter3', 'Quarter4', 'Quarter5'], key="quarter_input")
            department_val = st.selectbox("Department", ['sweing', 'finishing'], key="department_input") # Include raw as in original data
            day_val = st.selectbox("Day", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], key="day_input")
            team_val = st.number_input("Team", min_value=1, max_value=12, value=8, key="team_input")

        with col2:
            targeted_productivity_val = st.number_input("Targeted Productivity (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=0.8, step=0.01, key="target_prod_input")
            smv_val = st.number_input("Standard Minute Value (SMV)", min_value=0.0, value=26.16, step=0.01, key="smv_input")
            wip_val = st.number_input("Work in Progress (WIP)", value=float(WIP_MEDIAN), key="wip_input") # Default to median for new entry
            over_time_val = st.number_input("Over Time (minutes)", min_value=0, value=7080, key="overtime_input")
            incentive_val = st.number_input("Incentive", min_value=0, value=98, key="incentive_input")

        with col3:
            idle_time_val = st.number_input("Idle Time", min_value=0.0, value=0.0, step=0.1, key="idle_time_input")
            idle_men_val = st.number_input("Idle Men", min_value=0, value=0, key="idle_men_input")
            no_of_style_change_val = st.number_input("No. of Style Change", min_value=0, max_value=2, value=0, key="style_change_input")
            no_of_workers_val = st.number_input("No. of Workers", min_value=1.0, value=59.0, step=0.5, key="workers_input")

        input_data_dict = {
            'date': [date_val],
            'quarter': [quarter_val],
            'department': [department_val],
            'day': [day_val],
            'team': [team_val],
            'targeted_productivity': [targeted_productivity_val],
            'smv': [smv_val],
            'wip': [wip_val],
            'over_time': [over_time_val],
            'incentive': [incentive_val],
            'idle_time': [idle_time_val],
            'idle_men': [idle_men_val],
            'no_of_style_change': [no_of_style_change_val],
            'no_of_workers': [no_of_workers_val]
        }
        input_df = pd.DataFrame(input_data_dict)

        if st.button("Predict Productivity", key="predict_button_manual"):
            try:
                processed_input_df = preprocess_input(input_df)
                prediction = model.predict(processed_input_df)[0]
                prediction_label = "Productive (1)" if prediction == 1 else "Not Productive (0)"

                st.session_state.single_prediction_result = {
                    'input_data': input_df,
                    'processed_data': processed_input_df,
                    'prediction_label': prediction_label
                }
                st.session_state.page = 'View Results & Analysis' 
                st.rerun()

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.write("Please check your input values and ensure the model and necessary files are correctly loaded.")


    else:
        st.subheader("Upload CSV File for Batch Prediction")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="csv_uploader")

        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.write("Original uploaded data (first 5 rows):")
                st.dataframe(batch_df.head())

                if st.button("Predict Productivity for Uploaded CSV", key="predict_button_csv"):
                    final_results_df = batch_df.copy()

                    processed_batch_df = preprocess_input(batch_df.copy())
                    
                    batch_predictions_encoded = model.predict(processed_batch_df)
                    
                    batch_predictions_label = np.array(["Productive (1)" if p == 1 else "Not Productive (0)" for p in batch_predictions_encoded])

                    final_results_df['predicted_productivity_label'] = batch_predictions_label
                    final_results_df['smv_category'] = processed_batch_df['smv_category']

                    st.session_state.batch_prediction_df = final_results_df
                    st.session_state.batch_view_index = 0
                    
                    st.session_state.page = 'View Results & Analysis'
                    st.rerun()

            except Exception as e:
                st.error(f"An error occurred while processing the CSV file: {e}")
                st.write("Please ensure the CSV file has the correct columns and format.")

elif st.session_state.page == 'View Results & Analysis':
    st.title("📊 View Results & Analysis")
    st.markdown("Here you can see the results of your predictions.")

    if st.session_state.single_prediction_result:
        result = st.session_state.single_prediction_result
        input_df = result['input_data']
        processed_input_df = result['processed_data']
        prediction_label = result['prediction_label']

        team_num = input_df['team'].iloc[0]
        date_val = input_df['date'].iloc[0]
        dept_name = input_df['department'].iloc[0].title() 
        smv_val = input_df['smv'].iloc[0]
        smv_cat_val = processed_input_df['smv_category'].iloc[0] 
        over_time_val = input_df['over_time'].iloc[0]
        idle_time_val = input_df['idle_time'].iloc[0]
        incentive_val = input_df['incentive'].iloc[0]

        st.markdown(f"### Team {team_num} Productivity Analysis:")
        st.write("---")
        st.markdown(f"**Date** : {date_val}")
        st.markdown(f"**Departement** : {dept_name}")
        st.markdown(f"**SMV (Standard Minute Value)** : {smv_val} (Category: {smv_cat_val})")
        st.markdown(f"**Worked Overtime** : {over_time_val} Minutes")
        st.markdown(f"**Idle Time** : {idle_time_val} Minutes")
        if "Productive (1)" in prediction_label:
            st.success(f"Predicted Productivity Level: **{result['prediction_label']}**")
        else:
            st.warning(f"Predicted Productivity Level: **{result['prediction_label']}**")

        st.subheader("📝 Recommendations")
        with st.expander("Click to see actionable recommendations for this team"):
            recommendations = []
            if "Productive (1)" in prediction_label:
                st.success("This team is performing well. Here are some recommendations to maintain their momentum:")
                recommendations.append("The current incentive strategy is effective. Consider maintaining or slightly increasing it to sustain high performance.")
                recommendations.append("This team's workflow and efficiency can serve as a positive benchmark for other teams.")
                if over_time_val > 4000:
                    recommendations.append(f"**Caution:** The overtime ({over_time_val} minutes) is significant. It is advisable to monitor the team's workload to prevent potential burnout.")
            else:
                st.warning("This team is predicted to be underperforming. Here are some potential areas for improvement:")
                recommendations.append("Investigate the production process for this team to identify any potential sources of idle time or bottlenecks.")
                recommendations.append("Review the worker allocation and team composition to ensure it is optimized for the assigned task.")
                if smv_cat_val == 'high':
                    recommendations.append("**Suggestion:** The task's Standard Minute Value (SMV) is high, indicating its complexity. Consider breaking the task down into smaller, more manageable steps.")
                if incentive_val < 50:
                    recommendations.append(f"**Suggestion:** The financial incentive (BDT {incentive_val}) is relatively low. Consider increasing the incentive to significantly boost team motivation and productivity.")
            
            for rec in recommendations:
                st.markdown(f"- {rec}")

        st.write("---")
        st.subheader("Input Data:")
        st.dataframe(result['input_data'])
        st.subheader("Processed Data used for Prediction:")
        st.dataframe(result['processed_data'])
        st.markdown("---")

    if st.session_state.batch_prediction_df is not None:
        st.subheader("Batch Prediction Results:")
        
        batch_df = st.session_state.batch_prediction_df
        
        st.dataframe(batch_df)
        st.markdown("---")

        st.subheader("Detailed Analysis per Row")

        if 'batch_view_index' not in st.session_state:
            st.session_state.batch_view_index = 0
        
        current_index = st.session_state.batch_view_index
        total_rows = len(batch_df)

        # --- Navigation Button ---
        col1, col2, col3 = st.columns([1, 1, 5])
        with col1:
            if st.button("⬅️ Previous"):
                if current_index > 0:
                    st.session_state.batch_view_index -= 1
                    st.rerun()
        with col2:
            if st.button("Next ➡️"):
                if current_index < total_rows - 1:
                    st.session_state.batch_view_index += 1
                    st.rerun()
        with col3:
            st.write(f"Viewing Record: **{current_index + 1} of {total_rows}**")

        current_row = batch_df.iloc[current_index]

        team_num = current_row['team']
        date_val = current_row.get('date', 'N/A')
        dept_name = str(current_row['department']).title()
        smv_val = current_row['smv']
        smv_cat_val = current_row['smv_category']
        over_time_val = current_row['over_time']
        idle_time_val = current_row['idle_time']
        incentive_val = current_row['incentive']
        prediction_label = current_row['predicted_productivity_label']

        st.markdown(f"#### Analysis for Team {team_num}")
        st.markdown(f"**Date**: {date_val}")
        st.markdown(f"**Department**: {dept_name}")
        st.markdown(f"**SMV**: {smv_val} (Category: {smv_cat_val})")
        st.markdown(f"**Overtime**: {over_time_val} mins")
        st.markdown(f"**Incentive**: {incentive_val}")
        if prediction_label == 'Not Productive (0)':
            st.warning(f"**Prediction**: **{prediction_label}**")
        else:
            st.success(f"**Prediction**: **{prediction_label}**")

        with st.expander("Click to see HRD Recommendations for this specific team"):
            recommendations = []
            if "Productive (1)" in prediction_label:
                st.success("This team is performing well. Recommendations:")
                recommendations.append("Incentive strategy is effective. Consider maintaining it.")
                recommendations.append("This team's workflow can be a benchmark for others.")
                if over_time_val > 4000:
                    recommendations.append(f"**Caution:** Overtime ({over_time_val} mins) is high. Monitor workload to prevent burnout.")
            else:
                st.warning("This team is underperforming. Recommendations:")
                recommendations.append("Investigate potential idle time or bottlenecks.")
                recommendations.append("Review team composition for this task.")
                if smv_cat_val == 'high':
                    recommendations.append("**Suggestion:** Task SMV is high. Consider breaking it into smaller steps.")
                if incentive_val < 50:
                    recommendations.append(f"**Suggestion:** Incentive (BDT {incentive_val}) is low. Consider increasing it to boost motivation.")
            
            for rec in recommendations:
                st.markdown(f"- {rec}")

        csv_output = batch_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Full Predictions as CSV",
            data=csv_output,
            file_name="productivity_predictions.csv",
            mime="text/csv",
            key="download_csv_button"
        )
        st.markdown("---")
    
    if st.session_state.single_prediction_result is None and st.session_state.batch_prediction_df is None:
        st.info("No predictions have been made yet. Please go to the 'Input Data' section to make a prediction.")

st.markdown("---")
st.info("Developed with Streamlit and CatBoost. Data source: Garment Worker Productivity Dataset.")