import streamlit as st
import pandas as pd
import joblib
import base64
import os
import plotly.express as px

st.markdown("""
<style>
/* Style the entire app background */
            .stApp {
    background-image: url('https://www.transparenttextures.com/patterns/subtle-white-feathers.png'); /* Subtle Google-like texture */
    background-size: cover; /* Cover the entire app */
    background-position: center; /* Center the image */
    background-repeat: no-repeat; /* Prevent tiling */
    background-attachment: fixed; /* Fixed background for smooth scrolling */
    font-family: 'Roboto', sans-serif; /* Google's preferred font */
}

/* Style the headers */
h1 {
    color: #202124; /* Google's dark text color */
    text-align: center;
    font-weight: 500; /* Medium weight for a modern look */
    margin-bottom: 16px;
}

h2, h3 {
    color: #5F6368; /* Google's secondary text color */
    font-weight: 400;
}

/* Style the buttons */
.stButton > button {
    background-color: #1A73E8; /* Google's primary blue */
    color: #FFFFFF;
    border: none;
    border-radius: 50px; /* As per your request */
    padding: 12px 24px; /* Slightly larger padding for balance */
    text-align: center;
    font-size: 14px; /* Google's standard button text size */
    font-weight: 500;
    margin: 8px 4px;
    cursor: pointer;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2); /* Subtle shadow for depth */
    transition: background-color 0.2s, box-shadow 0.2s; /* Smooth hover effects */
}

/* Hover effect for buttons */
.stButton > button:hover {
    background-color: #1357B9; /* Darker blue on hover */
    box-shadow: 0 3px 6px rgba(0, 0, 0, 0.3); /* Elevated shadow on hover */
}

/* Focus effect for accessibility */
.stButton > button:focus {
    outline: none;
    box-shadow: 0 0 0 3px rgba(26, 115, 232, 0.3); /* Google's focus ring */
}

/* Style input fields to match Google design */
.stTextInput > div > div > input {
    border: 1px solid #DADCE0; /* Google's input border */
    border-radius: 4px;
    padding: 10px;
    font-size: 14px;
    color: #202124;
}

.stTextInput > div > div > input:focus {
    border-color: #1A73E8; /* Blue border on focus */
    box-shadow: 0 0 0 2px rgba(26, 115, 232, 0.2);
    outline: none;
}

/* Style containers for a card-like effect */
.stContainer {
    background-color: #FFFFFF; /* White background for containers to stand out */
    border-radius: 8px;
    padding: 16px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); /* Subtle card shadow */
}
.stApp {
    background-color: #FFFFFF; /* Google's clean white background */
    font-family: 'Roboto', sans-serif; /* Google's preferred font */
}

/* Style the headers */
h1 {
    color: #202124; /* Google's dark text color */
    text-align: center;
    font-weight: 500; /* Medium weight for a modern look */
    margin-bottom: 16px;
}

h2, h3 {
    color: #5F6368; /* Google's secondary text color */
    font-weight: 400;
}

/* Style the buttons */
.stButton > button {
    background-color: #1A73E8; /* Google's primary blue */
    color: #FFFFFF;
    border: none;
    border-radius: 50px; /* As per your request */
    padding: 12px 24px; /* Slightly larger padding for balance */
    text-align: center;
    font-size: 14px; /* Google's standard button text size */
    font-weight: 500;
    margin: 8px 4px;
    cursor: pointer;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2); /* Subtle shadow for depth */
    transition: background-color 0.2s, box-shadow 0.2s; /* Smooth hover effects */
}

/* Hover effect for buttons */
.stButton > button:hover {
    background-color: #1357B9; /* Darker blue on hover */
    box-shadow: 0 3px 6px rgba(0, 0, 0, 0.3); /* Elevated shadow on hover */
}

/* Focus effect for accessibility */
.stButton > button:focus {
    outline: none;
    box-shadow: 0 0 0 3px rgba(26, 115, 232, 0.3); /* Google's focus ring */
}

/* Style input fields to match Google design */
.stTextInput > div > div > input {
    border: 1px solid #DADCE0; /* Google's input border */
    border-radius: 4px;
    padding: 10px;
    font-size: 14px;
    color: #202124;
}

.stTextInput > div > div > input:focus {
    border-color: #1A73E8; /* Blue border on focus */
    box-shadow: 0 0 0 2px rgba(26, 115, 232, 0.2);
    outline: none;
}

/* Style containers for a card-like effect */
.stContainer {
    background-color: #FFFFFF;
    border-radius: 8px;
    padding: 16px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); /* Subtle card shadow */
}
</style>
""", unsafe_allow_html=True)

def get_binary_file_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions CSV</a>'
    return href

st.title("Heart Disease Predictor")
tab1, tab2, tab3 = st.tabs(['Predict', 'Bulk Predict', 'Model Information'])

with tab1:
    age = st.number_input("Age (years)", min_value=0, max_value=150)
    sex = st.selectbox("Sex", ["Male", "Female"])
    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"]
    )
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300)
    cholesterol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0)
    fasting_bs = st.selectbox("Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"])
    resting_ecg = st.selectbox(
        "Resting ECG Results",
        ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
    )
    max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=202)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0)
    st_slope = st.selectbox(
        "Slope of Peak Exercise ST Segment",
        ["Upsloping", "Flat", "Downsloping"]
    )

    # Convert categorical inputs to numeric
    sex_num = 0 if sex == "Male" else 1
    chest_pain_num = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(chest_pain)
    fasting_bs_num = 1 if fasting_bs == "> 120 mg/dl" else 0
    resting_ecg_num = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg)
    exercise_angina_num = 1 if exercise_angina == "Yes" else 0
    st_slope_num = ["Upsloping", "Flat", "Downsloping"].index(st_slope)

    # Create a DataFrame with user inputs
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex_num],
        'ChestPainType': [chest_pain_num],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs_num],
        'RestingECG': [resting_ecg_num],
        'MaxHR': [max_hr],
        'ExerciseAngina': [exercise_angina_num],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope_num]
    })

    # Model information
    algonames = ['Decision Trees', 'Logistic Regression', 'Random Forest', 'Support Vector Machine']
    modelnames = [
        r"C:\Users\amogh\OneDrive\Desktop\Heartdisease\DTS.pkl",
        r"C:\Users\amogh\OneDrive\Desktop\Heartdisease\LogisticRegression.pkl",
        r"C:\Users\amogh\OneDrive\Desktop\Heartdisease\Random Forest Classifier.pkl",
        r"C:\Users\amogh\OneDrive\Desktop\Heartdisease\SVM.pkl"
    ]

    # Function to predict using joblib
    def predict_heart_disease(data, modelnames):
        predictions = []
        for modelname in modelnames:
            if os.path.exists(modelname):
                model = joblib.load(modelname)
                prediction = model.predict(data)
                predictions.append(prediction[0])
            else:
                st.error(f"Model file not found: {modelname}")
                predictions.append(-1) # Placeholder for error
        return predictions

    # Submit button to make predictions
    if st.button("Submit"):
        st.subheader("Results...")
        st.markdown("-----------")
        
        result = predict_heart_disease(input_data, modelnames)
        
        for i in range(len(algonames)):
            st.subheader(algonames[i])
            if result[i] == 0:
                st.write("No heart disease detected.")
            elif result[i] == 1:
                st.write("Heart disease detected.")
            else:
                st.write("Prediction failed due to missing model file.")
        
        st.markdown("-----------")

with tab2:
    st.title("Upload CSV File")
    st.subheader('Instructions to note before uploading the file:')
    st.info("""
    1. Check for missing values in your CSV file. The application will fill them with the mean for numeric columns and mode for categorical columns.
    2. Total 11 features in this order ('Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
    'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope').
    3. Check the spellings of the feature names.
    4. Feature values conventions (for text-based CSVs): 
        Sex: "Male" or "Female"
        ChestPainType: "Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic" (or 'TA', 'ATA', 'NAP', 'ASY')
        FastingBS: "<= 120 mg/dl" or "> 120 mg/dl" (or 0, 1)
        RestingECG: "Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy" (or 'Normal', 'ST', 'LVH')
        ExerciseAngina: "Yes" or "No" 
        ST_Slope: "Upsloping", "Flat", "Downsloping" (or 'Up', 'Flat', 'Down')
    """)
    
    # Define a new uploader for the bulk prediction tab
    uploaded_file_bulk = st.file_uploader("Upload your CSV file for Bulk Prediction", type=["csv"], key='bulk_uploader')

    # Expected columns
    expected_columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 
                        'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']

    # Function to validate categorical values
    def validate_categorical(data, column, valid_values, column_name):
        invalid_values = data[column][~data[column].isin(valid_values)].unique()
        if len(invalid_values) > 0:
            st.error(f"Invalid values found in {column_name}: {invalid_values}. Please correct the CSV file.")
            return False
        return True

    if uploaded_file_bulk is not None:
        try:
            input_data_bulk = pd.read_csv(uploaded_file_bulk)
            st.subheader("Raw CSV Data Preview:")
            st.write(input_data_bulk.head())

            # Impute missing values
            numeric_columns = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
            input_data_bulk[numeric_columns] = input_data_bulk[numeric_columns].fillna(input_data_bulk[numeric_columns].mean())
            categorical_columns = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
            for col in categorical_columns:
                input_data_bulk[col] = input_data_bulk[col].fillna(input_data_bulk[col].mode()[0])

            # Map abbreviated values to expected text values
            chest_pain_abbrev_map = {
                'TA': 'Typical Angina',
                'ATA': 'Atypical Angina',
                'NAP': 'Non-Anginal Pain',
                'ASY': 'Asymptomatic'
            }
            input_data_bulk['ChestPainType'] = input_data_bulk['ChestPainType'].replace(chest_pain_abbrev_map)

            fasting_bs_abbrev_map = {
                0: '<= 120 mg/dl',
                1: '> 120 mg/dl',
                '0': '<= 120 mg/dl',  # Handle string versions in case CSV has mixed types
                '1': '> 120 mg/dl'
            }
            input_data_bulk['FastingBS'] = input_data_bulk['FastingBS'].replace(fasting_bs_abbrev_map)

            resting_ecg_abbrev_map = {
                'ST': 'ST-T Wave Abnormality',
                'LVH': 'Left Ventricular Hypertrophy',
                'Normal': 'Normal'  # Ensure 'Normal' is preserved
            }
            input_data_bulk['RestingECG'] = input_data_bulk['RestingECG'].replace(resting_ecg_abbrev_map)

            st_slope_abbrev_map = {
                'Up': 'Upsloping',
                'Down': 'Downsloping',
                'Flat': 'Flat'
            }
            input_data_bulk['ST_Slope'] = input_data_bulk['ST_Slope'].replace(st_slope_abbrev_map)

            # Validate categorical columns
            valid_values = {
                'Sex': ['Male', 'Female'],
                'ChestPainType': ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'],
                'FastingBS': ['<= 120 mg/dl', '> 120 mg/dl'],
                'RestingECG': ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'],
                'ExerciseAngina': ['Yes', 'No'],
                'ST_Slope': ['Upsloping', 'Flat', 'Downsloping']
            }
            all_valid = True
            for col, values in valid_values.items():
                if not validate_categorical(input_data_bulk, col, values, col):
                    all_valid = False

            if all_valid:
                # Convert categorical string values to numerical values
                input_data_bulk['Sex'] = input_data_bulk['Sex'].apply(lambda x: 0 if x == 'Male' else 1)
                chest_pain_map = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-Anginal Pain': 2, 'Asymptomatic': 3}
                input_data_bulk['ChestPainType'] = input_data_bulk['ChestPainType'].map(chest_pain_map)
                input_data_bulk['FastingBS'] = input_data_bulk['FastingBS'].apply(lambda x: 1 if x == '> 120 mg/dl' else 0)
                resting_ecg_map = {'Normal': 0, 'ST-T Wave Abnormality': 1, 'Left Ventricular Hypertrophy': 2}
                input_data_bulk['RestingECG'] = input_data_bulk['RestingECG'].map(resting_ecg_map)
                input_data_bulk['ExerciseAngina'] = input_data_bulk['ExerciseAngina'].apply(lambda x: 1 if x == 'Yes' else 0)
                st_slope_map = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
                input_data_bulk['ST_Slope'] = input_data_bulk['ST_Slope'].map(st_slope_map)

                # Check for any remaining NaNs
                if input_data_bulk[expected_columns].isna().any().any():
                    st.error("NaN values still present after preprocessing. Please check your CSV file.")
                    st.write(input_data_bulk[expected_columns].isna().sum())
                    st.stop()

                # Check if all required columns exist
                if set(expected_columns).issubset(input_data_bulk.columns):
                    model = joblib.load(r"C:\Users\amogh\OneDrive\Desktop\Heartdisease\LogisticRegression.pkl")
                    predictions = model.predict(input_data_bulk[expected_columns])
                    input_data_bulk['Prediction LR'] = predictions
                    st.subheader("Predictions:")
                    st.write(input_data_bulk)
                    csv = input_data_bulk.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=csv,
                        file_name='PredictedHeartLR.csv',
                        mime='text/csv'
                    )
                else:
                    st.warning("Please make sure the uploaded CSV file has the correct columns:")
                    st.info(", ".join(expected_columns))
            else:
                st.stop()

        except Exception as e:
            st.error(f"An error occurred: {e}. Please check your CSV file and ensure it contains only valid data.")
    else:
        st.info("Upload a CSV file to get predictions.")

with tab3:
    st.header("Model Information")
    st.write("""
    This application uses several machine learning models to predict the presence of heart disease.
    The models are trained on a dataset of patient health metrics.
    
    ### Models used:
    - **Decision Trees (DTS.pkl)**
    - **Logistic Regression (LogisticRegression.pkl)**
    - **Random Forest (Random Forest Classifier.pkl)**
    - **Support Vector Machine (SVM.pkl)**
    
    ### Features (Inputs):
    The models use the following 11 features:
    - Age
    - Sex
    - ChestPainType
    - RestingBP (Resting Blood Pressure)
    - Cholesterol (Serum Cholesterol)
    - FastingBS (Fasting Blood Sugar)
    - RestingECG (Resting Electrocardiogram results)
    - MaxHR (Maximum Heart Rate Achieved)
    - ExerciseAngina (Exercise-induced Angina)
    - Oldpeak (ST Depression)
    - ST_Slope (Slope of the peak exercise ST segment)
    
    ### Prediction Output:
    The models output a binary prediction:
    - **0**: No heart disease detected
    - **1**: Heart disease detected
    
    ### Model Accuracies:
    The bar chart below shows the accuracy of each model on the test dataset.
    """)
    
    data = {'Decision Trees': 80.97, 'Logistic Regression': 85.86, 'Random Forest': 84.23, 'Support Vector Machine': 83.5}
    Models = list(data.keys())
    Accuracies = list(data.values())
    df = pd.DataFrame(list(zip(Models, Accuracies)), columns=['Models', 'Accuracies'])
    fig = px.bar(df, y='Accuracies', x='Models')
    st.plotly_chart(fig)


