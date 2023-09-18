
from flask import Flask, request, jsonify, render_template
import pickle
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the model
with open('xgb_model_with_auc_0_85.pkl', 'rb') as f:
    model = pickle.load(f)

# Expected features
numeric_cols = ['Weight', 'Height', 'BodyMassIndex', 'Hematocrit', 'Leukocytes', 'Platelets', 'TotalBilirubin', 'DirectBilirubin', 'Creatinine', 'Urea', 'ProthrombinTimeActivity', 'InternationalNormalizedRatio', 'Sodium', 'Potassium', 'Albumin', 'AST', 'ALT', 'GGT', 'AlkalinePhosphatase', 'LeftAtriumSize', 'DistalVolumeOfLeftVentricle', 'SystolicVolumeOfLeftVentricle']
categorical_cols = ['Race', 'Sex', 'PreviousVaricealBandLigation', 'PortalHypertensiveGastropathy', 'Ascites', 'SpontaneousBacterialPeritonitis', 'HepatopulmonarySyndrome', 'BetaBlockerUse', 'PortalVeinThrombosis', 'HepaticEncephalopathy', 'HepatorenalSyndrome', 'AntibioticTherapyFor24h', 'HospitalizedFor48h', 'PreTransplantHemodialysis', 'HepatocellularCarcinoma', 'BloodGroup', 'CongestiveHeartFailure', 'Angioplasty', 'Dyslipidemia', 'Hypertension', 'AcuteMyocardialInfarction', 'Stroke', 'DiabetesMellitus', 'ValveReplacement', 'MitralInsufficiency', 'TricuspidInsufficiency', 'NonInvasiveMethod', 'DynamicAlteration']
expected_features = {'Weight': 'float64', 'Height': 'float64', 'BodyMassIndex': 'float64', 'Hematocrit': 'float64', 'Leukocytes': 'float64', 'Platelets': 'float64', 'TotalBilirubin': 'float64', 'DirectBilirubin': 'float64', 'Creatinine': 'float64', 'Urea': 'float64', 'ProthrombinTimeActivity': 'float64', 'InternationalNormalizedRatio': 'float64', 'Sodium': 'float64', 'Potassium': 'float64', 'Albumin': 'float64', 'AST': 'float64', 'ALT': 'float64', 'GGT': 'float64', 'AlkalinePhosphatase': 'float64', 'LeftAtriumSize': 'float64', 'DistalVolumeOfLeftVentricle': 'float64', 'SystolicVolumeOfLeftVentricle': 'float64', 'Race': 'category', 'Sex': 'category', 'PreviousVaricealBandLigation': 'category', 'PortalHypertensiveGastropathy': 'category', 'Ascites': 'category', 'SpontaneousBacterialPeritonitis': 'category', 'HepatopulmonarySyndrome': 'category', 'BetaBlockerUse': 'category', 'PortalVeinThrombosis': 'category', 'HepaticEncephalopathy': 'category', 'HepatorenalSyndrome': 'category', 'AntibioticTherapyFor24h': 'category', 'HospitalizedFor48h': 'category', 'PreTransplantHemodialysis': 'category', 'HepatocellularCarcinoma': 'category', 'BloodGroup': 'category', 'CongestiveHeartFailure': 'category', 'Angioplasty': 'category', 'Dyslipidemia': 'category', 'Hypertension': 'category', 'AcuteMyocardialInfarction': 'category', 'Stroke': 'category', 'DiabetesMellitus': 'category', 'ValveReplacement': 'category', 'MitralInsufficiency': 'category', 'TricuspidInsufficiency': 'category', 'NonInvasiveMethod': 'category', 'DynamicAlteration': 'category'}

# Initialize label encoders for categorical variables
label_encoders = {}

def validate_input(user_input):
    input_df = pd.DataFrame([user_input])
    for feature in expected_features.keys():
        if feature not in input_df.columns:
            input_df[feature] = np.nan
    for feature, dtype in expected_features.items():
        input_df[feature] = input_df[feature].astype(dtype)
    return input_df

def encode_categorical(input_df):
    encoded_df = input_df.copy()
    for feature, dtype in expected_features.items():
        if dtype == 'category':
            le = LabelEncoder()
            encoded_df[feature] = le.fit_transform(input_df[feature].astype(str))
    return encoded_df

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form.to_dict()
        valid_input = validate_input(user_input)
        encoded_input = encode_categorical(valid_input)
        dmatrix_input = xgb.DMatrix(encoded_input, enable_categorical=True)
        prediction_prob = model.predict(dmatrix_input, validate_features=False)
        prediction = 1 if prediction_prob >= 0.2 else 0
        risk_mapping = {0: 'Low-moderate risk for MACE', 1: 'High risk for MACE'}
        return render_template('index.html', prediction=risk_mapping[prediction])
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
