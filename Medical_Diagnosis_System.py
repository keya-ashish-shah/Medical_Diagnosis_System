import numpy as np
import pandas as pd
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import streamlit as st
import pdfplumber
import re
from utils import generate_health_report_pdf
from diet_loader import load_diet_plan
from streamlit_option_menu import option_menu
from bs4 import BeautifulSoup
from streamlit import download_button

# --- Load datasets ---
def load_dataset(dataset_type):
    try:
        if dataset_type == "diabetes":
            return pd.read_csv("diabetes.csv")
        elif dataset_type == "heart":
            return pd.read_csv("heart.csv")
        elif dataset_type == "thyroid":
            file_path = "thyroid_dataset_300_rows.csv"
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip()
            df.rename(columns={
                'TSH (mIU/L)': 'TSH',
                'T3 (ng/dL)': 'T3',
                'T4 (µg/dL)': 'T4'
            }, inplace=True)
            return df
        elif dataset_type == "pcod":
            return pd.read_csv("pcod.csv")
        elif dataset_type == "anxiety":
            return pd.read_csv("anxiety_dataset_300_modified.csv")
    except Exception as e:
        st.error(f"Error loading {dataset_type} dataset: {e}")
        return None


# --- Fuzzy Logic for Heart Disease ---
def create_fuzzy_heart(df):
    age = ctrl.Antecedent(np.arange(df['age'].min(), df['age'].max() + 1, 1), 'age')
    cholesterol = ctrl.Antecedent(np.arange(df['Cholesterol'].min(), df['Cholesterol'].max() + 1, 1), 'cholesterol')
    thalach = ctrl.Antecedent(np.arange(df['thalach'].min(), df['thalach'].max() + 1, 1), 'thalach')
    chest_pain = ctrl.Antecedent(np.arange(0, 4, 1), 'chest_pain')
    resting_bp = ctrl.Antecedent(np.arange(df['trestbps'].min(), df['trestbps'].max() + 1, 1), 'resting_bp')
    heart_risk = ctrl.Consequent(np.arange(0, 101, 1), 'heart_risk')

    age.automf(3)
    cholesterol.automf(3)
    thalach.automf(3)
    chest_pain.automf(3)
    resting_bp.automf(3)
    heart_risk.automf(3)

    rules = [
        ctrl.Rule(age['poor'] & cholesterol['poor'], heart_risk['poor']),
        ctrl.Rule(chest_pain['poor'] & resting_bp['poor'], heart_risk['poor']),
        ctrl.Rule(age['average'] & cholesterol['average'], heart_risk['average']),
        ctrl.Rule(thalach['good'] & cholesterol['good'], heart_risk['good']),
        ctrl.Rule(chest_pain['good'] & thalach['good'] & age['good'], heart_risk['good']),
    ]

    heart_ctrl = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(heart_ctrl)


# --- Fuzzy Logic for Diabetes ---
def create_fuzzy_diabetes(df):
    glucose = ctrl.Antecedent(np.arange(df['Glucose'].min(), df['Glucose'].max() + 1, 1), 'glucose')
    bmi = ctrl.Antecedent(np.arange(df['BMI'].min(), df['BMI'].max() + 1, 1), 'bmi')
    age = ctrl.Antecedent(np.arange(df['Age'].min(), df['Age'].max() + 1, 1), 'age')
    blood_pressure = ctrl.Antecedent(np.arange(df['BloodPressure'].min(), df['BloodPressure'].max() + 1, 1), 'blood_pressure')
    diabetes_risk = ctrl.Consequent(np.arange(0, 101, 1), 'diabetes_risk')

    glucose.automf(3)
    bmi.automf(3)
    age.automf(3)
    blood_pressure.automf(3)
    diabetes_risk.automf(3)

    rules = [
        ctrl.Rule(glucose['poor'] & bmi['poor'] & age['poor'], diabetes_risk['poor']),
        ctrl.Rule(glucose['average'] & bmi['average'] & age['average'], diabetes_risk['average']),
        ctrl.Rule(glucose['good'] & bmi['good'] & age['good'], diabetes_risk['good']),
        ctrl.Rule(blood_pressure['poor'] & glucose['poor'], diabetes_risk['poor']),
        ctrl.Rule(blood_pressure['average'] & bmi['average'] & age['average'], diabetes_risk['average']),
        ctrl.Rule(glucose['average'] & blood_pressure['good'], diabetes_risk['average'])
    ]

    diabetes_ctrl = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(diabetes_ctrl)






# --- Fuzzy Logic for Thyroid ---
def create_fuzzy_thyroid(df):
    tsh = ctrl.Antecedent(np.arange(df['TSH'].min(), df['TSH'].max() + 1, 0.1), 'tsh')
    t3 = ctrl.Antecedent(np.arange(df['T3'].min(), df['T3'].max() + 1, 1), 't3')
    t4 = ctrl.Antecedent(np.arange(df['T4'].min(), df['T4'].max() + 1, 0.1), 't4')
    thyroid_risk = ctrl.Consequent(np.arange(0, 101, 1), 'thyroid_risk')

    tsh.automf(3)
    t3.automf(3)
    t4.automf(3)
    thyroid_risk.automf(3)

    rules = [
        ctrl.Rule(tsh['good'] & t3['good'] & t4['good'], thyroid_risk['poor']),
        ctrl.Rule(tsh['average'] | t3['average'] | t4['average'], thyroid_risk['average']),
        ctrl.Rule(tsh['poor'] | t3['poor'] | t4['poor'], thyroid_risk['good']),
    ]

    system = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(system)


# --- Fuzzy Logic for PCOD ---
def create_fuzzy_pcod(df):
    bmi = ctrl.Antecedent(np.arange(df['BMI'].min(), df['BMI'].max() + 1, 1), 'bmi')
    insulin = ctrl.Antecedent(np.arange(df['Insulin_Level'].min(), df['Insulin_Level'].max() + 1, 1), 'insulin')
    lh = ctrl.Antecedent(np.arange(df['LH'].min(), df['LH'].max() + 1, 1), 'lh')
    pcod_risk = ctrl.Consequent(np.arange(0, 101, 1), 'pcod_risk')

    bmi.automf(3)
    insulin.automf(3)
    lh.automf(3)
    pcod_risk.automf(3)

    rules = [
        ctrl.Rule(bmi['good'] & insulin['good'] & lh['good'], pcod_risk['poor']),
        ctrl.Rule(bmi['average'] | insulin['average'] | lh['average'], pcod_risk['average']),
        ctrl.Rule(bmi['poor'] | insulin['poor'] | lh['poor'], pcod_risk['good'])
    ]

    system = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(system)

# --- Fuzzy Logic for Anxiety ---

def create_fuzzy_anxiety(df):
    sleep = ctrl.Antecedent(np.arange(df['SleepHours'].min(), df['SleepHours'].max()+0.1, 0.1), 'sleep')
    heart_rate = ctrl.Antecedent(np.arange(df['HeartRate'].min(), df['HeartRate'].max()+1, 1), 'heart_rate')
    fatigue = ctrl.Antecedent(np.arange(0, 2, 1), 'fatigue')
    irritability = ctrl.Antecedent(np.arange(0, 2, 1), 'irritability')
    restlessness = ctrl.Antecedent(np.arange(0, 2, 1), 'restlessness')
    score = ctrl.Antecedent(np.arange(df['ScoreGAD7'].min(), df['ScoreGAD7'].max()+1, 1), 'score')
    anxiety_risk = ctrl.Consequent(np.arange(0, 101, 1), 'anxiety_risk')

    sleep.automf(3)
    heart_rate.automf(3)
    fatigue.automf(3)
    irritability.automf(3)
    restlessness.automf(3)
    score.automf(3)
    anxiety_risk.automf(3)

    rules = [
        ctrl.Rule(sleep['poor'] | score['good'] | heart_rate['good'], anxiety_risk['good']),
        ctrl.Rule(score['average'] | irritability['average'] | fatigue['average'], anxiety_risk['average']),
        ctrl.Rule(sleep['good'] & fatigue['poor'] & irritability['poor'], anxiety_risk['poor']),
        ctrl.Rule(restlessness['poor'] | fatigue['good'], anxiety_risk['good']),  # ✅ Added rule with restlessness
    ]


    system = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(system)


# --- Streamlit UI ---
def main():
    st.title("🩺 Medical Diagnosis System")

    with st.sidebar:
        selected = option_menu("Main Menu",["Home", "Heart Disease", "Diabetes", "Thyroid", "PCOD", "Anxiety", "Diet Recommendation","Dataset","Doctors"],
                                icons=['house', 'heart', 'activity', 'diagram-3', 'droplet', 'arrow-down-circle', 'cup-straw', "table", "person"],
                                menu_icon="cast",default_index=0

)

    if selected == "Home":
        st.markdown("""<div style='text-align: justify; font-size: 16px; line-height: 1.6'>
                <p><strong>Take control of your well-being with our Intelligent Medical Diagnosis System</strong> — a powerful tool designed to help you monitor your health and detect early signs of common medical conditions.</p>
                <p>Backed by clinical data and advanced algorithms, our system can assess your risk for:</p>
                <ul>
                    <li>❤️ <strong>Heart Disease</strong></li>
                    <li>🩸 <strong>Diabetes</strong></li>
                    <li>🦋 <strong>Thyroid Disorders</strong></li>
                    <li>⚕️ <strong>Polycystic Ovarian Disease (PCOD)</strong></li>
                </ul>
                <p>Simply enter your health details and get instant, data-driven insights — all in one secure, user-friendly platform.</p>
                <p>✅ Early detection<br>
                ✅ Personalized feedback<br>
               ✅ Faster decisions for a healthier life</p>
            </div>""", unsafe_allow_html=True)




    elif selected == "Doctors":
        st.subheader("Doctors' Dataset")
        try:
            doctor_data = pd.read_csv("indian_doctors_dataset.csv")
            st.dataframe(doctor_data)
        except FileNotFoundError:
            st.warning("Doctor dataset not found. Please upload or check the file.")
    elif selected == "Dataset":
        st.subheader(" View Dataset")
        dataset_choice = st.selectbox("Select which dataset to display", ["Heart", "Diabetes", "Thyroid", "PCOD","Anxiety"])

        df = load_dataset(dataset_choice.lower())
        if df is not None:
            st.dataframe(df)

# Heart disease logic

    elif selected == "Heart Disease":
        df = load_dataset("heart")
        if df is None:
            return

        sim = create_fuzzy_heart(df)
        st.subheader("Heart Disease")

        age = st.number_input("Age", min_value=1, max_value=100, value=int(df['age'].mean()))
        cholesterol = st.number_input("Cholesterol", int(df['Cholesterol'].min()), int(df['Cholesterol'].max()), int(df['Cholesterol'].mean()))
        thalach = st.number_input("Max Heart Rate", min_value=40, max_value=220, value=int(df['thalach'].mean()))
        chest_pain = st.slider("Chest Pain Type (0: Typical, 1: Atypical, 2: Non-anginal, 3: Asymptomatic)", 0, 3, 1)
        resting_bp = st.number_input("Resting Blood Pressure", min_value=60, max_value=200, value=int(df['trestbps'].mean()))

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            predict = st.button("Predict Heart Status", key="predict_heart")
        with col3:
            show_doctors = st.button("Show Recommended Doctors", key="show_heart_doctors")

        # Handle persistent diagnosis and PDF
        if predict:
            sim.input['age'] = age
            sim.input['cholesterol'] = cholesterol
            sim.input['thalach'] = thalach
            sim.input['chest_pain'] = chest_pain
            sim.input['resting_bp'] = resting_bp
            sim.compute()

            risk = sim.output['heart_risk']
            st.session_state.heart_diagnosis = "Yes" if risk >= 50 else "No"
            st.success(f"Predicted Heart Disease Risk: {st.session_state.heart_diagnosis}")

        name = st.text_input("Enter your name", key="user_name_heart")

        st.write(f"🛠️ Debug: diagnosis = {st.session_state.get('heart_diagnosis')}, name = '{name}'")

        if st.session_state.get("heart_diagnosis") and name:
            # Only generate once
            if "heart_pdf" not in st.session_state or st.session_state.get("pdf_name") != name:
                health_tips = [
                    "Maintain a balanced diet",
                    "Exercise regularly",
                    "Avoid processed sugar",
                    "Follow up with a physician"
                ]
                raw_diet = load_diet_plan("Heart Disease", 3, "diet_plans.xlsx")
                diet_text = BeautifulSoup(raw_diet, "html.parser").get_text()
                pdf_file = generate_health_report_pdf(name)
                st.session_state.heart_pdf = pdf_file
                st.session_state.pdf_name = name

            st.download_button(
                label="📄 Download PDF",
                data=st.session_state.heart_pdf,
                file_name=f"{name}_Health_Report.pdf",
                mime="application/pdf"
            )

            # Health Warnings
            warnings = []
            if cholesterol > 240:
                warnings.append("⚠️ High Cholesterol – Increased risk of heart disease.")
            if cholesterol < 125:
                warnings.append("⚠️ Low Cholesterol – May indicate underlying health issues or malnutrition.")
            if resting_bp > 140:
                warnings.append("⚠️ High Resting Blood Pressure – May indicate hypertension.")
            if resting_bp < 100:
                warnings.append("⚠️ Low Resting Blood Pressure – May lead to dizziness or fainting.")
            if thalach < 100:
                warnings.append("⚠️ Low Max Heart Rate – May be a sign of poor cardiovascular fitness.")
            if age > 60:
                warnings.append("⚠️ Age over 60 – Age is a major risk factor for heart disease.")
            if chest_pain == 3:
                warnings.append("⚠️ Asymptomatic Chest Pain Type – Often linked with higher heart disease risk.")

            if warnings:
                warning_html = '<div style="background-color:#fbeaea;padding:10px;border-radius:8px;">' + \
                               '<br>'.join(f'<span style="color:#d00000;">{w}</span>' for w in warnings) + \
                               '</div>'
                st.markdown(warning_html, unsafe_allow_html=True)

        elif show_doctors:
            st.subheader("Recommended Doctors for Heart Disease")
            try:
                doctor_data = pd.read_csv("indian_doctors_dataset.csv")
                h_doctors = doctor_data[
                    doctor_data['Specialist'].str.lower().str.contains("heart", na=False)
                ]
                if not h_doctors.empty:
                    grouped = h_doctors.groupby('Clinic Address')
                    count = 0
                    for name, group in grouped:
                        st.markdown(f"##### 📍 Location: {name}")
                        top5 = group.head(3)[['Doctor Name', 'Specialist', 'Phone Number', 'Email']]
                        st.dataframe(top5, use_container_width=True)
                        count += 1
                        if count >= 3:
                            break
                else:
                    st.info("No doctors found for Heart Disease.")
            except Exception as e:
                st.error(f"Error loading doctors data: {e}")



        
# Diabetes logic

    elif selected == "Diabetes":
        df = load_dataset("diabetes")
        if df is None:
            return

        sim = create_fuzzy_diabetes(df)
        st.subheader("Diabetes")

        glucose = st.number_input("Glucose", min_value=40, max_value=300, value=int(df['Glucose'].mean()), key="glucose_input")
        bmi = st.number_input("BMI", min_value=10.5, max_value=60.0, value=float(df['BMI'].mean()))
        age = st.number_input("Age", min_value=1, max_value=100, value=int(df['Age'].mean()))
        bp = st.number_input("Blood Pressure", min_value=60, max_value=200, value=int(df['BloodPressure'].mean()))

        st.markdown("### 📄 Upload Blood Test Report (CSV, PDF, or Excel)")
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "pdf"])

        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    user_data = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    user_data = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.pdf'):
                    with pdfplumber.open(uploaded_file) as pdf:
                        text = ""
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
                    patterns = {
                        "Glucose": r"Glucose[:\s]+([\d.]+)",
                        "BMI": r"BMI[:\s]+([\d.]+)",
                        "Age": r"Age[:\s]+([\d.]+)",
                        "BloodPressure": r"BloodPressure[:\s]+([\d.]+)"
                    }
                    data = {}
                    for key, pattern in patterns.items():
                        match = re.search(pattern, text, re.IGNORECASE)
                        if match:
                            data[key] = float(match.group(1))
                        else:
                            data[key] = None
                    user_data = pd.DataFrame([data])
                else:
                    st.error("Unsupported file type.")
                    st.stop()

                st.success("✅ Report uploaded successfully!")
                st.dataframe(user_data)

                glucose = user_data['Glucose'][0]
                bmi = user_data['BMI'][0]
                age = user_data['Age'][0]
                bp = user_data['BloodPressure'][0]

            except Exception as e:
                st.error(f"⚠️ Error reading file: {e}")
                return

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            predict = st.button("Predict Diabetes Status", key="predict_diabetes")
        with col3:
            show_doctors = st.button("Show Recommended Doctors", key="show_diabetes_doctors")

        if predict:
            sim.input['glucose'] = glucose
            sim.input['bmi'] = bmi
            sim.input['age'] = age
            sim.input['blood_pressure'] = bp
            sim.compute()

            risk = sim.output['diabetes_risk']
            st.session_state.diabetes_diagnosis = "Yes" if risk >= 50 else "No"
            st.success(f"Predicted Diabetes Status: {st.session_state.diabetes_diagnosis}")

        name = st.text_input("Enter your name", key="user_name_diabetes")
        st.write(f"🛠️ Debug: diagnosis = {st.session_state.get('diabetes_diagnosis')}, name = '{name}'")

        if st.session_state.get("diabetes_diagnosis") and name:
            if "diabetes_pdf" not in st.session_state or st.session_state.get("pdf_name_diabetes") != name:
                health_tips = [
                    "Maintain a balanced diet",
                    "Exercise regularly",
                    "Avoid processed sugar",
                    "Follow up with a physician"
                ]
                raw_diet = load_diet_plan("Diabetes", 3, "diet_plans.xlsx")
                diet_text = BeautifulSoup(raw_diet, "html.parser").get_text()
                pdf_file = generate_health_report_pdf(name)
                st.session_state.diabetes_pdf = pdf_file
                st.session_state.pdf_name_diabetes = name

            st.download_button(
                label="📄 Download PDF",
                data=st.session_state.diabetes_pdf,
                file_name=f"{name}_Health_Report.pdf",
                mime="application/pdf"
            )

            warnings = []
            if glucose > 140:
                warnings.append("⚠️ High glucose")
            if glucose < 70:
                warnings.append("⚠️ Low glucose – Your sugar level is too low.")
            if bmi > 30:
                warnings.append("⚠️ High BMI – Risk of obesity-related diabetes.")
            if bmi < 18.5:
                warnings.append("⚠️ Low BMI – Consider nutritional evaluation.")
            if bp > 80:
                warnings.append("⚠️ High Blood Pressure")
            if bp < 60:
                warnings.append("⚠️ Low Blood Pressure – May cause dizziness.")

            if warnings:
                warning_html = '<div style="background-color:#fbeaea;padding:10px;border-radius:8px;">' + \
                               '<br>'.join(f'<span style="color:#d00000;">{w}</span>' for w in warnings) + \
                               '</div>'
                st.markdown(warning_html, unsafe_allow_html=True)

        elif show_doctors:
            st.subheader("Recommended Doctors for Diabetes")
            try:
                doctor_data = pd.read_csv("indian_doctors_dataset.csv")
                diabetes_doctors = doctor_data[
                    doctor_data['Specialist'].str.lower().str.contains("diabet", na=False)
                ]
                if not diabetes_doctors.empty:
                    grouped = diabetes_doctors.groupby('Clinic Address')
                    count = 0
                    for name, group in grouped:
                        st.markdown(f"##### 📍 Location: {name}")
                        top5 = group.head(3)[['Doctor Name', 'Specialist', 'Phone Number', 'Email']]
                        st.dataframe(top5, use_container_width=True)
                        count += 1
                        if count >= 3:
                            break
                else:
                    st.info("No doctors found for Diabetes.")
            except Exception as e:
                st.error(f"Error loading doctors data: {e}")
# thyroid logic
    elif selected == "Thyroid":
        df = load_dataset("thyroid")
        if df is None:
            return

        sim = create_fuzzy_thyroid(df)
        st.subheader("Thyroid")

        tsh = st.number_input("TSH (mIU/L)", float(df['TSH'].min()), float(df['TSH'].max()), float(df['TSH'].mean()))
        t3 = st.number_input("T3 (ng/dL)", min_value=45.9, max_value=500.0, value=float(df['T3'].mean()))
        t4 = st.number_input("T4 (µg/dL)", min_value=4.5, max_value=500.0, value=float(df['T4'].mean()))

        st.markdown("### 📄 Upload Blood Test Report (CSV, PDF, or Excel)")
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "pdf"])

        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    user_data = pd.read_csv(uploaded_file, encoding='latin1')
                elif uploaded_file.name.endswith('.xlsx'):
                    user_data = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.pdf'):
                    with pdfplumber.open(uploaded_file) as pdf:
                        text = ""
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
                    patterns = {
                        "TSH": r"TSH[:\s]+([\d.]+)",
                        "T3": r"T3[:\s]+([\d.]+)",
                        "T4": r"T4[:\s]+([\d.]+)"
                    }
                    data = {}
                    for key, pattern in patterns.items():
                        match = re.search(pattern, text, re.IGNORECASE)
                        if match:
                            data[key] = float(match.group(1))
                        else:
                            data[key] = None
                    user_data = pd.DataFrame([data])
                else:
                    st.error("Unsupported file type.")
                    st.stop()

                st.success("✅ Report uploaded successfully!")
                st.dataframe(user_data)

                tsh = user_data['TSH'][0]
                t3 = user_data['T3'][0]
                t4 = user_data['T4'][0]

            except Exception as e:
                st.error(f"⚠️ Error reading file: {e}")
                return

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            predict = st.button("Predict Thyroid Status", key="predict_thyroid")
        with col3:
            show_doctors = st.button("Show Recommended Doctors", key="show_thyroid_doctors")

        if predict:
            sim.input['tsh'] = tsh
            sim.input['t3'] = t3
            sim.input['t4'] = t4
            sim.compute()

            risk = sim.output['thyroid_risk']
            st.session_state.thyroid_diagnosis = "Yes" if risk >= 50 else "No"
            st.success(f"Predicted Thyroid Status: {st.session_state.thyroid_diagnosis}")

        name = st.text_input("Enter your name", key="user_name_thyroid")
        st.write(f"🛠️ Debug: diagnosis = {st.session_state.get('thyroid_diagnosis')}, name = '{name}'")

        if st.session_state.get("thyroid_diagnosis") and name:
            if "thyroid_pdf" not in st.session_state or st.session_state.get("pdf_name_thyroid") != name:
                health_tips = [
                    "Maintain a balanced diet",
                    "Exercise regularly",
                    "Avoid processed sugar",
                    "Follow up with a physician"
                ]
                raw_diet = load_diet_plan("Thyroid", 3, "diet_plans.xlsx")
                diet_text = BeautifulSoup(raw_diet, "html.parser").get_text()
                pdf_file = generate_health_report_pdf(name)
                st.session_state.thyroid_pdf = pdf_file
                st.session_state.pdf_name_thyroid = name

            st.download_button(
                label="📄 Download PDF",
                data=st.session_state.thyroid_pdf,
                file_name=f"{name}_Health_Report.pdf",
                mime="application/pdf"
            )

            warnings = []
            if tsh > 4.0:
                warnings.append("⚠️ High TSH – Could indicate hypothyroidism.")
            if tsh < 0.4:
                warnings.append("⚠️ Low TSH – Could be a sign of hyperthyroidism.")
            if t3 < 70:
                warnings.append("⚠️ Low T3 – Often seen in hypothyroidism.")
            if t3 > 200:
                warnings.append("⚠️ High T3 – Might indicate an overactive thyroid.")
            if t4 < 5.0:
                warnings.append("⚠️ Low T4 – May signal thyroid hormone deficiency.")
            if t4 > 12:
                warnings.append("⚠️ High T4 – Often seen in hyperthyroidism.")

            if warnings:
                warning_html = '<div style="background-color:#fbeaea;padding:10px;border-radius:8px;">' + \
                               '<br>'.join(f'<span style="color:#d00000;">{w}</span>' for w in warnings) + \
                               '</div>'
                st.markdown(warning_html, unsafe_allow_html=True)

        elif show_doctors:
            st.subheader("Recommended Doctors for Thyroid")
            try:
                doctor_data = pd.read_csv("indian_doctors_dataset.csv")
                t_doctors = doctor_data[
                    doctor_data['Specialist'].str.lower().str.contains("thyroid", na=False)
                ]
                if not t_doctors.empty:
                    grouped = t_doctors.groupby('Clinic Address')
                    count = 0
                    for name, group in grouped:
                        st.markdown(f"##### 📍 Location: {name}")
                        top5 = group.head(3)[['Doctor Name', 'Specialist', 'Phone Number', 'Email']]
                        st.dataframe(top5, use_container_width=True)
                        count += 1
                        if count >= 3:
                            break
                else:
                    st.info("No doctors found for Thyroid.")
            except Exception as e:
                st.error(f"Error loading doctors data: {e}")


# PCOD Logic          

    elif selected == "PCOD":
        df = load_dataset("pcod")
        if df is None:
            return

        sim = create_fuzzy_pcod(df)
        st.subheader("PCOD")

        bmi = st.number_input("BMI", min_value=10.5, max_value=60.0, value=float(df['BMI'].mean()))
        insulin = st.number_input("Insulin Level", float(df['Insulin_Level'].min()), max_value=400.0, value=float(df['Insulin_Level'].mean()))
        lh = st.number_input("LH", float(df['LH'].min()), max_value=120.0, value=float(df['LH'].mean()))

        st.markdown("### 📄 Upload Blood Test Report (CSV, PDF, or Excel)")
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "pdf"])

        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    user_data = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    user_data = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.pdf'):
                    with pdfplumber.open(uploaded_file) as pdf:
                        text = ""
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
                    patterns = {
                        "BMI": r"BMI[:\s]+([\d.]+)",
                        "Insulin_Level": r"Insulin[_\s]?Level[:\s]+([\d.]+)",
                        "LH": r"LH[:\s]+([\d.]+)"
                    }
                    data = {}
                    for key, pattern in patterns.items():
                        match = re.search(pattern, text, re.IGNORECASE)
                        data[key] = float(match.group(1)) if match else None
                    user_data = pd.DataFrame([data])
                else:
                    st.error("Unsupported file type.")
                    st.stop()

                st.success("✅ Report uploaded successfully!")
                st.dataframe(user_data)

                bmi = user_data['BMI'][0]
                insulin = user_data['Insulin_Level'][0]
                lh = user_data['LH'][0]

            except Exception as e:
                st.error(f"⚠️ Error reading file: {e}")
                return

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            predict = st.button("Predict PCOD Status", key="predict_pcod")
        with col3:
            show_doctors = st.button("Show Recommended Doctors", key="show_pcod_doctors")

        if predict:
            sim.input['bmi'] = bmi
            sim.input['insulin'] = insulin
            sim.input['lh'] = lh
            sim.compute()

            risk = sim.output['pcod_risk']
            st.session_state.pcod_diagnosis = "Yes" if risk >= 50 else "No"
            st.success(f"Predicted PCOD Status: {st.session_state.pcod_diagnosis}")

        name = st.text_input("Enter your name", key="user_name_pcod")
        st.write(f"🛠️ Debug: diagnosis = {st.session_state.get('pcod_diagnosis')}, name = '{name}'")

        if st.session_state.get("pcod_diagnosis") and name:
            if "pcod_pdf" not in st.session_state or st.session_state.get("pdf_name_pcod") != name:
                health_tips = [
                    "Maintain a balanced diet",
                    "Exercise regularly",
                    "Avoid processed sugar",
                    "Follow up with a physician"
                ]
                raw_diet = load_diet_plan("PCOD", 3, "diet_plans.xlsx")
                diet_text = BeautifulSoup(raw_diet, "html.parser").get_text()
                pdf_file = generate_health_report_pdf(name)
                st.session_state.pcod_pdf = pdf_file
                st.session_state.pdf_name_pcod = name

            st.download_button(
                label="📄 Download PDF",
                data=st.session_state.pcod_pdf,
                file_name=f"{name}_Health_Report.pdf",
                mime="application/pdf"
            )

            warnings = []
            if bmi > 25:
                warnings.append("⚠️ High BMI – Obesity is a major risk factor for PCOD.")
            if bmi < 18.5:
                warnings.append("⚠️ Low BMI – Consider monitoring nutritional health.")
            if insulin > 15:
                warnings.append("⚠️ Elevated Insulin – May indicate insulin resistance.")
            if insulin < 10:
                warnings.append("⚠️ Very Low Insulin – May need medical attention.")
            if lh > 9:
                warnings.append("⚠️ High LH – Can contribute to irregular ovulation.")

            if warnings:
                warning_html = '<div style="background-color:#fbeaea;padding:10px;border-radius:8px;">' + \
                               '<br>'.join(f'<span style="color:#d00000;">{w}</span>' for w in warnings) + \
                               '</div>'
                st.markdown(warning_html, unsafe_allow_html=True)

        elif show_doctors:
            st.subheader("Recommended Doctors for PCOD")
            try:
                doctor_data = pd.read_csv("indian_doctors_dataset.csv")
                p_doctors = doctor_data[
                    doctor_data['Specialist'].str.lower().str.contains("pcod", na=False)
                ]
                if not p_doctors.empty:
                    grouped = p_doctors.groupby('Clinic Address')
                    count = 0
                    for name, group in grouped:
                        st.markdown(f"##### 📍 Location: {name}")
                        top5 = group.head(3)[['Doctor Name', 'Specialist', 'Phone Number', 'Email']]
                        st.dataframe(top5, use_container_width=True)
                        count += 1
                        if count >= 3:
                            break
                else:
                    st.info("No doctors found for PCOD.")
            except Exception as e:
                st.error(f"Error loading doctors data: {e}")


# Anxiety logic

    elif selected == "Anxiety":
        df = load_dataset("anxiety")
        if df is None:
            return

        sim = create_fuzzy_anxiety(df)
        st.subheader("Anxiety")

        sleep = st.number_input("Sleep Hours", min_value=0.0, max_value=12.0, value=float(df['SleepHours'].mean()))
        heart_rate = st.number_input("Heart Rate", min_value=40, max_value=220, value=int(df['HeartRate'].mean()))

        fatigue_str = st.selectbox("Fatigue", ["No", "Yes"])
        fatigue = 1 if fatigue_str == "Yes" else 0

        irritability_str = st.selectbox("Irritability", ["No", "Yes"])
        irritability = 1 if irritability_str == "Yes" else 0

        restlessness_str = st.selectbox("Restlessness", ["No", "Yes"])
        restlessness = 1 if restlessness_str == "Yes" else 0

        score = st.slider("GAD-7 Score", int(df['ScoreGAD7'].min()), int(df['ScoreGAD7'].max()), int(df['ScoreGAD7'].mean()))

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            predict = st.button("Predict Anxiety Status", key="predict_anxiety")
        with col3:
            show_doctors = st.button("Show Recommended Doctors", key="show_anxiety_doctors")

        if predict:
            sim.input['sleep'] = sleep
            sim.input['heart_rate'] = heart_rate
            sim.input['fatigue'] = fatigue
            sim.input['irritability'] = irritability
            sim.input['restlessness'] = restlessness
            sim.input['score'] = score
            sim.compute()

            risk = sim.output['anxiety_risk']
            st.session_state.anxiety_diagnosis = "Yes" if risk >= 50 else "No"
            st.success(f"Predicted Anxiety Status: {st.session_state.anxiety_diagnosis}")

        name = st.text_input("Enter your name", key="user_name_anxiety")
        st.write(f"🛠️ Debug: diagnosis = {st.session_state.get('anxiety_diagnosis')}, name = '{name}'")

        if st.session_state.get("anxiety_diagnosis") and name:
            if "anxiety_pdf" not in st.session_state or st.session_state.get("pdf_name_anxiety") != name:
                health_tips = [
                    "Maintain a balanced diet",
                    "Exercise regularly",
                    "Avoid processed sugar",
                    "Follow up with a physician"
                ]
                raw_diet = load_diet_plan("Anxiety", 3, "diet_plans.xlsx")
                diet_text = BeautifulSoup(raw_diet, "html.parser").get_text()
                pdf_file = generate_health_report_pdf(name)
                st.session_state.anxiety_pdf = pdf_file
                st.session_state.pdf_name_anxiety = name

            st.download_button(
                label="📄 Download PDF",
                data=st.session_state.anxiety_pdf,
                file_name=f"{name}_Health_Report.pdf",
                mime="application/pdf"
            )

            warnings = []
            if restlessness == 1:
                warnings.append("⚠️ High Restlessness – Major contributor to anxiety.")
            if sleep < 6:
                warnings.append("⚠️ Poor Sleep – Can worsen anxiety.")
            if fatigue == 1:
                warnings.append("⚠️ Fatigue – Often linked with anxiety symptoms.")
            if irritability == 1:
                warnings.append("⚠️ Irritability – A strong emotional indicator.")
            if sleep > 9:
                warnings.append("⚠️ Oversleeping – May signal underlying issues.")
            if heart_rate < 60:
                warnings.append("⚠️ Low Heart Rate – May indicate bradycardia or exhaustion.")
            if heart_rate > 100:
                warnings.append("⚠️ Elevated Heart Rate – Could be a sign of anxiety or stress.")
            if score < 5:
                warnings.append("✅ Low GAD-7 Score – Minimal anxiety risk.")
            elif score >= 15:
                warnings.append("⚠️ Severe anxiety – Consider professional evaluation.")

            if warnings:
                warning_html = '<div style="background-color:#fbeaea;padding:10px;border-radius:8px;">' + \
                               '<br>'.join(f'<span style="color:#d00000;">{w}</span>' for w in warnings) + \
                               '</div>'
                st.markdown(warning_html, unsafe_allow_html=True)

        elif show_doctors:
            st.subheader("Recommended Doctors for Anxiety")
            try:
                doctor_data = pd.read_csv("indian_doctors_dataset.csv")
                a_doctors = doctor_data[
                    doctor_data['Specialist'].str.lower().str.contains("anxiety", na=False)
                ]
                if not a_doctors.empty:
                    grouped = a_doctors.groupby('Clinic Address')
                    count = 0
                    for name, group in grouped:
                        st.markdown(f"##### 📍 Location: {name}")
                        top5 = group.head(3)[['Doctor Name', 'Specialist', 'Phone Number', 'Email']]
                        st.dataframe(top5, use_container_width=True)
                        count += 1
                        if count >= 3:
                            break
                else:
                    st.info("No doctors found for Anxiety.")
            except Exception as e:
                st.error(f"Error loading doctors data: {e}")

    elif selected == "Diet Recommendation":
        st.subheader("🍽️ Diet Recommendation")

        condition = st.selectbox("Select Condition", ["Heart Disease", "Diabetes", "Thyroid", "PCOD", "Anxiety"])
        days = st.selectbox("Select Duration (in days)", [1, 2, 3, 5, 7], index=2)

        if st.button("Show Diet Plan", key="show_diet_plan"):
            try:
                plan_text = load_diet_plan(condition, days, "diet_plans.xlsx")
                st.markdown(plan_text)
            except Exception as e:
                st.error(f"Failed to load diet plan: {e}")




if __name__ == "__main__":
    main()
