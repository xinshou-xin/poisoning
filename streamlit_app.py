import streamlit as st
import joblib
from PIL import Image
import plotly.express as px
import pandas as pd
import base64
import io
import shap
import matplotlib.pyplot as plt
import matplotlib
from streamlit_echarts import st_echarts

if not hasattr(pd.DataFrame, "iteritems"):
    pd.DataFrame.iteritems = pd.DataFrame.items

# matplotlib.use("Agg")

st.set_page_config(page_title="Poisoning Prediction", layout="wide", page_icon="🦈")

st.set_option('deprecation.showPyplotGlobalUse', False)

# =================== Sidebar ===================
# model_choice = st.sidebar.radio("Please select a prediction model:", ["Model 1 (Mortality Prediction)", "Model 2 (Recovery Status Prediction)"])
st.sidebar.markdown(
    """
    <div style=' font-weight: bold;'>Please select a prediction model:</div>
    """,
    unsafe_allow_html=True
)
model_choice = st.sidebar.radio(
    "", ["Model 1 (Mortality Prediction)", "Model 2 (Recovery Status Prediction)"]
)

# Reset prediction when model is changed
if "last_model" not in st.session_state:
    st.session_state["last_model"] = model_choice
if st.session_state["last_model"] != model_choice:
    st.session_state["predict_done"] = False
    st.session_state["last_model"] = model_choice

# =================== Sidebar Info ===================
st.sidebar.markdown("""
### 📖 Model Overview

**Model 1: Mortality Prediction**  
Predicts the likelihood of death or survival in cases of poisoning, using key indicators such as poison type, exposure route, time to treatment, and initial vital signs.

**Model 2: Recovery Status Prediction**  
For patients who survive poisoning, this model further their recovery outcome into "recovered" or "unrecovered" based on clinical course, treatment response, and post-exposure complications.
""")


# # 更改预测状态按钮
# if st.sidebar.button("🔄 Reset Prediction"):
#     st.session_state["predict_done"] = False

# =================== Header(免责声明) ===================

st.sidebar.markdown(
    """
    <div style='font-size: 0.9em;'>
       ⚠️ Note: This model was developed using poisoning data from specific populations, and its applicability to other types of poisoning cases or populations may be limited.
    </div>
    """,
    unsafe_allow_html=True
)

if model_choice == "Model 1 (Mortality Prediction)":
    st.markdown("## Mortality Prediction")
elif model_choice == "Model 2 (Recovery Status Prediction)":
    st.markdown("## Recovery Status Prediction")

# =================== Pre-hospital Features (A组) ===================
# st.markdown("## Return of Spontaneous Circulation on-site")

a_data = {}
b_data = {}
if model_choice == "Model 1 (Mortality Prediction)":
   
    col1, col2, col3 = st.columns(3)
    
    with col1:

        a_data["Age"] = st.number_input("Age", 0, 120, 45)
        poison_type = st.selectbox("Type of Poisoning",["Uncertain", "Industrial", "Pharmaceutical", "Pesticide", "Alcohol"])
        poison_mapping = {'Uncertain':0, 'Industrial':1, 'Pharmaceutical':2, 'Pesticide':3, 'Alcohol':4}
        a_data["Type of Poisoning"] = poison_mapping[poison_type]
        severity = st.selectbox("Degree of poisoning",["Unjudgeable", "Mild", "Moderate", "Severe"])
        severity_mapping = {'Unjudgeable':0, 'Mild':1, 'Moderate':2, 'Severe':3}
        a_data["Degree of poisoning"] = severity_mapping[severity]
        a_data["White Blood Cell Count"] = st.number_input("White Blood Cell Count (10*9/L)", 0.0, 50.0, 7.5)
        a_data["Alanine Aminotransferase (ALT)"] = st.number_input("Alanine Aminotransferase (ALT) (U/L)", 0.0, 2000.0, 40.0)
        a_data["Total Bilirubin"] = st.number_input("Total Bilirubin (umol/L)", 0.0, 500.0, 15.0)
        a_data["Lactate Dehydrogenase (LDH)"] = st.number_input("Lactate Dehydrogenase (LDH) (U/L)", 50.0, 5000.0, 250.0)

    with col2:
        a_data["Urea"] = st.number_input("Urea (mmol/L)", 0.0, 50.0, 5.0)
        a_data["Uric Acid"] = st.number_input("Uric Acid (umol/L)", 50.0, 1500.0, 300.0)
        a_data["Creatine Kinase (CK)"] = st.number_input("CK (ng/mL)", 0.0, 10000.0, 150.0)
        a_data["Creatine Kinase-MB Isoenzyme"] = st.number_input("Creatine Kinase-MB Isoenzyme (ng/mL)", 0.0, 1000.0, 20.0)
        a_data["Albumin (First Measurement)"] = st.number_input("Albumin (First Measurement) (g/L)", 10.0, 60.0, 40.0)
        a_data["Length of Stay"] = st.number_input("Length of Stay (days)", 0, 365, 7)
        
        # Education level
        edu_level = st.selectbox(
            "Education Level",
            ["Illiterate", "Primary school", "Junior high school", "High school", "College"]
        )
        edu_mapping = {'Illiterate':1, 'Primary school':2, 'Junior high school':3, 'High school':4, 'College':5}
        a_data["Education Level"] = edu_mapping[edu_level]
        
    with col3:
        a_data["Red Blood Cell Count"] = st.number_input("Red Blood Cell Count (10*12/L)", 1.0, 7.0, 4.5)
        a_data["Hemoglobin Concentration"] = st.number_input("Hemoglobin Concentration (g/L)", 30.0, 200.0, 130.0)
        a_data["Mean Corpuscular Hemoglobin Concentration"] = st.number_input("Mean Corpuscular Hemoglobin Concentration (g/L)", 200.0, 400.0, 330.0)
        a_data["Direct Bilirubin"] = st.number_input("Direct Bilirubin (umol/L)", 0.0, 300.0, 5.0)
        a_data["Homocysteine"] = st.number_input("Homocysteine (umol/L)", 0.0, 100.0, 10.0)
        a_data["Altered Consciousness or Syncope"] = int(st.selectbox("Altered Consciousness or Syncope", ["No", "Yes"]) == "Yes")
    
# =================== Post-hospital Features (B组，仅M2) ===================
else:
    # st.markdown("## Recovery Status Prediction")
    col1, col2, col3 = st.columns(3)

    # Column 1
    with col1:
        b_data["Age"] = st.number_input("Age", 0, 120, 45)
        poison_type = st.selectbox("Type of Poisoning", ["Uncertain", "Industrial", "Pharmaceutical", "Pesticide", "Alcohol"])
        b_data["Type of Poisoning"] = {"Uncertain":0, "Industrial":1, "Pharmaceutical":2, "Pesticide":3, "Alcohol":4}[poison_type]
        poison_severity = st.selectbox("Degree of poisoning", ["Unjudgeable", "Mild", "Moderate", "Severe"])
        b_data["Degree of poisoning"] = {"Unjudgeable":0, "Mild":1, "Moderate":2, "Severe":3}[poison_severity]
        b_data["White Blood Cell Count"] = st.number_input("White Blood Cell Count (10*9/L)", 0.0, 50.0, 7.5)
        b_data["Alanine Aminotransferase (ALT)"] = st.number_input("Alanine Aminotransferase (ALT) (U/L)", 0.0, 2000.0, 40.0)
        b_data["Total Bilirubin"] = st.number_input("Total Bilirubin (umol/L)", 0.0, 500.0, 15.0)
        b_data["Lactate Dehydrogenase (LDH)"] = st.number_input("Lactate Dehydrogenase (LDH) (U/L)", 50.0, 5000.0, 250.0)


    # Column 2
    with col2:
        b_data["Urea"] = st.number_input("Urea (mmol/L)", 0.0, 50.0, 5.0)
        b_data["Uric Acid"] = st.number_input("Uric Acid (umol/L)", 50.0, 1500.0, 300.0)
        b_data["Creatine Kinase (CK)"] = st.number_input("CK (ng/mL)", 0.0, 10000.0, 150.0)
        b_data["Creatine Kinase-MB Isoenzyme"] = st.number_input("Creatine Kinase-MB Isoenzyme (ng/mL)", 0.0, 1000.0, 20.0)
        b_data["Albumin (First Measurement)"] = st.number_input("Albumin (First Measurement) (g/L)", 10.0, 60.0, 40.0)
        b_data["Length of Stay"] = st.number_input("Length of Stay (days)", 0, 365, 7)
        b_data["High-Sensitivity C-Reactive Protein (hs-CRP)"] = st.number_input("High-Sensitivity C-Reactive Protein (hs-CRP) (mg/L)", 0.0, 500.0, 5.0)


    # Column 3
    with col3:
        b_data["Blood Cholinesterase Test Results"] = st.number_input("Blood Cholinesterase Test Results (U/L)", 0.0, 20000.0, 6000.0)
        b_data["Albumin (Last Measurement)"] = st.number_input("Albumin (Last Measurement) (g/L)", 10.0, 60.0, 38.0)
        b_data["Weight"] = st.number_input("Weight (Kg)", 1.0, 200.0, 65.0)
        b_data["Diastolic Blood Pressure"] = st.number_input("Diastolic Blood Pressure (mmHg)", 40, 120, 80)
        b_data["Mean Corpuscular Volume"] = st.number_input("Mean Corpuscular Volume (fL)", 60.0, 150.0, 90.0)
        b_data["Vomiting"] = int(st.selectbox("Vomiting", ["No", "Yes"]) == "Yes")


# =================== Load Models & Data ===================
model1_path = "M1_compare/models/catboost_model_fold_3.pkl"
model2_path = "M2_compare/models/catboost_model_fold_1.pkl"
shap_fig1_path = "M1_compare/SHAP/shap_summary_plot.png"
shap_fig2_path = "M2_compare/SHAP/shap_summary_plot.png"
shap1_data_csv = pd.read_csv("M1_compare/SHAP/shap_data.csv")
shap1_value_csv = pd.read_csv("M1_compare/SHAP/shap_values.csv")
shap2_data_csv = pd.read_csv("M2_compare/SHAP/shap_data.csv")
shap2_value_csv = pd.read_csv("M2_compare/SHAP/shap_values.csv")


x_features_m1 = [
    'Age', 'Education Level', 'Type of Poisoning', 'Degree of poisoning',
    'Altered Consciousness or Syncope', 'White Blood Cell Count',
    'Red Blood Cell Count', 'Hemoglobin Concentration',
    'Mean Corpuscular Hemoglobin Concentration', 'Alanine Aminotransferase (ALT)',
    'Total Bilirubin', 'Direct Bilirubin', 'Lactate Dehydrogenase (LDH)',
    'Urea', 'Uric Acid', 'Creatine Kinase (CK)', 'Creatine Kinase-MB Isoenzyme',
    'Homocysteine', 'Albumin (First Measurement)', 'Length of Stay'
]
x_features_m2 = [
    'Age', 'Length of Stay', 'Weight', 'Diastolic Blood Pressure',
    'Type of Poisoning', 'Degree of poisoning', 'Vomiting',
    'White Blood Cell Count', 'Mean Corpuscular Volume',
    'Alanine Aminotransferase (ALT)', 'Total Bilirubin',
    'Lactate Dehydrogenase (LDH)', 'Urea', 'Uric Acid',
    'Creatine Kinase (CK)', 'Creatine Kinase-MB Isoenzyme',
    'High-Sensitivity C-Reactive Protein (hs-CRP)',
    'Blood Cholinesterase Test Results',
    'Albumin (First Measurement)',
    'Albumin (Last Measurement)'
]

if model_choice == "Model 1 (Mortality Prediction)":
    # st.write("a_data:", a_data)
    model = joblib.load(model1_path)
    shap_fig_path = shap_fig1_path
    shap_data = shap1_data_csv
    shap_value = shap1_value_csv
    features = x_features_m1
else:
    model = joblib.load(model2_path)
    shap_fig_path = shap_fig2_path
    shap_data = shap2_data_csv
    shap_value = shap2_value_csv
    features = x_features_m2

# =================== Prediction ===================
# 按钮样式
st.markdown("""
    <style>
        /* 默认样式 */
        .stButton > button {
            display: block;
            margin: 0 auto;
            background-color: #e8f0fe;
            color: #1967d2;
            border: 1px solid #1967d2;
            border-radius: 10px;
            padding: 12px 28px;
            font-size: 16px;
            font-weight: 600;
            transition: all 0.3s ease;
        }

        /* hover 状态 */
        .stButton > button:hover {
            background-color: #d2e3fc;
            color: #174ea6;
            border-color: #174ea6;
        }

        /* 按下状态 */
        .stButton > button:active {
            background-color: #e6f4ea;  /* 淡绿色 */
            color: #137333;
            border-color: #137333;
        }

        /* 点击后获得焦点状态 */
        .stButton > button:focus:not(:active) {
            background-color: #e6f4ea;
            color: #137333;
            border-color: #137333;
        }
    </style>
""", unsafe_allow_html=True)
if st.button("🚀 Predict"):
    st.session_state["predict_done"] = True

# 仪表盘配置模板函数
def get_gauge_option(value):
    return {
        "series": [
            {
                "type": "gauge",
                "center": ["50%", "60%"],
                "startAngle": 200,
                "endAngle": -20,
                "min": 0,
                "max": 100,
                "splitNumber": 10,
                "itemStyle": {
                    "color": "#91cc75"
                },
                "progress": {
                    "show": True,
                    "width": 30
                },
                "pointer": {
                    "show": False
                },
                "axisLine": {
                    "lineStyle": {
                        "width": 30
                    }
                },
                "axisTick": {
                    "distance": -45,
                    "splitNumber": 5,
                    "lineStyle": {
                        "width": 2,
                        "color": "#999"
                    }
                },
                "splitLine": {
                    "distance": -52,
                    "length": 14,
                    "lineStyle": {
                        "width": 3,
                        "color": "#999"
                    }
                },
                "axisLabel": {
                    "distance": -20,
                    "color": "#666",
                    "fontSize": 14
                },
                "anchor": {
                    "show": False
                },
                "title": {
                    "show": False  # 隐藏内部 title
                },
                "detail": {
                    "valueAnimation": True,
                    "width": "60%",
                    "lineHeight": 40,
                    "borderRadius": 8,
                    "offsetCenter": [0, "-15%"],
                    "fontSize": 30,
                    "fontWeight": "bolder",
                    "formatter": "{value} %",
                    "color": "#000"
                },
                "data": [
                    {
                        "value": round(value * 100, 1)
                    }
                ]
            }
        ]
    }


if st.session_state.get("predict_done", False):

    model_feature_names = model.feature_names_

    # st.write("model_feature_names:", model_feature_names)

    if model_choice == "Model 1 (Mortality Prediction)":
        X_input_df = pd.DataFrame([a_data])
        X_input_df = X_input_df.reindex(columns=model_feature_names)
        proba = model.predict_proba(X_input_df)[0][1]
        label_text = "Probability of Mortality"
    else:
        X_input_df = pd.DataFrame([b_data])
        X_input_df = X_input_df.reindex(columns=model_feature_names)
        proba = model.predict_proba(X_input_df)[0][1]
        label_text = "Probability of Non-recovery"


    st.write("<h2>Predict Result</h2>", unsafe_allow_html=True)
    
    st.markdown(f'<h5 style="color: #0775eb"> {label_text}:</h5>', unsafe_allow_html=True)

    # 显示仪表盘
    st_echarts(get_gauge_option(proba), height="400px")

    # =================== SHAP 可视化 ===================
    st.write("<h2>SHAP Analysis Visualization</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<h5 style="color: #0775eb">Summary plot:</h5>', unsafe_allow_html=True)
        st.image(Image.open(shap_fig_path), caption=f"SHAP Summary Plot ({model_choice})")
    with col2:
        st.markdown('<h5 style="color: #0775eb">Dependence plot:</h5>', unsafe_allow_html=True)
        selected_feature = st.selectbox("Choose a feature", shap_data.columns)
        fig = px.scatter(
            x=shap_data[selected_feature],
            y=shap_value[selected_feature],
            color=shap_data[selected_feature],
            color_continuous_scale=["blue", "red"],
            labels={"x": "Original value", "y": "SHAP value"},
        )
        st.write(fig)

    # =================== Waterfall ===================
    # st.write("<h2>Personalized Risk Interpretation</h2>", unsafe_allow_html=True)
    # st.markdown('<h5 style="color: #0775eb">Waterfall plot:</h5>', unsafe_allow_html=True)
    # explainer = shap.Explainer(model)
    # X_input_df = pd.DataFrame([X_input], columns=features)
    # shap_values = explainer(X_input_df)
    # plt.figure(figsize=(8, 5))
    # shap.waterfall_plot(shap_values[0], max_display=shap_values[0].values.shape[0], show=False)
    # buf = io.BytesIO()
    # plt.savefig(buf, format='png', bbox_inches='tight')
    # buf.seek(0)
    # st.markdown(
    #     f'<div style="display: flex; justify-content: center;">'
    #     f'<img src="data:image/png;base64,{base64.b64encode(buf.read()).decode()}" alt="Waterfall plot" style="width: 900px;">'
    #     f'</div>', unsafe_allow_html=True)
    # plt.close()
    # =================== Waterfall ===================
    st.write("<h2>Personalized Risk Interpretation</h2>", unsafe_allow_html=True)
    st.markdown('<h5 style="color: #0775eb">Waterfall plot:</h5>', unsafe_allow_html=True)

    explainer = shap.Explainer(model)
    shap_values = explainer(X_input_df)
    plt.figure(figsize=(8, 5))
    shap.waterfall_plot(shap_values[0], max_display=shap_values[0].values.shape[0], show=False)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    st.markdown(
        f'<div style="display: flex; justify-content: center;">'
        f'<img src="data:image/png;base64,{base64.b64encode(buf.read()).decode()}" alt="Waterfall plot" style="width: 900px;">'
        f'</div>', unsafe_allow_html=True)
    plt.close()

    # =================== Force ===================
    st.markdown('<h5 style="color: #0775eb">Force plot:</h5>', unsafe_allow_html=True)
    shap_values_array = shap.Explainer(model).shap_values(X_input_df.values)
    fig = shap.force_plot(explainer.expected_value, shap_values_array[0], X_input_df, matplotlib=True)
    st.pyplot(fig)


