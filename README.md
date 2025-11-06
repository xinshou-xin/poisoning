# 🧪 Poisoning Outcome Prediction App

**Streamlit-based clinical AI tool** that predicts:

* **Model 1 (Mortality)**: Death or survival in poisoning cases
* **Model 2 (Recovery Status)**: Recovery outcome for surviving patients (recovered/unrecovered)

Designed for **real-time decision support** in poisoning emergencies, this project integrates machine learning with clinical domain knowledge to assist emergency care providers in assessing patient prognosis.

---

## 📘 Table of Contents

* [🔍 Overview](#-overview)
* [🧠 Models Description](#-models-description)
* [🖥️ App Features](#️-app-features)
* [📦 Installation & Deployment](#-installation--deployment)
* [🧪 Dataset & Preprocessing](#-dataset--preprocessing)
* [⚠️ Disclaimer](#️-disclaimer)
* [📬 Contact](#-contact)

---

## 🔍 Overview

Poisoning is a critical medical emergency with highly variable outcomes. Prognostication tools can guide clinical decisions, treatment strategies, and resource allocation. This app provides real-time probability estimates of:

* Mortality (Model 1): Whether the patient will survive the poisoning incident
* Recovery Status (Model 2): Whether surviving patients will achieve recovery or remain unrecovered

Both models are based on **CatBoost classifiers**, trained on labeled clinical data, and made interpretable with **SHAP** visualizations.

---

## 🧠 Models Description

### 🩺 Model 1 – Mortality Prediction

* **Goal**: Estimate the likelihood of death or survival in poisoning cases.
* **Inputs**:

  * Poison type and dosage
  * Route of exposure (oral, dermal, inhalation, etc.)
  * Time from exposure to treatment
  * Initial vital signs (blood pressure, heart rate, oxygen saturation)
  * Patient demographics (age, comorbidities)
* **Output**: Probability of mortality from poisoning.

### 🏥 Model 2 – Recovery Status Prediction

* **Goal**: Estimate recovery outcome for patients who survive the acute poisoning phase.
* **Inputs**:

  * All Model 1 features
  * Treatment response (antidote effectiveness, detoxification success)
  * Hospital course (complications, organ function)
  * Duration of critical care
* **Output**: Probability of full recovery vs. unrecovered status.

---

## 🖥️ App Features

✅ **Interactive User Input**:
Easily enter model variables using dropdowns, sliders, and forms.

✅ **Dual Model Switching**:
Choose between mortality or recovery prediction via sidebar toggle.

✅ **Visual Model Explanation**:
Understand model output with SHAP bar plots showing feature contributions.

✅ **Persistent State**:
Predictions are updated only when the “Predict” button is clicked to avoid auto-refresh.

✅ **Compact UI**:
Tabbed or accordion-style layout to group features by category (exposure details, clinical status, treatment).

---

## 📦 Installation & Deployment

### 🔧 Local Setup

```bash
# Clone the repo
git clone https://github.com/xinshou-xin/poisoning.git
cd poisoning-prediction

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Launch the app
streamlit run main.py
```

### ☁️ Streamlit Cloud

The app is deployable via [Streamlit Community Cloud](https://streamlit.io/cloud). Just connect your GitHub repo, and make sure you include:

* `main.py` (entry point)
* `requirements.txt`
* `.streamlit/runtime.txt` (e.g. `python-3.10`)

---

## 🧪 Dataset & Preprocessing

The models were trained using a structured poisoning registry with features including:

* Exposure details: Poison type, route, dosage, timing
* Clinical data: Vital signs, symptoms, laboratory findings
* Treatment information: Antidotes, supportive care, duration
* Outcomes: Survival status, recovery level

Missing values were handled with median/mode imputation. Feature importance was assessed using SHAP values and clinical domain knowledge.

---

## ⚠️ Disclaimer

> This tool is intended **for research and educational use only**.
> It is **not approved** for clinical use and should **not replace medical judgment**.
> External validation and expert oversight are required before integration into poisoning management workflows.

---

## 📬 Contact

For collaboration, questions, or feedback:

**huangjinxin**
Email: `2602535898@qq.com`
GitHub: https://github.com/xinshou-xin

---