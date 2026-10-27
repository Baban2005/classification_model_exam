# Healthcare Classification Model Exam: 30-Day Patient Readmission

This repository contains a synthetic classification dataset, exploratory data analysis, data preprocessing workflow, and machine learning models predicting 30-day unplanned hospital readmissions in the healthcare sector.

## Direct Links

- Raw Dataset (CSV): [healthcare_patient_readmission.csv](https://raw.githubusercontent.com/Baban2005/classification_model_exam/main/healthcare_patient_readmission.csv)
- Google Colab Notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Baban2005/classification_model_exam/blob/main/Healthcare_Patient_Readmission_Classification.ipynb)

---

## Dataset Description

- Sector: Healthcare
- Table Name: `healthcare_patient_readmission.csv`
- Records: 650 patient intake and discharge records
- Problem Type: Binary Classification
- Dimensions: 650 rows, 6 columns (5 numerical features and 1 target response variable)

Unplanned hospital readmissions within 30 days of discharge place severe strain on hospital capacity, raise patient morbidity, and result in reimbursement penalties from healthcare payers. This dataset simulates clinical, vital, and metabolic measurements collected from adult patients at discharge to identify individuals at elevated risk of unplanned readmission.

### Data Dictionary

| Column Name | Data Type | Description | Unit / Range |
| :--- | :--- | :--- | :--- |
| `age` | Integer (`int64`) | Patient age at admission | 25 to 85 years |
| `bmi` | Float (`float64`) | Body mass index | 18.5 to 44.0 kg/m² |
| `fasting_glucose` | Float (`float64`) | Fasting blood glucose level | 70.0 to 260.0 mg/dL |
| `systolic_bp` | Integer (`int64`) | Resting systolic blood pressure | 95 to 178 mmHg |
| `prior_admissions` | Integer (`int64`) | Number of admissions in preceding 12 months | 0 to 6 admissions |
| `readmitted_30d` | Integer (`int64`) | Target: unplanned readmission within 30 days | 0 = No, 1 = Yes |

All predictor features are numerical, removing the need for categorical dummy encoding and simplifying preprocessing.

---

## Repository Contents

- `healthcare_patient_readmission.csv`: The primary dataset containing 650 patient records.
- `Healthcare_Patient_Readmission_Classification.ipynb`: Google Colab compatible notebook demonstrating exploratory data analysis, data preprocessing, and model training.
- `generate_dataset.py`: Python script used to synthesize the dataset using pandas and numpy.
- `README.md`: Project summary, data dictionary, and documentation.

---

## Notebook Structure

1. Data Loading: Direct ingestion of the raw CSV file hosted on GitHub via pandas.
2. Exploratory Data Analysis (EDA):
   - Structural verification (`info`, `describe`, null check, duplicate check)
   - Target class distribution inspection (37.2% readmission rate)
   - Univariate distributions with histograms and density estimates
   - Bivariate boxplots stratified by readmission status
   - Pearson correlation matrix and heatmap
3. Data Preprocessing:
   - Predictor and target separation ($X$ and $y$)
   - Stratified 80/20 train-test split (`stratify=y`, `random_state=42`)
   - Feature standardization via `StandardScaler` fitted strictly on training data to prevent leakage
4. Classification Modeling & Evaluation:
   - Logistic Regression
   - Random Forest Classifier
   - Confusion matrices, accuracy, precision, recall, F1-scores, and ROC-AUC comparison curves
5. Findings and Discussion: Summary of feature importances and clinical takeaways.
