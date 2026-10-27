import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Generate 650 patient records (exceeding the 500 row minimum requirement)
n_samples = 650

# Feature 1: age (Patient age in years, integer: 25 to 85)
age = np.clip(np.random.normal(loc=58, scale=14, size=n_samples).round(), 25, 85).astype(int)

# Feature 2: bmi (Body Mass Index, float: 18.5 to 44.0 kg/m^2)
bmi = np.clip(np.random.normal(loc=28.4, scale=5.1, size=n_samples), 18.5, 44.0).round(1)

# Feature 3: fasting_glucose (Fasting blood glucose level, float: 70.0 to 260.0 mg/dL)
fasting_glucose = np.clip(np.random.normal(loc=126.0, scale=36.0, size=n_samples), 70.0, 260.0).round(1)

# Feature 4: systolic_bp (Resting systolic blood pressure, integer: 95 to 178 mmHg)
systolic_bp = np.clip(np.random.normal(loc=132, scale=17, size=n_samples).round(), 95, 178).astype(int)

# Feature 5: prior_admissions (Prior hospital admissions within past 12 months, integer: 0 to 6)
prior_admissions = np.clip(np.random.poisson(lam=1.3, size=n_samples), 0, 6).astype(int)

# Target: readmitted_30d (0 = No, 1 = Yes)
# Realistic logistic probability model based on clinical risk factors
z_age = (age - 58) / 14
z_bmi = (bmi - 28.4) / 5.1
z_gluc = (fasting_glucose - 126.0) / 36.0
z_bp = (systolic_bp - 132) / 17
z_prior = (prior_admissions - 1.3) / 1.1

log_odds = -0.65 + 0.45 * z_age + 0.30 * z_bmi + 0.55 * z_gluc + 0.35 * z_bp + 0.60 * z_prior
probabilities = 1 / (1 + np.exp(-log_odds))
readmitted_30d = (np.random.rand(n_samples) < probabilities).astype(int)

# Construct DataFrame with exactly 5 features and 1 response variable
df = pd.DataFrame({
    'age': age,
    'bmi': bmi,
    'fasting_glucose': fasting_glucose,
    'systolic_bp': systolic_bp,
    'prior_admissions': prior_admissions,
    'readmitted_30d': readmitted_30d
})

# Save to CSV file
csv_filename = 'healthcare_patient_readmission.csv'
df.to_csv(csv_filename, index=False)
print(f"Dataset successfully created and saved to '{csv_filename}'.")
print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nFirst 5 rows:")
print(df.head())
print("\nClass distribution:")
print(df['readmitted_30d'].value_counts())
