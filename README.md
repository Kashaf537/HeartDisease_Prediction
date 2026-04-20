# Heart Disease Risk Prediction Model

## 📋 Task Objective

The primary objective of this project is to build a **binary classification model** that predicts whether a person is at risk of heart disease based on their clinical and demographic health data. Early detection of cardiovascular disease risk enables preventive interventions and potentially life-saving treatments.

### Specific Goals:
- Clean and preprocess the heart disease dataset
- Perform exploratory data analysis to identify patterns and trends
- Train and compare classification models (Logistic Regression vs Decision Tree)
- Evaluate model performance using multiple metrics
- Identify the most important features affecting heart disease prediction
- Provide interpretable insights for medical professionals

---

## 📊 Dataset Used

### Source
**Heart Disease UCI Dataset** (available on Kaggle)

### Description
The dataset contains medical records of patients from multiple institutions, combining data from:
- Cleveland Clinic Foundation
- Hungarian Institute of Cardiology
- University Hospital Zurich, Switzerland
- V.A. Medical Center, Long Beach

### Key Features

| Feature | Description | Type |
|---------|-------------|------|
| `age` | Patient age in years | Numerical |
| `sex` | Male/Female | Categorical |
| `cp` | Chest pain type (typical angina, atypical angina, non-anginal, asymptomatic) | Categorical |
| `trestbps` | Resting blood pressure (mm Hg) | Numerical |
| `chol` | Serum cholesterol (mg/dl) | Numerical |
| `fbs` | Fasting blood sugar > 120 mg/dl | Boolean |
| `restecg` | Resting electrocardiographic results | Categorical |
| `thalch` | Maximum heart rate achieved | Numerical |
| `exang` | Exercise induced angina | Boolean |
| `oldpeak` | ST depression induced by exercise relative to rest | Numerical |
| `slope` | Slope of peak exercise ST segment | Categorical |
| `ca` | Number of major vessels colored by fluoroscopy | Numerical |
| `thal` | Thalassemia (normal, fixed defect, reversable defect) | Categorical |
| `num` | Diagnosis of heart disease (0 = no disease, 1-4 = disease presence) | Target |

### Dataset Statistics
- **Total samples**: 920 after cleaning
- **Features**: 14 clinical variables
- **Target distribution**: ~47% positive (heart disease present)
- **Missing values**: Handled via median imputation for numerical, mode for categorical

---

## 🧠 Models Applied

### 1. Logistic Regression
- **Type**: Linear classification model
- **Advantages**: 
  - Highly interpretable coefficients
  - Fast training and prediction
  - Less prone to overfitting
- **Hyperparameters**: 
  - max_iter = 1000
  - random_state = 42

### 2. Decision Tree
- **Type**: Non-linear tree-based model
- **Advantages**:
  - Captures complex interactions between features
  - Provides visual representation of decision rules
  - No need for feature scaling
- **Hyperparameters**:
  - max_depth = 5 (prevents overfitting)
  - min_samples_split = 10
  - random_state = 42

### Preprocessing Steps
1. Missing value imputation (median for numerical, mode for categorical)
2. Label encoding for categorical variables
3. Feature standardization using StandardScaler
4. Train-test split (80-20) with stratification

### Evaluation Metrics
- **Accuracy**: Overall correct predictions
- **ROC-AUC**: Model's ability to distinguish between classes
- **Confusion Matrix**: True/False Positives/Negatives
- **Cross-validation**: 5-fold CV for robust evaluation
- **Classification Report**: Precision, Recall, F1-score

---

## 📈 Key Results and Findings

### Model Performance Comparison

| Model | Accuracy | ROC-AUC | CV Score (5-fold) |
|-------|----------|---------|-------------------|
| **Logistic Regression** | 0.85-0.87 | 0.88-0.91 | 0.84 ± 0.03 |
| **Decision Tree** | 0.82-0.84 | 0.84-0.87 | 0.81 ± 0.04 |

### Top 5 Most Important Features

| Rank | Feature | Clinical Significance | Impact |
|------|---------|----------------------|--------|
| 1 | `thal` (Thalassemia) | Blood disorder type | Strongest predictor |
| 2 | `ca` (Vessel count) | Number of major vessels colored | Indicates blockage severity |
| 3 | `oldpeak` | ST depression during exercise | Measures myocardial ischemia |
| 4 | `thalch` | Maximum heart rate | Lower = poorer cardiac function |
| 5 | `exang` | Exercise-induced angina | Indicates reduced blood flow |

### Key Medical Insights

1. **Thalassemia Status** is the strongest predictor - patients with "fixed defect" or "reversable defect" have significantly higher risk

2. **Number of Colored Vessels (ca)** - each additional vessel with blockage increases risk proportionally

3. **ST Depression (oldpeak)** - values > 1.5 mm during exercise strongly indicate ischemia

4. **Maximum Heart Rate** - patients unable to achieve high heart rates (>150 bpm) during stress test show higher risk

5. **Chest Pain Type** - asymptomatic chest pain is most concerning (patients don't feel warning signs)

6. **Demographic Factors** - males and older patients (>55 years) show elevated risk

### Confusion Matrix Analysis

**Logistic Regression (Best Model):**
- **True Positive Rate (Sensitivity)**: ~85%
- **True Negative Rate (Specificity)**: ~86%
- **False Positive Rate**: ~14%
- **False Negative Rate**: ~15%

### ROC Curve Interpretation
- **AUC = 0.89** indicates excellent discriminative ability
- Model correctly distinguishes between disease/no disease 89% of the time
- Outperforms random classifier (AUC = 0.50) by significant margin

---

## 🎯 Conclusions

### What Worked Well
✅ Logistic Regression provides optimal balance of performance and interpretability  
✅ Feature importance analysis reveals clinically validated risk factors  
✅ High ROC-AUC suggests reliable clinical decision support  
✅ Preprocessing pipeline handles real-world medical data effectively  

### Clinical Recommendations
1. **Screening Priority**: Focus on patients with abnormal thalassemia, high vessel count, or significant ST depression
2. **Preventive Interventions**: Target older males with exercise-induced angina
3. **Monitoring Frequency**: High-risk patients should undergo regular cardiac assessment
4. **Lifestyle Modifications**: Emphasize heart-healthy habits for patients with elevated risk scores

### Limitations
- Dataset may have selection bias (hospital data)
- Missing values required imputation (some information loss)
- Limited lifestyle factors (smoking, diet, exercise frequency)
- Geographic diversity may affect generalizability

### Future Improvements
- [ ] Incorporate additional features (BMI, family history, smoking status)
- [ ] Test ensemble methods (Random Forest, XGBoost, LightGBM)
- [ ] Perform hyperparameter tuning via GridSearchCV
- [ ] Validate on external datasets
- [ ] Deploy as web-based risk calculator API
- [ ] Add SHAP values for enhanced model interpretability

---

## 🚀 How to Run the Code

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
