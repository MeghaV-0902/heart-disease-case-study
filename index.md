---
layout: default
title: Heart Disease Prediction
---

# Heart Disease Prediction

## Overview

This project develops and evaluates multiple machine learning models to predict the presence of heart disease using structured clinical data. The objective is to support early detection and improve preventive healthcare decision-making.

---

## Objective

The objective of this project is to build a reliable classification model that predicts the presence of heart disease using demographic and clinical features.

The analysis includes:

- Exploratory Data Analysis (EDA)
- Clinical range validation of medical variables
- Data preprocessing and feature engineering
- Model development using Logistic Regression, Random Forest, and XGBoost
- Stability validation
- Threshold tuning
- Model comparison using Accuracy, Precision, Recall, and ROC-AUC
- Feature importance interpretation
- Hospital-level recommendations

---

## Dataset

- 180 observations  
- 19 total columns  
- 18 input features  
- Target variable: `heart_disease_present`

The dataset includes demographic, clinical, and stress test-related measurements such as:

- age  
- resting_blood_pressure  
- serum_cholesterol_mg_per_dl  
- max_heart_rate_achieved  
- oldpeak_eq_st_depression  
- num_major_vessels  

The target variable `heart_disease_present` indicates whether the patient has heart disease.

---

## Clinical Range Validation

Before building any predictive models, medical variables were examined to ensure they fall within realistic physiological ranges. Validating clinical data is critical in healthcare projects, as unrealistic values can distort model learning and lead to misleading predictions.

A statistical summary was first generated:

```python
df.describe()
```

### Key Validations Included

- **Resting Blood Pressure:** Checked for extreme low or high values outside normal clinical ranges.  
- **Serum Cholesterol:** Examined for abnormal outliers that may indicate data entry errors.  
- **Maximum Heart Rate Achieved:** Compared against age to ensure physiological plausibility.  
- **ST Depression (`oldpeak_eq_st_depression`):** Validated to confirm values align with stress test expectations.  
- **Number of Major Vessels:** Confirmed categorical consistency and valid encoding.

### Observations

- No unrealistic physiological anomalies were detected.  
- No missing values were present in the dataset.  
- Feature distributions appeared consistent with expected medical ranges.

This validation step ensured that the dataset was clinically reliable before proceeding to modeling.

---

## Exploratory Data Analysis

<details>
<summary><strong>Click to Expand EDA</strong></summary>

<h3>Univariate Analysis</h3>
<div id="ageHistogram" style="margin-top:30px;"></div>

<p>Initial analysis focused on understanding the distribution of key numerical features.</p>

<pre><code class="language-python">
df.describe()
</code></pre>

<p><strong>Key Observations:</strong></p>
<ul>
<li>Age distribution centered around middle-aged individuals.</li>
<li>Cholesterol showed mild right skew.</li>
<li>ST depression values were concentrated near lower ranges.</li>
<li>Number of major vessels showed separation patterns relevant to disease presence.</li>
</ul>

<hr>

<h3>Bivariate Analysis</h3>

<p>The relationship between features and the target variable <code>heart_disease_present</code> was analyzed.</p>

<pre><code class="language-python">
df.groupby("heart_disease_present").mean()
</code></pre>

<ul>
<li>Higher values of <code>num_major_vessels</code> were strongly associated with heart disease.</li>
<li>Exercise-induced angina showed a clear relationship with positive cases.</li>
<li>ST depression demonstrated predictive behavior.</li>
<li>Maximum heart rate tended to be lower among patients with heart disease.</li>
</ul>

<hr>

<h3>Correlation Analysis</h3>

<pre><code class="language-python">
df.corr()
</code></pre>

<ul>
<li><code>num_major_vessels</code> showed strong positive correlation with the target.</li>
<li>ST depression had moderate correlation.</li>
<li>Most features showed low multicollinearity, reducing risk of instability in Logistic Regression.</li>
</ul>

</details>

---

## Data Preprocessing

Before training the models, the dataset was prepared to ensure compatibility with machine learning algorithms.

### Steps Performed

- Verified absence of missing values  
- Confirmed correct encoding of categorical variables  
- Separated features and target variable  
- Performed train-test split  

```python
from sklearn.model_selection import train_test_split

X = df.drop("heart_disease_present", axis=1)
y = df["heart_disease_present"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

The dataset was split into training and testing sets to evaluate generalization performance.

No scaling was required for tree-based models, while Logistic Regression handled the feature distributions appropriately for this dataset.

---

## Modeling

### Logistic Regression (Baseline)

Logistic Regression was selected as the baseline model due to its interpretability and suitability for binary classification problems in healthcare.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
```

### Results

- **Accuracy:** 0.89  
- **Precision:** 0.80  
- **Recall:** 1.00  
- **ROC-AUC:** 0.96  

### Interpretation

The model achieved perfect recall, meaning all positive heart disease cases in the test set were correctly identified.

In medical applications, recall is especially important because missing a true positive case can have serious consequences.

The high ROC-AUC score (0.96) indicates strong discriminative ability between positive and negative classes.

---

### Logistic Regression (Stability Check)

To verify robustness, the model was retrained using a different random state during train-test split.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=7
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
```

### Results

- **Accuracy:** 0.83  
- **Precision:** 0.77  
- **Recall:** 0.875  
- **ROC-AUC:** 0.94  

### Interpretation

Performance remained strong across different data splits.  
Although metrics slightly decreased, ROC-AUC remained high, indicating the model generalizes reasonably well.

This confirms that the baseline model performance was not due to a favorable train-test split.

---

### Threshold Tuning

By default, Logistic Regression classifies probabilities above 0.5 as positive. However, in healthcare applications, adjusting the decision threshold can improve recall while maintaining acceptable precision.

Instead of relying on the default threshold, predicted probabilities were analyzed:

```python
import numpy as np

threshold = 0.4
y_pred_adjusted = (y_prob >= threshold).astype(int)
```

Different thresholds were evaluated to balance recall and precision.

### Why Threshold Tuning Matters

- A lower threshold increases recall (fewer missed disease cases).  
- A higher threshold increases precision (fewer false alarms).  
- In medical contexts, recall is often prioritized to reduce the risk of missing high-risk patients.

The analysis confirmed that Logistic Regression maintains strong performance even when adjusting classification thresholds, reinforcing its suitability for clinical screening applications.

---

### Random Forest (Base Model)

A Random Forest classifier was trained to compare ensemble performance against Logistic Regression.

```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]
```

### Results

- **Accuracy:** 0.86  
- **Precision:** 0.79  
- **Recall:** 0.94  
- **ROC-AUC:** 0.93  

### Interpretation

Random Forest performed well, achieving high recall and strong ROC-AUC.  

However, it did not outperform Logistic Regression in overall discriminative ability.  

While ensemble models can capture nonlinear patterns, in this dataset the simpler linear model generalized better.

---

### Random Forest (Tuned Model)

Hyperparameters were adjusted to improve model performance and reduce potential overfitting.

```python
rf_tuned = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    random_state=42
)

rf_tuned.fit(X_train, y_train)

y_pred = rf_tuned.predict(X_test)
y_prob = rf_tuned.predict_proba(X_test)[:, 1]
```

### Results

- **Accuracy:** 0.81  
- **Precision:** 0.71  
- **Recall:** 0.94  
- **ROC-AUC:** 0.94  

### Interpretation

Tuning did not significantly improve performance compared to the base Random Forest model.  

In fact, overall accuracy and precision decreased slightly, while ROC-AUC remained similar.

This reinforces an important insight: increasing model complexity does not always lead to better performance, especially with smaller datasets.

---

### XGBoost (Base Model)

An XGBoost classifier was trained to evaluate gradient boosting performance on the dataset.

```python
from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)

xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)
y_prob = xgb_model.predict_proba(X_test)[:, 1]
```

### Results

- **Accuracy:** 0.86  
- **Precision:** 0.79  
- **Recall:** 0.94  
- **ROC-AUC:** 0.93  

### Interpretation

XGBoost achieved performance comparable to Random Forest.  

However, it did not exceed the ROC-AUC achieved by Logistic Regression.  

Given the small dataset size, the added complexity of boosting did not provide significant gains over the simpler baseline model.

---

## Model Comparison

| Model | Accuracy | Precision | Recall | ROC-AUC |
|-------|----------|-----------|--------|---------|
| Logistic Regression | 0.89 | 0.80 | 1.00 | 0.96 |
| Logistic Regression (Stability Check) | 0.83 | 0.77 | 0.875 | 0.94 |
| Random Forest (Base) | 0.86 | 0.79 | 0.94 | 0.93 |
| Random Forest (Tuned) | 0.81 | 0.71 | 0.94 | 0.94 |
| XGBoost (Base) | 0.86 | 0.79 | 0.94 | 0.93 |

---

### Final Model Selection

Logistic Regression was selected as the final model based on the following:

- Highest **ROC-AUC (0.96)**  
- Perfect **Recall (1.00)**  
- Strong generalization across different random states  
- Simplicity and interpretability  

In healthcare applications, recall is critical because missing a positive heart disease case can have serious consequences.

Although ensemble models performed well, they did not surpass Logistic Regression in overall discriminative ability. Additionally, Logistic Regression provides greater interpretability, which is valuable in clinical decision-making environments.

---

## Feature Importance

Feature importance was analyzed using Logistic Regression coefficients to understand which variables contributed most to prediction.

```python
importance_df = pd.DataFrame({
    "feature": X.columns,
    "coefficient": model.coef_[0]
}).sort_values(by="coefficient", ascending=False)

importance_df
```

### Key Insights

- **num_major_vessels** emerged as one of the strongest predictors.  
- **oldpeak_eq_st_depression** showed significant positive influence.  
- Chest pain type indicators contributed meaningfully to prediction.  
- Age had moderate impact.  

Using Logistic Regression allowed direct interpretation of coefficients, making the model suitable for healthcare settings where explainability is essential.

Unlike black-box ensemble models, coefficient-based interpretation provides transparency in understanding patient risk factors.

---

## Hospital Recommendations

Based on the model findings, the following recommendations can support clinical decision-making:

- Patients with higher **num_major_vessels** should be prioritized for further cardiac evaluation.  
- Individuals showing **exercise-induced angina** should receive closer monitoring.  
- Elevated **ST depression** values during stress testing should be flagged for risk assessment.  
- The model can be deployed as a **screening support tool**, assisting physicians in identifying high-risk patients earlier.

It is important to emphasize that this model is not intended to replace clinical diagnosis, but rather to augment early detection efforts and improve preventive care strategies.

---

## Challenges Faced

- The dataset contained only **180 observations**, limiting model generalization.  
- Small sample size increased the risk of overfitting in complex ensemble models.  
- Balancing recall and precision required careful threshold tuning.  
- Limited feature diversity restricted the ability to capture broader cardiovascular risk factors.

Despite these limitations, the project demonstrates how structured modeling and careful evaluation can produce meaningful predictive insights even with constrained data.

---

## Conclusion

This project demonstrates that a well-structured Logistic Regression model can effectively predict heart disease using structured clinical data.

Despite evaluating more complex ensemble models such as Random Forest and XGBoost, the simpler linear model achieved the strongest overall performance.

Final Model Performance:

- **ROC-AUC:** 0.96  
- **Recall:** 1.00  
- **Accuracy:** 0.89  

The results highlight an important principle in machine learning: model complexity does not guarantee better performance.

In this healthcare context, Logistic Regression provided:

- High recall for patient safety  
- Strong discriminative ability  
- Interpretability suitable for clinical environments  

The project reinforces the value of structured analysis, model validation, and domain-aware decision-making in medical predictive modeling.

<script>
document.addEventListener("DOMContentLoaded", function () {

    fetch("assets/data/heart_cleaned.csv")
        .then(response => response.text())
        .then(data => {
            const rows = data.split("\n").slice(1);
            const ageIndex = 6; // age column index

            const ages = rows
                .map(row => row.split(",")[ageIndex])
                .filter(val => val !== "")
                .map(Number);

            var trace = {
    x: ages,
    type: "histogram",
    xbins: {
        size: 5 
    },
    marker: { color: "#3b82f6" }
};

            var layout = {
    title: "Age Distribution",
    xaxis: { 
        title: "Age",
        dtick: 5
    },
    yaxis: { 
        title: "Count"
    },
    bargap: 0.05
};

            Plotly.newPlot("ageHistogram", [trace], layout);
        });

});
</script>
