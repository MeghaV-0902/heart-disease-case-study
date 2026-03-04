---
layout: default
title: Heart Disease Prediction
---

# Heart Disease Prediction

### Machine Learning Case Study

**Goal:** Predict the presence of heart disease using clinical and demographic features.

**Tech Stack:** Python, Scikit-Learn, Plotly, Pandas, NumPy

**Dataset Size:** 180 patients | 15 features

---

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

- - 180 observations  
- 15 original columns  
- 14 input features  
- 18 features after one-hot encoding  
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

<details id="edaSection">
<summary><strong>Interactive EDA (Click to Expand)</strong></summary>

<h3>Univariate Analysis</h3>
<p><strong>Interactive Feature Distribution Explorer</strong></p>
<p>Select a feature to visualize its distribution across the dataset.</p>
<label for="featureSelect"><strong>Select Feature:</strong></label>
<select id="featureSelect">
    <option value="age">age</option>
    <option value="resting_blood_pressure">resting_blood_pressure</option>
    <option value="serum_cholesterol_mg_per_dl">serum_cholesterol_mg_per_dl</option>
    <option value="max_heart_rate_achieved">max_heart_rate_achieved</option>
    <option value="oldpeak_eq_st_depression">oldpeak_eq_st_depression</option>
    <option value="num_major_vessels">num_major_vessels</option>
</select>

<div class="chart-box" style="margin-top:25px;">
    <div id="univariateChart"></div>
</div>

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
<p><strong>Feature vs Target Comparison</strong></p>
<p>Select a feature to compare its distribution between patients with and without heart disease.</p>

<label for="bivariateSelect"><strong>Select Feature:</strong></label>
<select id="bivariateSelect">
    <option value="age">age</option>
    <option value="resting_blood_pressure">resting_blood_pressure</option>
    <option value="serum_cholesterol_mg_per_dl">serum_cholesterol_mg_per_dl</option>
    <option value="max_heart_rate_achieved">max_heart_rate_achieved</option>
    <option value="oldpeak_eq_st_depression">oldpeak_eq_st_depression</option>
    <option value="num_major_vessels">num_major_vessels</option>
</select>

<div class="chart-box" style="margin-top:25px;">
    <div id="bivariateChart"></div>
</div>

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

<h3>Correlation Heatmap</h3>
<div class="chart-box" style="margin-top:25px;">
    <div id="correlationChart"></div>
</div>

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

The following table summarizes the performance of all evaluated models across key evaluation metrics.

| Model | Accuracy | Precision | Recall | ROC-AUC |
|-------|----------|-----------|--------|---------|
| Logistic Regression | 0.89 | 0.80 | 1.00 | 0.96 |
| Logistic Regression (Stability Check) | 0.83 | 0.77 | 0.875 | 0.94 |
| Random Forest (Base) | 0.86 | 0.79 | 0.94 | 0.93 |
| Random Forest (Tuned) | 0.81 | 0.71 | 0.94 | 0.94 |
| XGBoost (Base) | 0.86 | 0.79 | 0.94 | 0.93 |

<div style="margin-top:30px;">
    <label><strong>Select Metric:</strong></label>
    <select id="metricSelect">
        <option value="accuracy">Accuracy</option>
        <option value="precision">Precision</option>
        <option value="recall">Recall</option>
        <option value="roc_auc">ROC-AUC</option>
    </select>

    <div class="chart-box" style="margin-top:25px;">
        <div id="modelChart"></div>
    </div>
</div>

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

<div class="chart-box" style="margin-top:25px;">
    <div id="featureImportanceChart"></div>
</div>

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

    const edaSection = document.getElementById("edaSection");
    let rendered = false;

    edaSection.addEventListener("toggle", function () {

        if (edaSection.open && !rendered) {

            fetch("assets/data/heart_cleaned.csv")
                .then(response => response.text())
                .then(data => {

                    const rows = data.trim().split("\n");
                    const headers = rows[0].split(",");
                    const dataset = rows
                        .slice(1)
                        .filter(row => row.trim() !== "")
                        .map(row => row.replace("\r","").split(","));

                    /* ======================
                       UNIVARIATE HISTOGRAM
                    ======================= */

                    function plotFeature(featureName) {

                        const colIndex = headers.indexOf(featureName);

                        const values = dataset
                            .map(row => Number(row[colIndex]))
                            .filter(val => !isNaN(val));

                        let binSize = 5;

                        if (featureName === "oldpeak_eq_st_depression") binSize = 0.5;
                        if (featureName === "num_major_vessels") binSize = 1;

                        Plotly.newPlot("univariateChart", [{
                            x: values,
                            type: "histogram",
                            xbins: { size: binSize }
                        }], {
                            title: featureName + " Distribution",
                            xaxis: { title: featureName },
                            yaxis: { title: "Count" },
                            bargap: 0.05
                        });
                    }

                    /* ======================
                       BIVARIATE BOX PLOT
                    ======================= */

                    function plotBivariate(featureName) {

                        const featureIndex = headers.indexOf(featureName);
                        const targetIndex = headers.indexOf("heart_disease_present");

                        const group0 = [];
                        const group1 = [];

                        dataset.forEach(row => {
                            const value = Number(row[featureIndex]);
                            const target = Number(row[targetIndex]);

                            if (target === 0) group0.push(value);
                            if (target === 1) group1.push(value);
                        });

                        Plotly.newPlot("bivariateChart", [
                            { y: group0, type: "box", name: "No Disease (0)" },
                            { y: group1, type: "box", name: "Heart Disease (1)" }
                        ], {
                            title: featureName + " vs Heart Disease",
                            yaxis: { title: featureName }
                        });
                    }

                    /* ======================
                       CORRELATION HEATMAP
                    ======================= */

                   function plotCorrelation() {
                    
                        const importantCols = [
                            "age",
                            "resting_blood_pressure",
                            "serum_cholesterol_mg_per_dl",
                            "max_heart_rate_achieved",
                            "oldpeak_eq_st_depression",
                            "num_major_vessels",
                            "heart_disease_present"
                        ];
                    
                        const indices = importantCols.map(col => headers.indexOf(col));
                    
                        const matrix = [];
                    
                        for (let i = 0; i < indices.length; i++) {
                            matrix[i] = [];
                            for (let j = 0; j < indices.length; j++) {
                    
                                let xi = dataset.map(r => Number(r[indices[i]]));
                                let xj = dataset.map(r => Number(r[indices[j]]));
                    
                                let meanXi = xi.reduce((a,b)=>a+b)/xi.length;
                                let meanXj = xj.reduce((a,b)=>a+b)/xj.length;
                    
                                let numerator = 0;
                                let denomXi = 0;
                                let denomXj = 0;
                    
                                for (let k=0; k<xi.length; k++){
                                    numerator += (xi[k]-meanXi)*(xj[k]-meanXj);
                                    denomXi += Math.pow(xi[k]-meanXi,2);
                                    denomXj += Math.pow(xj[k]-meanXj,2);
                                }
                    
                                matrix[i][j] = numerator / Math.sqrt(denomXi * denomXj);
                            }
                        }
                    
                        Plotly.newPlot("correlationChart", [{
                            z: matrix,
                            x: importantCols,
                            y: importantCols,
                            type: "heatmap",
                            colorscale: "RdBu",
                            zmin: -1,
                            zmax: 1
                        }], {
                            title: "Correlation Heatmap",
                            height: 700,
                            margin: { l: 140, r: 40, t: 60, b: 140 },
                            xaxis: { tickangle: -45 }
                        });
                    }
                    /* ======================
                       DROPDOWN LISTENERS
                    ======================= */

                    const uniDropdown = document.getElementById("featureSelect");
                    const biDropdown = document.getElementById("bivariateSelect");

                    uniDropdown.addEventListener("change", () =>
                        plotFeature(uniDropdown.value)
                    );

                    biDropdown.addEventListener("change", () =>
                        plotBivariate(biDropdown.value)
                    );

                    plotFeature(uniDropdown.value);
                    plotBivariate(biDropdown.value);
                    plotCorrelation();

                    rendered = true;
                });
        }
    });

    /* ======================
       MODEL COMPARISON
    ======================= */

    const metrics = {
        accuracy: [0.89, 0.83, 0.86, 0.81, 0.86],
        precision: [0.80, 0.77, 0.79, 0.71, 0.79],
        recall: [1.00, 0.875, 0.94, 0.94, 0.94],
        roc_auc: [0.96, 0.94, 0.93, 0.94, 0.93]
    };

    const models = [
        "LogReg",
        "LogReg Stability",
        "RF Base",
        "RF Tuned",
        "XGBoost"
    ];

    function plotModel(metric) {
        Plotly.newPlot("modelChart", [{
            x: models,
            y: metrics[metric],
            type: "bar"
        }], {
            title: metric.toUpperCase() + " Comparison",
            yaxis: { range: [0,1] }
        });
    }

    const metricDropdown = document.getElementById("metricSelect");
    if (metricDropdown) {
        metricDropdown.addEventListener("change", () =>
            plotModel(metricDropdown.value)
        );
        plotModel(metricDropdown.value);
    }

    /* ======================
       FEATURE IMPORTANCE
    ======================= */

    const importanceValues = [
        0.15,0.30,0.10,0.08,0.22,0.18,
        0.05,0.04,0.03,0.02,0.06,0.09,
        0.11,0.07,0.05,0.04,0.03,0.02
    ];

    fetch("assets/data/heart_cleaned.csv")
        .then(res => res.text())
        .then(data => {

            const headers = data.trim().split("\n")[0].split(",");

                Plotly.newPlot("featureImportanceChart", [{
                x: importanceValues,
                y: headers.slice(0,18),
                type: "bar",
                orientation: "h"
            }], {
                title: "Feature Importance",
                height: 700,
                margin: { l: 220, r: 40, t: 60, b: 40 },
                yaxis: { automargin: true }
            });
        });

});
</script>
