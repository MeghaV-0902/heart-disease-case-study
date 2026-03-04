---
layout: default
title: Heart Disease Prediction
---

<!-- HERO SECTION -->

<section class="hero">

<div class="hero-text">

<h1>Heart Disease Prediction</h1>

<h2>Machine Learning Case Study</h2>

<p class="hero-desc">
Predicting the presence of heart disease using clinical and demographic data using machine learning models.
</p>

<div class="hero-meta">
<span><strong>Tech:</strong> Python · Scikit-Learn · Plotly · Pandas · NumPy</span>
<span><strong>Dataset:</strong> 180 Patients · 15 Features</span>
</div>

</div>

<div class="hero-visual">
<img src="https://cdn-icons-png.flaticon.com/512/3774/3774299.png">
</div>

</section>

<!-- JUMP NAVIGATION -->

<div class="jump-box">
<strong>Jump to:</strong>
<a href="#overview">Overview</a>
<a href="#dataset">Dataset</a>
<a href="#eda">EDA</a>
<a href="#modeling">Modeling</a>
<a href="#comparison">Model Comparison</a>
<a href="#conclusion">Conclusion</a>
</div>

<hr>

---

## Overview
<span id="overview"></span>

This project develops and evaluates multiple machine learning models to predict the presence of heart disease using structured clinical data.

The objective is to support **early detection and preventive healthcare decision-making.**

---

## Objective

The goal of this project is to build a reliable classification model that predicts heart disease using clinical and demographic features.

The analysis includes:

- Exploratory Data Analysis
- Clinical range validation
- Data preprocessing
- Model development
- Model comparison
- Feature importance interpretation

---

## Dataset
<span id="dataset"></span>

- **180 observations**
- **15 columns**
- **14 input features**
- **18 features after one-hot encoding**

Target variable: **heart_disease_present**

Key variables include:

- age
- resting_blood_pressure
- serum_cholesterol_mg_per_dl
- max_heart_rate_achieved
- oldpeak_eq_st_depression
- num_major_vessels

---

# Exploratory Data Analysis
<span id="eda"></span>

<details id="edaSection">
<summary><strong>Click to Expand EDA</strong></summary>

### Univariate Analysis

<label><strong>Select Feature:</strong></label>

<select id="featureSelect">
<option value="age">age</option>
<option value="resting_blood_pressure">resting_blood_pressure</option>
<option value="serum_cholesterol_mg_per_dl">serum_cholesterol_mg_per_dl</option>
<option value="max_heart_rate_achieved">max_heart_rate_achieved</option>
<option value="oldpeak_eq_st_depression">oldpeak_eq_st_depression</option>
<option value="num_major_vessels">num_major_vessels</option>
</select>

<div class="chart-box">
<div id="univariateChart"></div>
</div>

---

### Bivariate Analysis

<label><strong>Select Feature:</strong></label>

<select id="bivariateSelect">
<option value="age">age</option>
<option value="resting_blood_pressure">resting_blood_pressure</option>
<option value="serum_cholesterol_mg_per_dl">serum_cholesterol_mg_per_dl</option>
<option value="max_heart_rate_achieved">max_heart_rate_achieved</option>
<option value="oldpeak_eq_st_depression">oldpeak_eq_st_depression</option>
<option value="num_major_vessels">num_major_vessels</option>
</select>

<div class="chart-box">
<div id="bivariateChart"></div>
</div>

---

### Correlation Heatmap

<div class="chart-box">
<div id="correlationChart"></div>
</div>

</details>

---

# Modeling
<span id="modeling"></span>

### Logistic Regression (Baseline)

Logistic Regression was used as the baseline model because of its interpretability and suitability for binary classification in healthcare datasets.

Results:

- Accuracy: **0.89**
- Precision: **0.80**
- Recall: **1.00**
- ROC-AUC: **0.96**

---

### Random Forest

Random Forest was trained to evaluate ensemble performance.

Results:

- Accuracy: **0.86**
- Precision: **0.79**
- Recall: **0.94**
- ROC-AUC: **0.93**

---

### XGBoost

Gradient boosting was also evaluated.

Results:

- Accuracy: **0.86**
- Precision: **0.79**
- Recall: **0.94**
- ROC-AUC: **0.93**

---

# Model Comparison
<span id="comparison"></span>

<label><strong>Select Metric:</strong></label>

<select id="metricSelect">
<option value="accuracy">Accuracy</option>
<option value="precision">Precision</option>
<option value="recall">Recall</option>
<option value="roc_auc">ROC-AUC</option>
</select>

<div class="chart-box">
<div id="modelChart"></div>
</div>

---

# Feature Importance

<div class="chart-box">
<div id="featureImportanceChart"></div>
</div>

Key influential predictors:

- num_major_vessels
- oldpeak_eq_st_depression
- chest pain types
- age

These variables demonstrated strong influence on prediction probability.

---

# Conclusion
<span id="conclusion"></span>

Logistic Regression emerged as the best performing model for this dataset.

Final performance:

- ROC-AUC: **0.96**
- Recall: **1.00**
- Accuracy: **0.89**

The results demonstrate that **simpler models can outperform complex ensemble models when the dataset size is small.**

---

<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>


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
