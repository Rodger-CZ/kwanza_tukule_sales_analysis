# Kwanza Tukule – Sales Data Analysis & Customer Segmentation

## 📌 Project Overview

This project presents an end-to-end analysis of anonymized sales transaction data, originally provided as part of a data analyst assessment. The objective is to derive actionable business insights from raw transactional data using data cleaning, exploratory analysis, customer segmentation, forecasting, and anomaly detection techniques.

The analysis demonstrates practical skills in data preparation, analytical reasoning, and business-oriented storytelling.

---

## 🎯 Objectives

* Assess and improve data quality
* Engineer time-based features for trend analysis
* Analyze sales performance by category, product, and customer
* Segment customers based on purchasing behavior
* Forecast short-term sales performance
* Detect unusual sales patterns (anomalies)
* Translate findings into strategic recommendations

---

## 🛠️ Tools & Technologies

* **Python**
* **Jupyter Notebook**
* pandas, numpy
* matplotlib, seaborn
* scikit-learn
* statsmodels

---

## 📂 Project Structure

```
kwanza-tukule-sales-analysis/
│
├── data/
│   └── case_study_data.csv
├── notebooks/
│   └── kwanza_tukule_sales_analysis.ipynb
├── README.md
└── requirements.txt
```

---

## 🔍 Key Analyses & Insights

### 1. Data Cleaning & Preparation

* Standardized column names for consistency
* Converted quantity and monetary fields from text to numeric
* Engineered a Month-Year feature for time-series analysis
* Removed invalid or duplicate records where necessary

### 2. Exploratory Data Analysis

* Sales value is concentrated in a small number of product categories
* A limited set of businesses contributes a disproportionate share of revenue
* Monthly sales trends reveal fluctuations indicative of seasonality or demand cycles
* High-volume products are not always the highest revenue generators

### 3. Customer Segmentation

Businesses were segmented into **High**, **Medium**, and **Low Value** groups based on:

* Total quantity purchased
* Total sales value
* Transaction frequency

**Key insight:**
High-value customers contribute the majority of revenue and should be prioritized for retention and loyalty initiatives.

### 4. Forecasting

* Used Exponential Smoothing (ETS) to forecast total sales value for the next three months
* Forecasts provide a short-term baseline for inventory and cash-flow planning

### 5. Anomaly Detection

* Identified unusual spikes and drops in monthly sales using z-score analysis
* Potential causes include bulk purchases, promotions, or supply disruptions

---

## 💡 Strategic Recommendations

* Prioritize high-performing product categories in marketing campaigns
* Introduce loyalty programs and preferential pricing for high-value customers
* Use forecast outputs to inform inventory planning
* Investigate anomaly periods to improve operational resilience

---

## 🚀 Future Improvements

* Incorporate external factors (e.g., pricing changes, promotions, economic indicators)
* Build an interactive dashboard (Plotly / Power BI)
* Scale analysis for larger datasets using optimized storage and processing

---

## 👤 Author

**Faustine Rodgers**
Data Analyst | Python | Business Analytics
