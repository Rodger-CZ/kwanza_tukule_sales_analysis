# %% [code] {"execution":{"iopub.status.busy":"2026-01-18T03:00:11.414044Z","iopub.execute_input":"2026-01-18T03:00:11.414658Z","iopub.status.idle":"2026-01-18T03:00:11.430946Z","shell.execute_reply.started":"2026-01-18T03:00:11.414598Z","shell.execute_reply":"2026-01-18T03:00:11.429332Z"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [markdown]
# ## 📊 Sales Analysis & Customer Segmentation Case Study
# 
# This notebook presents an end-to-end analysis of anonymized sales transaction data, focusing on uncovering trends, customer behavior, and actionable business insights.
# 
# ### What’s covered in this notebook:
# 
# * Data cleaning and preparation of real-world transactional data
# * Feature engineering for monthly time-series analysis
# * Exploratory analysis of sales by category, product, and customer
# * Customer segmentation based on purchasing behavior
# * Short-term sales forecasting using exponential smoothing
# * Detection of unusual sales spikes and drops (anomalies)
# * Business-focused interpretations and strategic recommendations
# 
# ### Why this analysis matters:
# 
# The project mirrors real commercial analytics workflows, where data quality issues, customer concentration, and demand volatility must be addressed to support decision-making. The insights generated can inform marketing prioritization, customer retention strategies, and operational planning.
# 
# This notebook is designed to demonstrate practical data analytics skills, clear reasoning, and the ability to translate data into business value.
# 

# %% [markdown]
# # Kwanza Tukule – Sales Data Analysis & Customer Segmentation
# 
# ## Project Overview
# This project analyzes anonymized sales transaction data provided as part of a
# data analyst assessment. The objective is to explore sales performance,
# customer purchasing behavior, and product trends, and to derive actionable
# business insights that can support strategic decision-making.
# 
# ## Key Objectives
# - Assess data quality and perform necessary cleaning
# - Engineer time-based features for trend analysis
# - Analyze sales performance by category, product, and business
# - Segment customers based on purchasing behavior
# - Provide data-driven recommendations
# 
# ## Tools & Libraries
# - Python
# - pandas, numpy
# - matplotlib, seaborn
# - scikit-learn
# 

# %% [code] {"execution":{"iopub.status.busy":"2026-01-18T03:00:11.433133Z","iopub.execute_input":"2026-01-18T03:00:11.433605Z","iopub.status.idle":"2026-01-18T03:00:11.441837Z","shell.execute_reply.started":"2026-01-18T03:00:11.433534Z","shell.execute_reply":"2026-01-18T03:00:11.440594Z"}}
# We import the required packages
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

plt.style.use("seaborn-v0_8")

# %% [code] {"execution":{"iopub.status.busy":"2026-01-18T03:00:11.443136Z","iopub.execute_input":"2026-01-18T03:00:11.443913Z","iopub.status.idle":"2026-01-18T03:00:12.055291Z","shell.execute_reply.started":"2026-01-18T03:00:11.443876Z","shell.execute_reply":"2026-01-18T03:00:12.054511Z"}}
# We proceed to load the Data
df = pd.read_csv('/kaggle/input/kwanza-tukule-dataset/Case Study Data - Read Only - case_study_data_2025-01-16T06_49_12.19881Z.csv')

# We visualize the first rows of the dataset to have an overview
df.head()

# %% [code] {"execution":{"iopub.status.busy":"2026-01-18T03:00:12.057230Z","iopub.execute_input":"2026-01-18T03:00:12.057525Z","iopub.status.idle":"2026-01-18T03:00:12.062604Z","shell.execute_reply.started":"2026-01-18T03:00:12.057496Z","shell.execute_reply":"2026-01-18T03:00:12.061913Z"}}
# We view all the columns
for col in df.columns:
    print(f"'{col}'")

# %% [code] {"execution":{"iopub.status.busy":"2026-01-18T03:00:12.063702Z","iopub.execute_input":"2026-01-18T03:00:12.064048Z","iopub.status.idle":"2026-01-18T03:00:12.082235Z","shell.execute_reply.started":"2026-01-18T03:00:12.064013Z","shell.execute_reply":"2026-01-18T03:00:12.081344Z"}}
# We start by cleaning the columns for Whitespaces which is a very common issue
# Clean column names
df.columns = (
    df.columns
      .str.strip()          # remove leading/trailing spaces
      .str.replace(" ", "_")
      .str.lower()
)

df.columns

# %% [code] {"execution":{"iopub.status.busy":"2026-01-18T03:00:12.083448Z","iopub.execute_input":"2026-01-18T03:00:12.084097Z","iopub.status.idle":"2026-01-18T03:00:12.214097Z","shell.execute_reply.started":"2026-01-18T03:00:12.084057Z","shell.execute_reply":"2026-01-18T03:00:12.212490Z"}}
# We use the info to gather more details on our dataset
df.info()

# %% [code] {"execution":{"iopub.status.busy":"2026-01-18T03:00:12.215863Z","iopub.execute_input":"2026-01-18T03:00:12.216227Z","iopub.status.idle":"2026-01-18T03:00:12.577747Z","shell.execute_reply.started":"2026-01-18T03:00:12.216199Z","shell.execute_reply":"2026-01-18T03:00:12.576851Z"}}
# We use describe to gather descriptive information about our dataset
df.describe(include="all")

# %% [markdown]
# ## 1. Data Cleaning and Preparation
# 
# ### 1.1 Data Quality Assessment
# The dataset was inspected for missing values, duplicate records, and
# inconsistent data types to ensure the reliability of the analysis.
# 

# %% [code] {"execution":{"iopub.status.busy":"2026-01-18T03:00:12.578941Z","iopub.execute_input":"2026-01-18T03:00:12.579602Z","iopub.status.idle":"2026-01-18T03:00:12.691718Z","shell.execute_reply.started":"2026-01-18T03:00:12.579535Z","shell.execute_reply":"2026-01-18T03:00:12.690649Z"}}
# We will begin by checking for Null values in the dataset
df.isna().sum()

# %% [code] {"execution":{"iopub.status.busy":"2026-01-18T03:00:12.692962Z","iopub.execute_input":"2026-01-18T03:00:12.693249Z","iopub.status.idle":"2026-01-18T03:00:12.834268Z","shell.execute_reply.started":"2026-01-18T03:00:12.693216Z","shell.execute_reply":"2026-01-18T03:00:12.833374Z"}}
# We drop all the null values before proceeding to duplicates
df = df.dropna()

# %% [markdown]
# From the above findings, we do have 8 null values in our Dataset
# 
# The null values are in the field "UNIT Price"

# %% [code] {"execution":{"iopub.status.busy":"2026-01-18T03:00:12.837577Z","iopub.execute_input":"2026-01-18T03:00:12.838255Z","iopub.status.idle":"2026-01-18T03:00:13.049387Z","shell.execute_reply.started":"2026-01-18T03:00:12.838221Z","shell.execute_reply":"2026-01-18T03:00:13.048486Z"}}
# We proceed to check for duplicates in the Dataset
df.duplicated().sum()

# %% [code] {"execution":{"iopub.status.busy":"2026-01-18T03:00:13.050571Z","iopub.execute_input":"2026-01-18T03:00:13.050876Z","iopub.status.idle":"2026-01-18T03:00:13.057989Z","shell.execute_reply.started":"2026-01-18T03:00:13.050849Z","shell.execute_reply":"2026-01-18T03:00:13.057042Z"}}
# We check for the Datatypes 
df.dtypes

# %% [markdown]
# The DATE field is considered an Object instead of Datetime
# 
# UNIT PRICE is also read as an object instead of currency

# %% [code] {"execution":{"iopub.status.busy":"2026-01-18T03:00:13.059266Z","iopub.execute_input":"2026-01-18T03:00:13.059737Z","iopub.status.idle":"2026-01-18T03:00:22.724649Z","shell.execute_reply.started":"2026-01-18T03:00:13.059686Z","shell.execute_reply":"2026-01-18T03:00:22.723789Z"}}
# Convert date column to datetime
df["date"] = pd.to_datetime(df["date"])

# Convert quantity to integer
df['quantity']=df['quantity'].astype('int')

# Convert unit_price to integer
df['unit_price']=df['unit_price'].str.replace(',','')
df['unit_price']=df['unit_price'].astype('int')

# Remove duplicates if any
df = df.drop_duplicates()


df.head()
df.info()

# %% [markdown]
# ### 1.2 Feature Engineering
# A `Month-Year` feature was created from the transaction date to support
# time-series and seasonal trend analysis.
# 

# %% [code] {"execution":{"iopub.status.busy":"2026-01-18T03:00:22.726052Z","iopub.execute_input":"2026-01-18T03:00:22.726306Z","iopub.status.idle":"2026-01-18T03:00:22.953265Z","shell.execute_reply.started":"2026-01-18T03:00:22.726281Z","shell.execute_reply":"2026-01-18T03:00:22.952503Z"}}
# We create a new field "Month-Year" to support time-series, it will also help us in trend analysis
df["month_year"] = df["date"].dt.to_period("M").astype(str)

# We visualize the "DATE" and "Month-Year" fields
df[["date", "month_year"]].head()

# %% [markdown]
# ## 2. Exploratory Data Analysis
# 
# ### 2.1 Sales Overview
# This section analyzes total sales quantity and value across product categories
# and businesses to understand where revenue and volume are concentrated.
# 

# %% [code] {"execution":{"iopub.status.busy":"2026-01-18T03:00:22.954356Z","iopub.execute_input":"2026-01-18T03:00:22.954733Z","iopub.status.idle":"2026-01-18T03:00:22.993570Z","shell.execute_reply.started":"2026-01-18T03:00:22.954693Z","shell.execute_reply":"2026-01-18T03:00:22.992686Z"}}
# We will categorize the unit_price and quantity based on the category 
category_summary = (
    df.groupby("anonymized_category")[["quantity", "unit_price"]]
      .sum()
      .sort_values("unit_price", ascending=False)
)

# We visualize the first 10 records
category_summary.head(10)

# %% [code] {"execution":{"iopub.status.busy":"2026-01-18T03:00:22.994661Z","iopub.execute_input":"2026-01-18T03:00:22.995044Z","iopub.status.idle":"2026-01-18T03:00:23.501921Z","shell.execute_reply.started":"2026-01-18T03:00:22.995016Z","shell.execute_reply":"2026-01-18T03:00:23.500981Z"}}
# We are going to plot a bar chart based on the category that we extracted above
plt.figure(figsize=(10,5))
sns.barplot(
    x=category_summary.index,
    y=category_summary["unit_price"]
)
plt.title("Total Sales Value by Product Category")
plt.xticks(rotation=45)
plt.ylabel("Total Sales Value")
plt.xlabel("Product Category")
plt.tight_layout()

# We visualize the bar chart
plt.show()

# %% [markdown]
# **Insight:**
# Sales value is concentrated in a limited number of categories, indicating
# opportunities for focused marketing, inventory prioritization, and supplier
# negotiation within high-performing categories.
# 

# %% [code] {"execution":{"iopub.status.busy":"2026-01-18T03:00:23.503057Z","iopub.execute_input":"2026-01-18T03:00:23.503393Z","iopub.status.idle":"2026-01-18T03:00:23.548534Z","shell.execute_reply.started":"2026-01-18T03:00:23.503365Z","shell.execute_reply":"2026-01-18T03:00:23.547774Z"}}
# Next, we group unit_price and quantity based on the Business type
business_summary = (
    df.groupby("anonymized_business")[["quantity", "unit_price"]]
      .sum()
      .sort_values("unit_price", ascending=False)
)

#We visualize the first 10 records
business_summary.head(10)


# %% [markdown]
# **Insight:**
# A small number of businesses account for a disproportionate share of total
# sales value, suggesting the presence of high-value customers critical to
# revenue stability.
# 

# %% [markdown]
# ### 2.2 Sales Trends Over Time
# Monthly trends were analyzed to identify growth patterns, seasonality, and
# volatility in sales performance.
# 

# %% [code] {"execution":{"iopub.status.busy":"2026-01-18T03:00:23.549908Z","iopub.execute_input":"2026-01-18T03:00:23.550273Z","iopub.status.idle":"2026-01-18T03:00:23.586828Z","shell.execute_reply.started":"2026-01-18T03:00:23.550234Z","shell.execute_reply":"2026-01-18T03:00:23.585900Z"}}
# We extract the monthly sales pattern to analyze monthly performance
monthly_sales = (
    df.groupby("month_year")[["quantity", "unit_price"]]
      .sum()
      .reset_index()
      .sort_values("month_year")
)

monthly_sales

# %% [code] {"execution":{"iopub.status.busy":"2026-01-18T03:00:23.587949Z","iopub.execute_input":"2026-01-18T03:00:23.588304Z","iopub.status.idle":"2026-01-18T03:00:23.799075Z","shell.execute_reply.started":"2026-01-18T03:00:23.588265Z","shell.execute_reply":"2026-01-18T03:00:23.798319Z"}}
# We plot a time series chart to view the fluctuation in monthly sales performance
plt.figure(figsize=(12,5))
plt.plot(monthly_sales["month_year"], monthly_sales["unit_price"], marker="o")
plt.title("Monthly Sales Value Trend")
plt.xlabel("month_year")
plt.ylabel("Total Sales Value")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Insight:**
# The time series reveals fluctuations in monthly sales performance, which may
# indicate seasonal demand patterns, changes in customer purchasing behavior,
# or operational factors such as supply availability.
# 

# %% [markdown]
# ### 2.3 Product Performance Analysis
# Products were evaluated based on total quantity sold and total sales value to
# distinguish high-volume items from high-revenue drivers.
# 

# %% [code] {"execution":{"iopub.status.busy":"2026-01-18T03:00:23.800333Z","iopub.execute_input":"2026-01-18T03:00:23.800698Z","iopub.status.idle":"2026-01-18T03:00:23.833665Z","shell.execute_reply.started":"2026-01-18T03:00:23.800669Z","shell.execute_reply":"2026-01-18T03:00:23.832926Z"}}
# We are analyzing the product with the highest demand in terms of quantity
top_qty_products = (
    df.groupby("anonymized_product")["quantity"]
      .sum()
      .sort_values(ascending=False)
      .head(5)
)

top_qty_products

# %% [code] {"execution":{"iopub.status.busy":"2026-01-18T03:00:23.834863Z","iopub.execute_input":"2026-01-18T03:00:23.835198Z","iopub.status.idle":"2026-01-18T03:00:23.869260Z","shell.execute_reply.started":"2026-01-18T03:00:23.835160Z","shell.execute_reply":"2026-01-18T03:00:23.868340Z"}}
# We are analyzing the products generating the highest value in terms of sales & pricing
top_value_products = (
    df.groupby("anonymized_product")["unit_price"]
      .sum()
      .sort_values(ascending=False)
      .head(5)
)

top_value_products

# %% [markdown]
# **Insight:**
# The most frequently purchased products are not always the most valuable.
# High-value products may have lower volumes but contribute more significantly
# to revenue, highlighting opportunities for pricing and bundling strategies.
# 

# %% [markdown]
# ## 3. Advanced Analysis
# 
# ### 3.1 Customer Segmentation
# Businesses were segmented based on purchasing behavior using:
# - Total Quantity purchased
# - Total Sales Value
# - Transaction frequency
# 
# This segmentation helps tailor engagement and retention strategies.
# 

# %% [code] {"execution":{"iopub.status.busy":"2026-01-18T03:00:23.870655Z","iopub.execute_input":"2026-01-18T03:00:23.870930Z","iopub.status.idle":"2026-01-18T03:00:23.920050Z","shell.execute_reply.started":"2026-01-18T03:00:23.870903Z","shell.execute_reply":"2026-01-18T03:00:23.919196Z"}}
# We build the customer metrics - Here we will identify the type of business and an aggregate of the value and transaction frequency
customer_metrics = (
    df.groupby("anonymized_business")
      .agg(
          total_quantity=("quantity", "sum"),
          total_value=("unit_price", "sum"),
          transaction_count=("date", "count")
      )
)

customer_metrics.head()


# %% [code] {"execution":{"iopub.status.busy":"2026-01-18T03:00:23.920915Z","iopub.execute_input":"2026-01-18T03:00:23.921201Z","iopub.status.idle":"2026-01-18T03:00:23.939884Z","shell.execute_reply.started":"2026-01-18T03:00:23.921174Z","shell.execute_reply":"2026-01-18T03:00:23.938569Z"}}
# We standardize the features using the standard scaler 
scaler = StandardScaler()
scaled_features = scaler.fit_transform(customer_metrics)

scaled_df = pd.DataFrame(
    scaled_features,
    index=customer_metrics.index,
    columns=customer_metrics.columns
)

scaled_df.head()

# %% [code] {"execution":{"iopub.status.busy":"2026-01-18T03:00:23.941294Z","iopub.execute_input":"2026-01-18T03:00:23.941652Z","iopub.status.idle":"2026-01-18T03:00:23.966400Z","shell.execute_reply.started":"2026-01-18T03:00:23.941613Z","shell.execute_reply":"2026-01-18T03:00:23.965635Z"}}
# We proceed to K-Means Clustering to fit the data
kmeans = KMeans(n_clusters=3, random_state=42)
customer_metrics["Segment"] = kmeans.fit_predict(scaled_df)
customer_metrics.head()

# %% [code] {"execution":{"iopub.status.busy":"2026-01-18T03:00:23.967489Z","iopub.execute_input":"2026-01-18T03:00:23.967776Z","iopub.status.idle":"2026-01-18T03:00:23.981500Z","shell.execute_reply.started":"2026-01-18T03:00:23.967750Z","shell.execute_reply":"2026-01-18T03:00:23.980434Z"}}
# We map the segments to the labels
segment_summary = (
    customer_metrics.groupby("Segment")
    .mean()
    .sort_values("total_value", ascending=False)
)

segment_summary

# %% [code] {"execution":{"iopub.status.busy":"2026-01-18T03:00:23.982791Z","iopub.execute_input":"2026-01-18T03:00:23.983161Z","iopub.status.idle":"2026-01-18T03:00:23.996275Z","shell.execute_reply.started":"2026-01-18T03:00:23.983134Z","shell.execute_reply":"2026-01-18T03:00:23.995322Z"}}
# We proceed to label the segments
segment_mapping = {
    segment_summary.index[0]: "High Value",
    segment_summary.index[1]: "Medium Value",
    segment_summary.index[2]: "Low Value"
}

customer_metrics["Segment Label"] = customer_metrics["Segment"].map(segment_mapping)
customer_metrics.head()


# %% [markdown]
# **Segmentation Insights & Recommendations**
# 
# - **High Value Customers**  
#   Contribute the majority of revenue. Prioritize with loyalty programs,
#   preferential pricing, and dedicated account management.
# 
# - **Medium Value Customers**  
#   Show growth potential. Encourage increased purchase frequency through
#   promotions and bundled offers.
# 
# - **Low Value Customers**  
#   Represent low engagement or sporadic purchasing. Use targeted reactivation
#   campaigns or cost-efficient communication strategies.
# 

# %% [markdown]
# **SECTION 3.2 - Forecasting Sales (Value)**
# 
# We'll forecast total monthly sales value for the next 3 months

# %% [code] {"execution":{"iopub.status.busy":"2026-01-18T03:00:23.997607Z","iopub.execute_input":"2026-01-18T03:00:23.998028Z","iopub.status.idle":"2026-01-18T03:00:24.042387Z","shell.execute_reply.started":"2026-01-18T03:00:23.997998Z","shell.execute_reply":"2026-01-18T03:00:24.041533Z"}}
# Preparing time series
ts = (
    df.groupby("month_year")["unit_price"]
      .sum()
      .reset_index()
)

ts["month_year"] = pd.to_datetime(ts["month_year"])
ts = ts.sort_values("month_year")
ts.set_index("month_year", inplace=True)

ts


# %% [code] {"execution":{"iopub.status.busy":"2026-01-18T03:00:24.043592Z","iopub.execute_input":"2026-01-18T03:00:24.043917Z","iopub.status.idle":"2026-01-18T03:00:24.257791Z","shell.execute_reply.started":"2026-01-18T03:00:24.043878Z","shell.execute_reply":"2026-01-18T03:00:24.257066Z"}}
# We plot a time series to view the historical series
plt.figure(figsize=(12,5))
plt.plot(ts.index, ts["unit_price"])
plt.title("Historical Monthly Sales Value")
plt.xlabel("Month")
plt.ylabel("Sales Value")
plt.tight_layout()
plt.show()
# This plot helps us to justify the forecasting device

# %% [markdown]
# **FORECASTING USING EXPONENTIAL SMOOTHING (ETS)**

# %% [code] {"execution":{"iopub.status.busy":"2026-01-18T03:00:24.261038Z","iopub.execute_input":"2026-01-18T03:00:24.261347Z","iopub.status.idle":"2026-01-18T03:00:24.300944Z","shell.execute_reply.started":"2026-01-18T03:00:24.261319Z","shell.execute_reply":"2026-01-18T03:00:24.300233Z"}}
from statsmodels.tsa.holtwinters import ExponentialSmoothing

model = ExponentialSmoothing(
    ts["unit_price"],
    trend="add",
    seasonal=None
)

fit = model.fit()
forecast = fit.forecast(3)

forecast

# %% [code] {"execution":{"iopub.status.busy":"2026-01-18T03:00:24.301770Z","iopub.execute_input":"2026-01-18T03:00:24.302042Z","iopub.status.idle":"2026-01-18T03:00:24.577246Z","shell.execute_reply.started":"2026-01-18T03:00:24.302012Z","shell.execute_reply":"2026-01-18T03:00:24.576286Z"}}
plt.figure(figsize=(12,5))
plt.plot(ts.index, ts["unit_price"], label="Historical")
plt.plot(forecast.index, forecast.values, label="Forecast", marker="o")
plt.title("3-Month Sales Forecast")
plt.xlabel("Month")
plt.ylabel("Sales Value")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3.2 Sales Forecasting
# Using an exponential smoothing model, total sales value was forecasted for the
# next three months. The forecast suggests a continuation of recent sales
# patterns, providing a baseline expectation for short-term planning.
# 
# These projections can support inventory planning, cash flow management,
# and marketing campaign timing.
# 

# %% [markdown]
# ****ANOMALY DETECTION ****
# 
# Here we will identify unusual spikes or drops, exactly as required

# %% [code] {"execution":{"iopub.status.busy":"2026-01-18T03:00:24.578431Z","iopub.execute_input":"2026-01-18T03:00:24.578815Z","iopub.status.idle":"2026-01-18T03:00:24.588511Z","shell.execute_reply.started":"2026-01-18T03:00:24.578777Z","shell.execute_reply":"2026-01-18T03:00:24.587690Z"}}
ts["z_score"] = (
    (ts["unit_price"] - ts["unit_price"].mean()) / ts["unit_price"].std()
)

anomalies = ts[ts["z_score"].abs() > 2]
anomalies

# %% [code] {"execution":{"iopub.status.busy":"2026-01-18T03:00:24.589608Z","iopub.execute_input":"2026-01-18T03:00:24.590329Z","iopub.status.idle":"2026-01-18T03:00:24.852315Z","shell.execute_reply.started":"2026-01-18T03:00:24.590302Z","shell.execute_reply":"2026-01-18T03:00:24.851111Z"}}
plt.figure(figsize=(12,5))
plt.plot(ts.index, ts["unit_price"], label="Sales Value")
plt.scatter(
    anomalies.index,
    anomalies["unit_price"],
    color="red",
    label="Anomaly"
)
plt.title("Anomaly Detection in Monthly Sales")
plt.xlabel("Month")
plt.ylabel("Sales Value")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3.3 Anomaly Detection
# Several months exhibit unusually high or low sales values relative to the
# overall trend. These anomalies may be driven by bulk purchases, promotional
# campaigns, supply disruptions, or changes in customer behavior.
# 
# Identifying such periods helps organizations investigate root causes and
# improve demand forecasting accuracy.
# 
