# European Demographics and Fertility Analysis (2023–2024)

**Interactive Data Visual Story:** (https://europe-fert-demograph.my.canva.site/)

## Overview
This project presents a comprehensive analysis of European fertility trends, demographic changes, and the role of immigration in shaping population dynamics. Using curated datasets on **Total Fertility Rate (TFR), Crude Birth Rate, population structure, and immigrant/native shares**, the project uncovers the scale and nuances of Europe’s demographic crisis.

The study combines **data science techniques, statistical analysis, and visual storytelling** to deliver actionable insights.

---

## Key Objectives
- Analyze fertility patterns across 45 European countries.  
- Quantify native vs immigrant contributions to births.  
- Identify demographic clusters to reveal population risk zones.  
- Create interactive maps and visualizations to tell the story.  
- Generate data-driven insights for policy and societal implications.  
- Lay groundwork for forecasting future demographic scenarios.  

---

## Datasets
The project uses three main datasets:

| Dataset | Description |
|---------|-------------|
| `TFR_2023.xlsx` | Total Fertility Rate for 45 European countries. |
| `Crude_BirthRate_2023.xlsx` | Crude birth rate, share of immigrant and native mothers. |
| `Population_Jan2024_Data.csv` | Total population, immigrant population, native vs immigrant shares, gender distribution in immigrant population. |

*Future iterations will include longitudinal data for enhanced projections and trend analysis.*

---

## Analyses & Processes

### 1. Exploratory Data Analysis (EDA)
- Identify fertility trends, distribution, and anomalies.  
- Examine correlations between TFR, crude birth rate, and immigrant shares.  
- Generate summary statistics, histograms, boxplots, and scatter plots.  

### 2. Clustering & Demographic Segmentation
- K-Means clustering to categorize countries into four demographic clusters:
  1. Shrinking Natives  
  2. Mixed Profile  
  3. Balanced Transition  
  4. Immigration-Sustained  
- Highlights risk zones and policy-relevant groups.  

### 3. Correlation & Regression Analysis
- Investigate the relationship between immigrant share and birth rate.  
- Test how immigration affects native fertility trends.  

### 4. Visual Storytelling
- **Map 1:** Total Fertility Rate per country (TFR 2023) — shows universal fertility decline and replacement thresholds.  
- **Map 2:** Crude birth rate with native vs immigrant mothers — reveals immigrant contribution as a “demographic lifeline.”  
- **Map 3:** Total population with immigrant/native shares — highlights East–West demographic divide.  

### 5. Projection & “What-If” Scenarios
- Future demographic modeling under different fertility and immigration assumptions.  
- Requires longitudinal TFR, birth rate, immigration trends, and population pyramids (10–20 years).  
- Purpose: illustrate potential population trajectories and inform policy decisions.  

### 6. Optional Advanced Techniques
- **Sentiment Analysis / NLP:** Measure public perception of immigration and fertility policies using news, Twitter, Reddit data.  
- **Machine Learning / Deep Learning:** Predict future fertility rates or cluster emerging demographic patterns.  

---

## Key Insights
- All European countries fall below replacement fertility (TFR < 2.1); 32 out of 45 in ultra-low fertility (≤1.5).  
- Immigrants disproportionately contribute to births: ~1.33× their population share.  
- Population decline is uneven; Southern and Eastern Europe are at greatest risk.  
- Immigration prevents demographic collapse but does not restore fertility naturally.  
- Policy interventions (family support, childcare, incentives) help but cannot fully reverse ultra-low fertility trends.  

---

## Storytelling Angle
The project combines hard data with narrative insights, showing that while Europe faces a “fertility crisis,” immigrants are the hidden force keeping the continent afloat. This makes the analysis suitable for **journalistic publication, policy briefings, and interactive data storytelling**.

---

## Tools & Libraries
- **Python:** pandas, numpy, matplotlib, seaborn, scikit-learn, plotly  
- **Data Visualization:** Datawrapper (maps), matplotlib, seaborn  
- **Document Preparation:** Jupyter Notebook  
- **Optional:** NLP libraries (transformers, spaCy) for sentiment analysis  

---

> **Note:** This project is a work in progress. Future iterations will expand datasets, refine analyses, and enhance interactive visual storytelling.
