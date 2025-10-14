# Fertility, Birth Rates, and Immigration in Europe: A Data Analysis (2023) 🚼🌍

> **Project Status:** UNDER CONSTRUCTION ⏳
> This repository is actively being developed. More insights, visualizations, and analyses will be added soon.

This project explores **fertility rates**, **crude birth rates**, and the impact of **immigration** across 36 European countries in 2023. The goal is to understand demographic patterns, identify clusters of similar countries, and visualize relationships between fertility, births, and immigrant populations.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Data Sources](#data-sources)
3. [Libraries & Tools](#libraries--tools)
4. [Data Cleaning & Merging](#data-cleaning--merging)
5. [Feature Engineering](#feature-engineering)
6. [Summary Statistics](#summary-statistics)
7. [Visualizations](#visualizations)
8. [Regression Analysis](#regression-analysis)
9. [Outlier Analysis](#outlier-analysis)
10. [Clustering Analysis](#clustering-analysis)
11. [Conclusions & Next Steps](#conclusions--next-steps)
12. [Acknowledgements](#acknowledgements)
13. [License](#license)

---

## Introduction
In this analysis, we study:

- **Total Fertility Rate (TFR)** – Average number of children born per woman.
- **Crude Birth Rate (CBR)** – Number of live births per 1,000 population.
- **Immigration impact** – Births from immigrant mothers and fertility patterns.

We explore how immigrant populations influence fertility trends and identify clusters of countries with similar demographic profiles.

---

## Data Sources
We used two datasets:

1. **TFR & Birth Data**: Fertility rates and birth distributions across European countries.
2. **Population Data (Jan 2024)**: Total population, immigrant population, and native population metrics.

> **Note:** Some values for immigrant subpopulations were missing and handled during cleaning.

---

## Libraries & Tools
We used **Python** and the following libraries:

- **Data Manipulation**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`
- **Machine Learning & Clustering**: `scikit-learn` (`LinearRegression`, `KMeans`, `PCA`, `StandardScaler`)
- **Statistical Analysis**: `statsmodels` (OLS regression, VIF calculation)

---

## Data Cleaning & Merging
Steps performed:

- Stripped whitespace from country names.
- Merged TFR & Birth and Population datasets on `Country`.
- Converted numeric columns.
- Dropped rows with missing key values (`TFR`, `Crude_Birth_Rate`).

**Result:** 32 complete observations available for analysis.

```python
data = pd.merge(tfr_birth, population, on='Country', how='inner')
data[num_cols] = data[num_cols].apply(pd.to_numeric, errors='coerce')
data = data.dropna(subset=['TFR', 'Crude_Birth_Rate'])

Here is the complete Markdown for your GitHub README:

Markdown

# Fertility, Birth Rates, and Immigration in Europe: A Data Analysis (2023) 🚼🌍

> **Project Status:** UNDER CONSTRUCTION ⏳
> This repository is actively being developed. More insights, visualizations, and analyses will be added soon.

This project explores **fertility rates**, **crude birth rates**, and the impact of **immigration** across 36 European countries in 2023. The goal is to understand demographic patterns, identify clusters of similar countries, and visualize relationships between fertility, births, and immigrant populations.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Data Sources](#data-sources)
3. [Libraries & Tools](#libraries--tools)
4. [Data Cleaning & Merging](#data-cleaning--merging)
5. [Feature Engineering](#feature-engineering)
6. [Summary Statistics](#summary-statistics)
7. [Visualizations](#visualizations)
8. [Regression Analysis](#regression-analysis)
9. [Outlier Analysis](#outlier-analysis)
10. [Clustering Analysis](#clustering-analysis)
11. [Conclusions & Next Steps](#conclusions--next-steps)
12. [Acknowledgements](#acknowledgements)
13. [License](#license)

---

## Introduction
In this analysis, we study:

- **Total Fertility Rate (TFR)** – Average number of children born per woman.
- **Crude Birth Rate (CBR)** – Number of live births per 1,000 population.
- **Immigration impact** – Births from immigrant mothers and fertility patterns.

We explore how immigrant populations influence fertility trends and identify clusters of countries with similar demographic profiles.

---

## Data Sources
We used two datasets:

1. **TFR & Birth Data**: Fertility rates and birth distributions across European countries.
2. **Population Data (Jan 2024)**: Total population, immigrant population, and native population metrics.

> **Note:** Some values for immigrant subpopulations were missing and handled during cleaning.

---

## Libraries & Tools
We used **Python** and the following libraries:

- **Data Manipulation**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`
- **Machine Learning & Clustering**: `scikit-learn` (`LinearRegression`, `KMeans`, `PCA`, `StandardScaler`)
- **Statistical Analysis**: `statsmodels` (OLS regression, VIF calculation)

---

## Data Cleaning & Merging
Steps performed:

- Stripped whitespace from country names.
- Merged TFR & Birth and Population datasets on `Country`.
- Converted numeric columns.
- Dropped rows with missing key values (`TFR`, `Crude_Birth_Rate`).

**Result:** 32 complete observations available for analysis.

```python
data = pd.merge(tfr_birth, population, on='Country', how='inner')
data[num_cols] = data[num_cols].apply(pd.to_numeric, errors='coerce')
data = data.dropna(subset=['TFR', 'Crude_Birth_Rate'])
Feature Engineering
Derived variables to capture fertility and immigration dynamics:

##Feature	Description
Immigrant_Share_Pop	Fraction of population that is immigrant
Immigrant_Share_Births	Fraction of births from immigrant mothers
Immigration_Fertility_Multiplier	Relative fertility of immigrants vs total population
Immigrant_Birth_Rate	Births per 1,000 immigrants
Native_Birth_Rate	Births per 1,000 native population

##Summary Statistics
We explored mean, standard deviation, minimum, maximum, and quartiles for key variables.

TFR: Average ~1.43, range 1.06–1.81

Crude Birth Rate: Average ~8.86, range 6.4–11.3

Immigration Fertility Multiplier: 0.39–2.69, showing wide variation

Here is the complete Markdown for your GitHub README:

Markdown

# Fertility, Birth Rates, and Immigration in Europe: A Data Analysis (2023) 🚼🌍

> **Project Status:** UNDER CONSTRUCTION ⏳
> This repository is actively being developed. More insights, visualizations, and analyses will be added soon.

This project explores **fertility rates**, **crude birth rates**, and the impact of **immigration** across 36 European countries in 2023. The goal is to understand demographic patterns, identify clusters of similar countries, and visualize relationships between fertility, births, and immigrant populations.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Data Sources](#data-sources)
3. [Libraries & Tools](#libraries--tools)
4. [Data Cleaning & Merging](#data-cleaning--merging)
5. [Feature Engineering](#feature-engineering)
6. [Summary Statistics](#summary-statistics)
7. [Visualizations](#visualizations)
8. [Regression Analysis](#regression-analysis)
9. [Outlier Analysis](#outlier-analysis)
10. [Clustering Analysis](#clustering-analysis)
11. [Conclusions & Next Steps](#conclusions--next-steps)
12. [Acknowledgements](#acknowledgements)
13. [License](#license)

---

## Introduction
In this analysis, we study:

- **Total Fertility Rate (TFR)** – Average number of children born per woman.
- **Crude Birth Rate (CBR)** – Number of live births per 1,000 population.
- **Immigration impact** – Births from immigrant mothers and fertility patterns.

We explore how immigrant populations influence fertility trends and identify clusters of countries with similar demographic profiles.

---

## Data Sources
We used two datasets:

1. **TFR & Birth Data**: Fertility rates and birth distributions across European countries.
2. **Population Data (Jan 2024)**: Total population, immigrant population, and native population metrics.

> **Note:** Some values for immigrant subpopulations were missing and handled during cleaning.

---

## Libraries & Tools
We used **Python** and the following libraries:

- **Data Manipulation**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`
- **Machine Learning & Clustering**: `scikit-learn` (`LinearRegression`, `KMeans`, `PCA`, `StandardScaler`)
- **Statistical Analysis**: `statsmodels` (OLS regression, VIF calculation)

---

## Data Cleaning & Merging
Steps performed:

- Stripped whitespace from country names.
- Merged TFR & Birth and Population datasets on `Country`.
- Converted numeric columns.
- Dropped rows with missing key values (`TFR`, `Crude_Birth_Rate`).

**Result:** 32 complete observations available for analysis.

```python
data = pd.merge(tfr_birth, population, on='Country', how='inner')
data[num_cols] = data[num_cols].apply(pd.to_numeric, errors='coerce')
data = data.dropna(subset=['TFR', 'Crude_Birth_Rate'])
Feature Engineering
Derived variables to capture fertility and immigration dynamics:

Feature	Description
Immigrant_Share_Pop	Fraction of population that is immigrant
Immigrant_Share_Births	Fraction of births from immigrant mothers
Immigration_Fertility_Multiplier	Relative fertility of immigrants vs total population
Immigrant_Birth_Rate	Births per 1,000 immigrants
Native_Birth_Rate	Births per 1,000 native population

Export to Sheets
Python

data['Immigrant_Share_Pop'] = data['Share_Immigrant_Percent'] / 100
data['Immigrant_Share_Births'] = data['Share_of_Births_Immigrant_Mothers'] / 100
data['Immigration_Fertility_Multiplier'] = (
    data['Immigrant_Share_Births'] / data['Immigrant_Share_Pop']
).replace([np.inf, -np.inf], np.nan)
data['Total_Births'] = (data['Crude_Birth_Rate'] / 1000) * (data['Total_Population_Millions'] * 1_000_000)
data['Immigrant_Births'] = data['Total_Births'] * data['Immigrant_Share_Births']
data['Native_Births'] = data['Total_Births'] - data['Immigrant_Births']
data['Immigrant_Birth_Rate'] = (data['Immigrant_Births'] / (data['Immigrant_Population_Millions'] * 1_000_000 + 1e-9)) * 1000
data['Native_Birth_Rate'] = (data['Native_Births'] / (data['Native_Population_Millions'] * 1_000_000 + 1e-9)) * 1000
Summary Statistics
We explored mean, standard deviation, minimum, maximum, and quartiles for key variables.

TFR: Average ~1.43, range 1.06–1.81

Crude Birth Rate: Average ~8.86, range 6.4–11.3

Immigration Fertility Multiplier: 0.39–2.69, showing wide variation

Python

desc = data[['TFR', 'Crude_Birth_Rate',
             'Immigrant_Share_Pop', 'Immigrant_Share_Births',
             'Immigration_Fertility_Multiplier',
             'Native_Birth_Rate', 'Immigrant_Birth_Rate']].describe()
print(desc)
Visualizations
1️⃣ Distributions
Histograms for TFR, Crude Birth Rate, and Immigration Fertility Multiplier reveal spread and central tendencies.

2️⃣ Relationships: TFR vs Crude Birth Rate
A scatter plot with a regression line shows a clear positive correlation between Total Fertility Rate and Crude Birth Rate.

##Regression Analysis
We performed Ordinary Least Squares (OLS) regression:

Simple Linear Regression: TFR → Crude Birth Rate.

Multiple Regression: Including Share_Immigrant_Percent and Immigration_Fertility_Multiplier.

Findings:
TFR is the strongest predictor of CBR.

Immigration variables are not statistically significant in this model.

R 
2
 ≈0.385

 Here is the complete Markdown for your GitHub README:

Markdown

# Fertility, Birth Rates, and Immigration in Europe: A Data Analysis (2023) 🚼🌍

> **Project Status:** UNDER CONSTRUCTION ⏳
> This repository is actively being developed. More insights, visualizations, and analyses will be added soon.

This project explores **fertility rates**, **crude birth rates**, and the impact of **immigration** across 36 European countries in 2023. The goal is to understand demographic patterns, identify clusters of similar countries, and visualize relationships between fertility, births, and immigrant populations.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Data Sources](#data-sources)
3. [Libraries & Tools](#libraries--tools)
4. [Data Cleaning & Merging](#data-cleaning--merging)
5. [Feature Engineering](#feature-engineering)
6. [Summary Statistics](#summary-statistics)
7. [Visualizations](#visualizations)
8. [Regression Analysis](#regression-analysis)
9. [Outlier Analysis](#outlier-analysis)
10. [Clustering Analysis](#clustering-analysis)
11. [Conclusions & Next Steps](#conclusions--next-steps)
12. [Acknowledgements](#acknowledgements)
13. [License](#license)

---

## Introduction
In this analysis, we study:

- **Total Fertility Rate (TFR)** – Average number of children born per woman.
- **Crude Birth Rate (CBR)** – Number of live births per 1,000 population.
- **Immigration impact** – Births from immigrant mothers and fertility patterns.

We explore how immigrant populations influence fertility trends and identify clusters of countries with similar demographic profiles.

---

## Data Sources
We used two datasets:

1. **TFR & Birth Data**: Fertility rates and birth distributions across European countries.
2. **Population Data (Jan 2024)**: Total population, immigrant population, and native population metrics.

> **Note:** Some values for immigrant subpopulations were missing and handled during cleaning.

---

## Libraries & Tools
We used **Python** and the following libraries:

- **Data Manipulation**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`
- **Machine Learning & Clustering**: `scikit-learn` (`LinearRegression`, `KMeans`, `PCA`, `StandardScaler`)
- **Statistical Analysis**: `statsmodels` (OLS regression, VIF calculation)

---

## Data Cleaning & Merging
Steps performed:

- Stripped whitespace from country names.
- Merged TFR & Birth and Population datasets on `Country`.
- Converted numeric columns.
- Dropped rows with missing key values (`TFR`, `Crude_Birth_Rate`).

**Result:** 32 complete observations available for analysis.

```python
data = pd.merge(tfr_birth, population, on='Country', how='inner')
data[num_cols] = data[num_cols].apply(pd.to_numeric, errors='coerce')
data = data.dropna(subset=['TFR', 'Crude_Birth_Rate'])
Feature Engineering
Derived variables to capture fertility and immigration dynamics:

Feature	Description
Immigrant_Share_Pop	Fraction of population that is immigrant
Immigrant_Share_Births	Fraction of births from immigrant mothers
Immigration_Fertility_Multiplier	Relative fertility of immigrants vs total population
Immigrant_Birth_Rate	Births per 1,000 immigrants
Native_Birth_Rate	Births per 1,000 native population

Export to Sheets
Python

data['Immigrant_Share_Pop'] = data['Share_Immigrant_Percent'] / 100
data['Immigrant_Share_Births'] = data['Share_of_Births_Immigrant_Mothers'] / 100
data['Immigration_Fertility_Multiplier'] = (
    data['Immigrant_Share_Births'] / data['Immigrant_Share_Pop']
).replace([np.inf, -np.inf], np.nan)
data['Total_Births'] = (data['Crude_Birth_Rate'] / 1000) * (data['Total_Population_Millions'] * 1_000_000)
data['Immigrant_Births'] = data['Total_Births'] * data['Immigrant_Share_Births']
data['Native_Births'] = data['Total_Births'] - data['Immigrant_Births']
data['Immigrant_Birth_Rate'] = (data['Immigrant_Births'] / (data['Immigrant_Population_Millions'] * 1_000_000 + 1e-9)) * 1000
data['Native_Birth_Rate'] = (data['Native_Births'] / (data['Native_Population_Millions'] * 1_000_000 + 1e-9)) * 1000
Summary Statistics
We explored mean, standard deviation, minimum, maximum, and quartiles for key variables.

TFR: Average ~1.43, range 1.06–1.81

Crude Birth Rate: Average ~8.86, range 6.4–11.3

Immigration Fertility Multiplier: 0.39–2.69, showing wide variation

Python

desc = data[['TFR', 'Crude_Birth_Rate',
             'Immigrant_Share_Pop', 'Immigrant_Share_Births',
             'Immigration_Fertility_Multiplier',
             'Native_Birth_Rate', 'Immigrant_Birth_Rate']].describe()
print(desc)
Visualizations
1️⃣ Distributions
Histograms for TFR, Crude Birth Rate, and Immigration Fertility Multiplier reveal spread and central tendencies.

2️⃣ Relationships: TFR vs Crude Birth Rate
A scatter plot with a regression line shows a clear positive correlation between Total Fertility Rate and Crude Birth Rate.

Regression Analysis
We performed Ordinary Least Squares (OLS) regression:

Simple Linear Regression: TFR → Crude Birth Rate.

Multiple Regression: Including Share_Immigrant_Percent and Immigration_Fertility_Multiplier.

Findings:
TFR is the strongest predictor of CBR.

Immigration variables are not statistically significant in this model.

R 
2
 ≈0.385

Python

X = data[['TFR', 'Share_Immigrant_Percent', 'Immigration_Fertility_Multiplier']].dropna()
y = data.loc[X.index, 'Crude_Birth_Rate']
X = sm.add_constant(X)
ols_model = sm.OLS(y, X).fit()
print(ols_model.summary())
Variance Inflation Factor (VIF)
Checked multicollinearity; no severe multicollinearity observed.

##Outlier Analysis
Residuals were computed from regression predictions and countries with residuals $ > 2.5 \times \text{std}$ were flagged.

Result: No extreme outliers identified.

Python

data['Predicted_CBR'] = ols_model.predict(X)
data['Residual'] = data['Crude_Birth_Rate'] - data['Predicted_CBR']
outliers = data[np.abs(data['Residual']) > 2.5 * data['Residual'].std()]
print(outliers[['Country', 'Residual']])

##Clustering Analysis
Methodology
Features: TFR, Crude_Birth_Rate, Share_Immigrant_Percent, Immigration_Fertility_Multiplier.

Pre-processing: Features were standardized using StandardScaler.

Dimensionality Reduction: PCA was applied.

Clustering: K-Means with k=4 was used.

Validation: Silhouette Score ≈0.44.

Cluster Summary
Countries were grouped based on similar fertility, birth, and immigrant characteristics. A heatmap displays the cluster mean values for the features.

Countries per cluster:

Cluster	Countries
Cluster 0	Estonia, Finland, Greece, Italy, Latvia, Lithuania, Poland, Spain
Cluster 1	Austria, Belgium, Cyprus, Germany, Iceland, Ireland, Malta, Netherlands, Norway, Portugal, Sweden, Switzerland
Cluster 2	Bulgaria, Croatia, Czechia, Denmark, France, Hungary, Romania, Slovakia, Slovenia, Türkiye
Cluster 3	Liechtenstein, Luxembourg
