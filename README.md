# COVID-19 Data Analysis

This project analyses COVID-19 datasets to explore trends in cases, deaths, and vaccinations using Python.

## Overview

The analysis includes:

* Summary statistics of COVID-19 cases and deaths by continent
* Ranking of European countries by total deaths
* Modelling early pandemic growth using an exponential model
* Estimation of growth rates with uncertainties
* Weighted linear regression of UK vaccination data
* Evaluation of model quality using the reduced chi-squared statistic

## Data

The following datasets are used:

* `modified_country_vaccinations.csv`
* `worldometer_coronavirus_daily_data.csv`
* `worldometer_coronavirus_summary_data.csv`

These should be placed in the `data/` directory.

## Project Structure

```
covid-data-analysis/
│
├── data/ # Datasets used for analysis
├── src/ # Analysis scripts and core logic
│ └── analysis.py # Main data analysis script
│
├── README.md # Project overview and documentation
├── requirements.txt # Python dependencies
├── .gitignore # Files ignored by Git
└── LICENSE # Project license
```

## How to Run

1. Install required packages:

```
pip install -r requirements.txt
```

2. Run the script:

```
python src/analysis.py
```

3. Output:

* Printed statistics in the terminal
* A figure saved in `figures/covid_analysis.png`

## Methods

* Exponential growth model:
  y(t) = A(1 + R)^t + O
* Weighted linear regression using `curve_fit`
* Uncertainty estimation from covariance matrices
* Reduced chi-squared used to assess fit quality

## Results

* Growth rates for early COVID-19 spread were estimated for the UK, France, and Germany
* Vaccination trends in the UK were modelled and quantified
* Model fit quality was assessed using statistical methods
<img width="1189" height="1589" alt="Analysis" src="https://github.com/user-attachments/assets/1f876113-17b2-4db6-969e-5f3b5798cc65" />

## Technologies Used

* Python
* pandas
* NumPy
* Matplotlib
* SciPy

## Notes

This project was developed as part of a university computing skills module.
