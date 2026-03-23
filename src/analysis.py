"""
COVID-19 data analysis project.

This script analyses case, death, and vaccination datasets using:
- summary statistics by continent
- ranking of European countries by total deaths
- exponential growth modelling of early case data
- weighted linear regression of vaccination trends
- reduced chi-squared evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path

def exponential_growth_model(t, A, R, O):
    return A * (1 + R)**t + O

def linear_model(x, m, c):
    return m * x + c
  
def main():
    # Defining directories
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    FIGURES_DIR = BASE_DIR / "figures"
    FIGURES_DIR.mkdir(exist_ok=True)
    
    vaccinations = pd.read_csv(DATA_DIR / "modified_country_vaccinations.csv")
    daily = pd.read_csv(DATA_DIR / "worldometer_coronavirus_daily_data.csv")
    summary = pd.read_csv(DATA_DIR / "worldometer_coronavirus_summary_data.csv")
    
    # Summary statistics by continent
    continents = summary['continent'].unique()
    for continent in continents:
      continent_data = summary[summary['continent'] == continent]
      total_cases = int(continent_data['total_confirmed'].sum())
      total_deaths = int(continent_data['total_deaths'].sum())
      percentage = (total_deaths/total_cases) * 100
      print(f'Continent: {continent}')
      print(f'Total confirmed cases: {total_cases}')
      print(f'Total deaths: {total_deaths}')
      print(f'Death percentage {percentage:.2f}%')
      print('-------------------------------------')
    
    # Filter and rank European countries by total deaths
    europe_countries = summary[summary['continent'] == 'Europe']
    europe_sorted = europe_countries.sort_values(by='total_deaths', ascending=False)
    print(europe_sorted.head(5))  # Display the top 5 for inspection
    # Total number of deaths for the 10 European countries with the highest number of total deaths.
    top10 = europe_sorted.head(10)
    fig = plt.figure(figsize=(12, 16))
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    
    ax1.bar(top10['country'], top10['total_deaths'], color=plt.cm.viridis(np.linspace(0, 1, 10)))
    ax1.set_title('Top 10 European Countries by Total COVID Deaths')
    ax1.set_xlabel('Country')
    ax1.set_ylabel('Total Deaths')
    
    # Plot daily new COVID-19 cases for the first 28 usable days for the UK, France, and Germany.
    # The first row is excluded because the day-zero data is incomplete.
    countries = daily[daily['country'].isin(['UK', 'France', 'Germany'])]
    
    uk = countries[countries['country'] == 'UK']
    france = countries[countries['country'] == 'France']
    germany = countries[countries['country'] == 'Germany']
    
    uk_cases = uk['daily_new_cases'][1:29]
    france_cases = france['daily_new_cases'][1:29]
    germany_cases = germany['daily_new_cases'][1:29]
    
    case_days = np.arange(1, 29)
    
    ax2.scatter(case_days, uk_cases, label='UK data', marker='o', s=20)
    ax2.scatter(case_days, france_cases, label='France data', marker='o', s=20)
    ax2.scatter(case_days, germany_cases, label='Germany data', marker='o', s=20)
    ax2.set_title('New COVID Cases in the First 28 Days')
    ax2.set_xlabel('Day Number')
    ax2.set_ylabel('Daily New Cases')
    
    # We model the number of daily new COVID cases using:
    # y(t) = A(1 + R)^t + O
    #
    # where:
    #   A = initial number of cases
    #   R = growth rate
    #   t = time (days)
    #   O = offset
    #
    # We fit this model to the first 28 days of data for the UK, France, and Germany
    # using scipy.optimize.curve_fit. This allows us to estimate the growth rate R
    # for each country, along with its uncertainty from the covariance matrix.
    uk_params, uk_cov = curve_fit(exponential_growth_model, case_days, uk_cases, p0=[1, 0.2, 0])
    fr_params, fr_cov = curve_fit(exponential_growth_model, case_days, france_cases, p0=[1, 0.2, 0])
    ge_params, ge_cov = curve_fit(exponential_growth_model, case_days, germany_cases, p0=[1, 0.2, 0])
    
    A_uk, R_uk, O_uk = uk_params
    A_fr, R_fr, O_fr = fr_params
    A_ge, R_ge, O_ge = ge_params
    
    R_uk_err = np.sqrt(uk_cov[1, 1])
    R_fr_err = np.sqrt(fr_cov[1, 1])
    R_ge_err = np.sqrt(ge_cov[1, 1])
  
    t_fit = np.linspace(1, 28, 100)
    
    y_uk_fit = exponential_growth_model(t_fit, A_uk, R_uk, O_uk)
    y_fr_fit = exponential_growth_model(t_fit, A_fr, R_fr, O_fr)
    y_ge_fit = exponential_growth_model(t_fit, A_ge, R_ge, O_ge)
    
    ax2.plot(t_fit, y_uk_fit, '-', color='tab:blue', linewidth=2.5, alpha=0.8, label='UK fit')
    ax2.plot(t_fit, y_fr_fit, '-', color='tab:orange', linewidth=2.5, alpha=0.8, label='France fit')
    ax2.plot(t_fit, y_ge_fit, '-', color='tab:green', linewidth=2.5, alpha=0.8, label='Germany fit')
    ax2.legend()
    
    print(f"UK growth rate R = {R_uk:.2g} ± {R_uk_err:.2f}")
    print(f"France growth rate R = {R_fr:.2g} ± {R_fr_err:.2f}")
    print(f"Germany growth rate R = {R_ge:.2g} ± {R_ge_err:.2f}")
    
    # We analyse the total number of vaccinations in the UK between days 10 and 50.
    # A subset of the data is selected, along with associated uncertainties given
    # by the "VaxError" column.
    #
    # We perform a weighted linear regression of the form:
    #   y = m x + c
    #
    # where:
    #   m = vaccination rate (vaccinations per day)
    #   c = intercept
    #
    # The fit is weighted using the provided uncertainties (sigma = VaxError),
    # meaning data points with smaller errors have a greater influence on the fit.
    #
    # The parameter m gives the vaccination rate, and its uncertainty is obtained
    # from the covariance matrix returned by curve_fit.
    uk_vax = vaccinations[vaccinations['country'] == 'United Kingdom']
    uk_vaccination_window = uk_vax.iloc[9:49]
    
    vaccinations_data = uk_vaccination_window['total_vaccinations']
    errors = uk_vaccination_window['VaxError']
    
    vaccination_days = np.arange(9, 49)
    
    vaccination_params, vaccination_cov = curve_fit(
      linear_model,
      vaccination_days,
      vaccinations_data,
      sigma=errors,
      absolute_sigma=True
    )
    
    m_err = np.sqrt(vaccination_cov[0, 0])
    
    x_fit = np.linspace(9, 48, 100)
    y_fit = linear_model(x_fit, vaccination_params[0], vaccination_params[1])
    
    ax3.errorbar(vaccination_days, vaccinations_data, yerr=errors, fmt='o', label='Data', markersize=4, capsize=3)
    ax3.plot(x_fit, y_fit, '--', label='Weighted fit')
    
    ax3.set_title('UK Vaccinations (Days 9–48)')
    ax3.set_xlabel('Day Number')
    ax3.set_ylabel('Total Vaccinations')
    ax3.legend()
    
    print(f'The vaccination rate over days 10 to 50 was {int(round(vaccination_params[0], -3))} ± {int(round(m_err, -3))} vaccinations per day.')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "covid_analysis.png", dpi=300)
    plt.show()
    
    # Evaluate goodness of fit using the reduced chi-squared statistic.
    # A value close to 1 indicates a good fit. Values significantly less than 1
    # may suggest that the uncertainties are overestimated or the model fits
    # the data better than expected.
    chi2 = np.sum(((vaccinations_data - linear_model(vaccination_days, vaccination_params[0], vaccination_params[1])) / errors) ** 2)
    
    dof = len(vaccination_days) - 2
    chi2_reduced = chi2 / dof
    print(
        f"Reduced chi-squared = {chi2_reduced:.3f}. "
        "This is significantly less than 1, which suggests that the "
        "uncertainties may be overestimated, or that the data follows "
        "the model more closely than expected given the stated errors."
    )
  
if __name__ == "__main__":
    main()
