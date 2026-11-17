"""
stats_tools.py
Copied from python_chronic_disease.ipynb (Depression Analysis in the U.S.).
Same code and comments, in the same order. Each step is wrapped in a function
so the agent can call it. Exploration-only cells, plots, and the Tableau export are left out.
"""

#Importing necessary libraries for analysis
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

#Our chosen predictors
predictors = ['inactivity', 'obesity', 'sleeplessness', 'checkup', 'poverty']


# ---------------------------------------------------------------
# Data Cleaning
# ---------------------------------------------------------------

def load_raw(path="chronic_disease.csv"):
    #Importing the data into the code
    disease = pd.read_csv(path, low_memory=False)

    #Changing column headers to lowercase to make calling them easier.
    disease.columns = disease.columns.str.lower()

    #Dropping unneeded columns
    disease = disease.drop(columns=['datavaluefootnotesymbol', 'stratificationcategory2', 'stratification2', 'stratificationcategory3',
    'stratification3', 'locationid', 'topicid', 'questionid', 'responseid', 'datavaluetypeid', 'stratificationcategoryid1',
    'stratificationid1', 'stratificationcategoryid2', 'stratificationid2', 'stratificationcategoryid3', 'stratificationid3', 'response', 'datavaluealt'])

    #renaming high confidence limit and low confidence limit to be a bit shorter
    disease = disease.rename(columns={'lowconfidencelimit': 'LCL', 'highconfidencelimit': 'HCL'})

    #Getting rid of the commas in number columns
    disease['datavalue'] = disease.datavalue.str.replace(",", "")
    disease['LCL'] = disease.LCL.str.replace(",", "")
    disease['HCL'] = disease.HCL.str.replace(",", "")

    #Converting datatypes of number columns to float
    disease['datavalue'] = disease.datavalue.astype('float64')
    disease['LCL'] = disease.LCL.astype('float64')
    disease['HCL'] = disease.HCL.astype('float64')

    return disease


def get_national_depression(disease):
    #Taking a look at rows with Depression among adults across the U.S.
    national_depression = disease.loc[ (disease.question == "Depression among adults") & (disease.locationabbr == "US")]
    return national_depression


# ---------------------------------------------------------------
# Statistical Analysis
# ---------------------------------------------------------------

def overall_depression(national_depression):
    #Viewing national overall depression rates in the U.S. from 2019 to 2023
    overall_depression = national_depression.loc[ (national_depression.stratificationcategory1 == "Overall") & (national_depression.datavaluetype == "Age-adjusted Prevalence")]
    overall_depression = overall_depression.sort_values('yearend')
    return overall_depression[['yearend', 'datavalue', 'LCL', 'HCL']]


def depression_by_sex(national_depression, year=2022):
    #Viewing national overall depression rates in the U.S. by sex
    s_depression = national_depression.loc[ (national_depression.stratificationcategory1 == "Sex") & (national_depression.datavaluetype == "Age-adjusted Prevalence")
          & (national_depression.yearend == year)]
    return s_depression[['stratification1', 'datavalue', 'LCL', 'HCL']]


def depression_by_age(national_depression, year=2022):
    #Viewing national overall depression rates in the U.S. by age
    a_depression = national_depression.loc[ (national_depression.stratificationcategory1 == "Age") & (national_depression.datavaluetype == "Crude Prevalence")
          & (national_depression.yearend == year)]
    return a_depression[['stratification1', 'datavalue', 'LCL', 'HCL']]


def depression_by_race(national_depression, year=2022):
    #Viewing national overall depression rates in the U.S. by race
    r_depression = national_depression.loc[ (national_depression.stratificationcategory1 == "Race/Ethnicity") & (national_depression.datavaluetype == "Age-adjusted Prevalence")
          & (national_depression.yearend == year)]
    r_depression = r_depression.sort_values('datavalue')
    return r_depression[['stratification1', 'datavalue', 'LCL', 'HCL']]


# ---------------------------------------------------------------
# Regression
# ---------------------------------------------------------------

def load_state_data(disease, year=2022):
    #Formatting our table to fit our regression model. We will treat the following as our variables for regression: inactivity', 'obesity', 'sleeplessness', 'checkup', and 'poverty'
    #We will be focusing on the relationship between depression and the other variables.
    D = disease.loc[ (disease.stratificationcategory1 == "Overall") & (disease.yearend == year) &
                    (disease.question.isin(['Depression among adults', 'No leisure-time physical activity among adults',
                                            'Obesity among adults', 'Short sleep duration among adults', 'Living below 150% of the poverty threshold among all people',
                                            'Routine checkup within the past year among adults']))]

    #Dropping any output that is not a U.S. state. This will be a state by state analysis.
    D = D[~D['locationabbr'].isin(['US', 'GU', 'PR', 'VI', 'DC'])]

    #Filtering our table for age-adjusted prevalence except for poverty which will be crude. Checking for any null values after.
    D = D[(D.datavaluetype == 'Age-adjusted Prevalence') |
          ((D.question == 'Living below 150% of the poverty threshold among all people') & (D.datavaluetype == 'Crude Prevalence'))]

    print(D.datavalue.isnull().sum())

    #Checking that each of our variables has the same amount of rows, one for each state.
    print(D.groupby(['question', 'datavaluetype']).size())

    #Pivoting our table so now each state is its own row with the columns being our questions (variables). Each output is a percentage.
    state_data = D.pivot_table(index='locationabbr', columns='question', values='datavalue')

    #Renaming our columns to be a bit shorter
    state_data = state_data.rename(columns={'Depression among adults': 'depression', 'No leisure-time physical activity among adults' : 'inactivity',
            'Obesity among adults' : 'obesity', 'Short sleep duration among adults' : 'sleeplessness', 'Routine checkup within the past year among adults' : 'checkup',
            'Living below 150% of the poverty threshold among all people' : 'poverty'})

    return state_data


def correlation(state_data):
    #Making a correlation table for our variables
    return state_data.corr()


def simple_regression(state_data, predictor='obesity'):
    #Running a least squares regression with obesity being the chosen predictor, due to it having the highest r value
    simple = smf.ols('depression ~ ' + predictor, data=state_data).fit()
    print(simple.summary())
    return simple


def full_regression(state_data):
    #Running a least squares regression with all our predictors.
    model1 = smf.ols('depression ~ inactivity + obesity + sleeplessness + checkup + poverty', data=state_data).fit()
    print(model1.summary())
    return model1


# ---------------------------------------------------------------
# Model Diagnostics
# ---------------------------------------------------------------

def vif_test(state_data):
    #Running a VIF test for our predictors
    X = sm.add_constant(state_data[predictors])
    vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(1, X.shape[1])], index=predictors)
    print(vif)
    return vif


def cooks_distance(model1, state_data):
    #Cook's distance to test outlier influence
    cook = pd.Series(model1.get_influence().cooks_distance[0], index=state_data.index)
    print(cook.sort_values(ascending=False).head())
    return cook


def cook_comparison(model1, state_data, cook):
    #The moderate threshold for Cook's distance is D > 4/n. Finding the states over that cutoff.
    influential = cook[cook > 4 / len(state_data)].index.tolist()

    #Running a comparison without the most influential states to see how much the results change
    model_cook = smf.ols('depression ~ inactivity + obesity + sleeplessness + checkup + poverty',
                           data=state_data.drop(influential)).fit()

    comparison = pd.DataFrame({'coeff full': model1.params, 'coeff cook': model_cook.params,
                               'p full': model1.pvalues, 'p cook': model_cook.pvalues})
    print(comparison.round(3))
    return influential, comparison
