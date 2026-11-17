"""
agent_tools.py
Connects the agent to stats_tools.py:
  - TOOLS: the descriptions the model reads to decide which tool to call
  - SYSTEM_PROMPT: standing instructions for the agent
  - run_tool(): runs the function the model asked for and returns the result as text
"""

import contextlib
import io

import stats_tools as st

#Loading the data once when this file is imported
disease = st.load_raw()
national_depression = st.get_national_depression(disease)


# ---------------------------------------------------------------
# Instructions the agent follows on every question
# ---------------------------------------------------------------

SYSTEM_PROMPT = """You are a careful statistical analyst answering questions about U.S. adult depression
using CDC Chronic Disease Indicators data (2019-2023). Always use your tools to get numbers; never guess.

About the data:
- National tools return U.S.-wide rates with 95% confidence intervals (LCL, HCL).
- State tools (correlation, regressions, VIF, outlier check) use one row per state (50 states).
- The regression predictors are: inactivity (behavior), obesity (physical health),
  sleeplessness (lifestyle), poverty (economic), and checkup (access to care).

Rules:
- Report the numbers you used (rates, confidence intervals, coefficients, p-values, R-squared).
- For models with more than one predictor, describe fit using adjusted R-squared, not plain R-squared.
- State-level results describe states, not individual people. Never apply them to individuals.
- This is observational data. Describe associations, never causes.
- Non-overlapping confidence intervals suggest a significant difference. Overlapping intervals
  do not prove there is no difference; say the evidence is inconclusive.
- If a coefficient's sign looks surprising, check the VIF and correlations before interpreting it.
- If a tool returns an error or data is missing, say so plainly instead of answering anyway.
- Keep answers short and in plain English."""


#Bare-bones prompt for the guardrails experiment: same job, none of the statistical rules
BASELINE_PROMPT = """You answer questions about U.S. adult depression using CDC Chronic Disease Indicators data (2019-2023).
Use your tools to get numbers."""


# ---------------------------------------------------------------
# Tool descriptions the model reads
# ---------------------------------------------------------------

def _tool(name, description, properties=None):
    return {
        "name": name,
        "description": description,
        "input_schema": {"type": "object", "properties": properties or {}, "required": []},
    }

YEAR = {"year": {"type": "integer", "description": "Year between 2019 and 2023. Defaults to 2022."}}

TOOLS = [
    _tool("overall_depression",
          "U.S. national adult depression rate (age-adjusted %) for every year 2019-2023, with 95% confidence intervals."),
    _tool("depression_by_sex",
          "U.S. national depression rate (age-adjusted %) for females and males in one year, with 95% confidence intervals.", YEAR),
    _tool("depression_by_age",
          "U.S. national depression rate (crude %) for age groups 18-44, 45-64, and 65+ in one year, with 95% confidence intervals.", YEAR),
    _tool("depression_by_race",
          "U.S. national depression rate (age-adjusted %) by race/ethnicity in one year, with 95% confidence intervals.", YEAR),
    _tool("correlation",
          "Correlation table across the 50 states between depression and the five predictors (inactivity, obesity, sleeplessness, checkup, poverty).", YEAR),
    _tool("simple_regression",
          "OLS regression of state depression rate on ONE predictor across the 50 states. Returns coefficients, p-values, R-squared.",
          {**YEAR, "predictor": {"type": "string",
                                 "enum": ["inactivity", "obesity", "sleeplessness", "checkup", "poverty"],
                                 "description": "The single predictor to use."}}),
    _tool("full_regression",
          "OLS regression of state depression rate on all five predictors together across the 50 states. Returns coefficients, p-values, R-squared, F-test.", YEAR),
    _tool("vif_test",
          "Variance inflation factors for the five predictors, to check multicollinearity. VIF above 5 is a concern.", YEAR),
    _tool("outlier_check",
          "Cook's distance for the full regression, the states above the 4/n cutoff, and the regression re-run without those states.", YEAR),
]


# ---------------------------------------------------------------
# Running the tool the model asked for
# ---------------------------------------------------------------

def run_tool(name, inputs):
    #stats_tools prints a lot of output; hide it so only the agent's steps show
    with contextlib.redirect_stdout(io.StringIO()):
        return _run_tool(name, inputs)


def _run_tool(name, inputs):
    year = inputs.get("year", 2022)

    if name == "overall_depression":
        return st.overall_depression(national_depression).to_string(index=False)

    if name == "depression_by_sex":
        return st.depression_by_sex(national_depression, year).to_string(index=False)

    if name == "depression_by_age":
        return st.depression_by_age(national_depression, year).to_string(index=False)

    if name == "depression_by_race":
        return st.depression_by_race(national_depression, year).to_string(index=False)

    #Everything below uses the 50-state table
    state_data = st.load_state_data(disease, year)
    missing = [p for p in ["depression"] + st.predictors if p not in state_data.columns]
    if missing:
        return f"Error: no data in {year} for: {', '.join(missing)}. This analysis can't be run for that year."
    note = (f"\n(State-level data, {year}, n = {state_data.dropna().shape[0]} states. "
            "Results describe states, not individuals.)")

    if name == "correlation":
        return st.correlation(state_data).round(2).to_string() + note

    if name == "simple_regression":
        model = st.simple_regression(state_data, inputs.get("predictor", "obesity"))
        return model.summary().as_text() + note

    if name == "full_regression":
        model = st.full_regression(state_data)
        return model.summary().as_text() + note

    if name == "vif_test":
        return st.vif_test(state_data).round(2).to_string() + note

    if name == "outlier_check":
        model = st.full_regression(state_data)
        cook = st.cooks_distance(model, state_data)
        influential, comparison = st.cook_comparison(model, state_data, cook)
        return (f"Cook's distance cutoff (4/n): {4 / len(state_data):.3f}\n"
                f"Influential states: {influential}\n"
                f"Top Cook's distances:\n{cook.sort_values(ascending=False).head().round(3).to_string()}\n\n"
                f"Regression with vs. without influential states:\n{comparison.round(3).to_string()}" + note)

    return f"Error: unknown tool '{name}'"
