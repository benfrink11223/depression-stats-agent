# Depression Stats Agent

An AI agent that answers plain-English questions about U.S. adult depression by running real statistical analyses on CDC data, then explaining the results without overclaiming.

Ask it *"What predicts depression across states?"* and it picks the right analysis (regression, multicollinearity check, outlier check), runs it in Python, and reports the numbers with the right caveats: state-level data doesn't describe individuals, association isn't causation, and overlapping confidence intervals aren't evidence of a change.

This project builds on my earlier analysis, [U.S. Depression Analysis](https://github.com/benfrink11223/us-depression-analysis) (Python + [Tableau dashboards](https://public.tableau.com/app/profile/benjamin.frink)). The agent's tools are the same analysis code from that project, so its answers can be checked against results I worked out by hand.

## How it works

1. **The question** goes to an LLM (Claude Haiku) along with descriptions of 9 analysis tools and a set of statistical rules (the system prompt).
2. **The model chooses a tool**, for example `full_regression` for 2022. It never runs code itself; it only asks for a tool.
3. **Python runs the tool** on the CDC data (pandas + statsmodels) and sends the results back as text.
4. **The loop repeats** until the model has what it needs, then it writes the answer.

```
question → model picks tool → Python runs analysis → results back to model → ... → answer
```

### Tools

| Tool | What it does |
|---|---|
| `overall_depression` | National depression rate 2019–2023 with 95% confidence intervals |
| `depression_by_sex` / `_by_age` / `_by_race` | National rates by group for a given year |
| `correlation` | Correlations across the 50 states between depression and 5 predictors |
| `simple_regression` | OLS regression of state depression on one predictor |
| `full_regression` | OLS regression on all 5 predictors (inactivity, obesity, sleeplessness, checkup, poverty) |
| `vif_test` | Variance inflation factors (multicollinearity check) |
| `outlier_check` | Cook's distance, influential states, and the model re-run without them |

### Guardrails
- **In the system prompt:** report the numbers used; state-level results don't apply to individuals; describe associations, not causes; overlapping confidence intervals are inconclusive; check VIF and correlations before interpreting a surprising coefficient; use adjusted R² for multi-predictor models; say so when data is missing.
- **In the tools:** every state-level result carries a note that it describes states, not individuals, and years with missing data return a clear error instead of a wrong answer.

## Evaluation

I wrote a 14-question answer key (`answer_key.xlsx`) from my original analysis: 9 standard questions and 5 **trap questions** designed to tempt an overclaim.

| Trap question | The mistake it tests |
|---|---|
| Are obese people more likely to be depressed? | Applying state-level data to individuals (ecological fallacy) |
| Does obesity cause depression? | Treating association as causation |
| Does being physically inactive protect against depression? | Trusting a coefficient whose sign flipped because of multicollinearity |
| Does poverty predict depression in 2023? | Answering when the data doesn't exist (no 2023 poverty data) |
| Did depression go down from 2022 to 2023? | Calling a change real when the confidence intervals overlap |

### Results

| | Score |
|---|---|
| Correct tool chosen | 14 / 14 |
| Correct numbers and conclusions | 14 / 14 |
| Trap questions handled correctly | 5 / 5 |

Example: asked whether inactivity protects against depression, the agent noted that inactivity's simple correlation with depression is near zero (r = 0.07), that its coefficient only turns negative in the full model, and that inactivity is strongly correlated with poverty (r = 0.80) and obesity (r = 0.72). It concluded the negative coefficient is a multicollinearity artifact, not a protective effect.

### Experiment: do the guardrails matter?

I re-ran the 5 trap questions with a bare-bones prompt that keeps the task but drops all the statistical rules.

| Setup | Trap score |
|---|---|
| With system prompt rules | 5 / 5 |
| Without system prompt rules | 4 / 5 |

The difference was smaller than I expected. Without the rules, the agent still flagged the ecological fallacy, refused to call obesity causal, caught the inactivity sign flip, and reported the missing 2023 data. It failed on the 2022→2023 question, opening with "Yes, but only slightly" before admitting the intervals overlap.

**Takeaway:** most of the protection came from tool design (built-in caveats and clear error messages) and the model's own defaults. The system prompt mainly changed framing: leading with the cautious answer and keeping the agent to the data instead of outside knowledge.

## Limitations
- Each question was run once. LLM answers vary between runs, so repeated trials would give a more reliable score.
- The data is state-level and observational. Nothing here describes individual people or causes.
- Some measures (poverty, sleeplessness) are missing for 2023, so state-level models run for 2019–2022.
- The tools are specific to this dataset. The agent loop and statistical rules are reusable, but a new dataset needs new tool functions.

## Project structure

```
├── agent.ipynb          # builds the agent step by step and runs the evaluation
├── agent_tools.py       # tool descriptions, system prompt, and run_tool()
├── stats_tools.py       # analysis functions from the original project
├── answer_key.xlsx      # questions, expected answers, and results
├── requirements.txt
└── .gitignore
```

## Running it yourself

1. Clone the repo and install the packages:
   ```
   pip install -r requirements.txt
   ```
2. Download the [CDC U.S. Chronic Disease Indicators](https://data.cdc.gov/Chronic-Disease-Indicators/U-S-Chronic-Disease-Indicators/hksd-2xuw) data as CSV, rename it `chronic_disease.csv`, and put it in the project folder.
3. Create a file named `.env` containing your Anthropic API key:
   ```
   ANTHROPIC_API_KEY=your-key-here
   ```
4. Open `agent.ipynb`, run the setup cells, and ask a question:
   ```python
   ask("What predicts depression across states in 2022?");
   ```

## Tools used
Python, pandas, statsmodels, Anthropic API (tool use), Jupyter

## Data source
Centers for Disease Control and Prevention, [U.S. Chronic Disease Indicators](https://data.cdc.gov/Chronic-Disease-Indicators/U-S-Chronic-Disease-Indicators/hksd-2xuw).
