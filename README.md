# Week 6 Tree Ensembles — Ioannis Tzalas

Continues [Week 1](https://github.com/flappybird69/week1-financial-analysis-ioannis-tzalas), [Week 2](https://github.com/flappybird69/week2-data-hygiene-ioannis-tzalas), [Week 3](https://github.com/flappybird69/week3-fintech-assignment-ioannis-tzalas), [Week 4](https://github.com/flappybird69/week4-time-series-forecasting-ioannis-tzalas), and [Week 5](https://github.com/flappybird69/week5-feature-tournament-ioannis-tzalas), using **SPY, EUR/USD, BTC, GLD and ETH**.

- [`W6_sprint_Tzalas.ipynb`](W6_sprint_Tzalas.ipynb): executed S0–S8 assignment, interpretations, prompt log and executable evaluation reports. Contains 11 code cells and 241 physical code lines, including imports and kit calls.
- [`fintech_kit_w6.py`](fintech_kit_w6.py): the supplied kit's definition cells, extracted verbatim from the notebook-format kit; its final smoke-test execution is excluded. The assignment discloses a runtime close-rule correction to `rel_vs_spy`.
- [`data/W2_own_calendar_prices.csv`](data/W2_own_calendar_prices.csv): frozen cleaned Week 2 prices, retaining each asset's observed calendar.
- [`requirements.txt`](requirements.txt): exact dependency versions used.

## Assignment

A tree and two boosting steps by hand; forest disagreement and OOB diagnostics; time-ordered complexity ladders for the null, Lasso, random forest, gradient boosting and histogram gradient boosting; holdout mean/variance comparisons; permutation noise controls and SHAP; risk-forecast selection and capped inverse-volatility portfolios; five isolated repairs to the supplied faulty pipeline; and five per-asset evaluation reports.

## Protocol and interpretation

The holdout boundary is **2024-10-01**. Hyperparameters are selected only on earlier data, using five expanding folds and the one-standard-error rule. Linear preprocessing stays inside each fold; trees are unscaled. Holdout forecasts follow the kit's 63-observation refit cadence. The variance Lasso uses the complete feature matrix, the constant variance benchmark is the expanding past mean of squared returns, and GLD/ETH comparisons include constant-mean Student-t GARCH(1,1).

Portfolio forecast availability is merged in UTC onto each interval's previous SPY close. Seven-day asset variances are scaled by interval length, and weights are capped at 40%. The notebook includes both the kit's log-return approximation and an exact simple-return accounting supplement.

**Predictive FAIL states are results, not execution errors.** T3/V2 report actual forecasting hypotheses; separate rules check disclosure and selection integrity. The primary EUR/USD return result is timing-sensitive: an additional cross-asset delay removes its apparent advantage. No mean forecast sets portfolio weights. SHAP concerns a fitted model, not market causality; ETH has no predictor above the displayed permutation floor in this experiment.

The risk-model identities are selected using the full holdout as required by the brief. Their combined portfolio is therefore a retrospective course comparison, not independent forward validation. Declared daily-bar close conventions and asynchronous price labels remain limitations; timestamped forecast availability does not create synchronous execution prices. The input panel ends in September 2026 and is intentionally frozen.

## Reproduce offline

Use Python 3.14 with a Python kernel from the environment containing these dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name fintech-week6 --display-name "Python (FinTech Week 6)"
```

Open `W6_sprint_Tzalas.ipynb`, select **Python (FinTech Week 6)**, and use **Kernel → Restart & Run All** from the repository root. No price download is needed. The notebook creates the per-asset input CSVs and recalculates tables in `results/`; `results/.gitkeep` preserves the required output directory in a fresh clone. All figures are embedded in the executed notebook. Execution can take several minutes.

Course handouts are supplied separately by the lecturer. The prompt log discloses AI assistance. The notebook has been run from a fresh kernel and independently audited; unsupported predictive claims are retained as failures. The optional stretch is not claimed.
