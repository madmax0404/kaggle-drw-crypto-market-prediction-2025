# DRW — Crypto Market Prediction

Kaggle Competition
[https://www.kaggle.com/competitions/drw-crypto-market-prediction/overview](https://www.kaggle.com/competitions/drw-crypto-market-prediction/overview)

---

## Project Overview

I competed in Kaggle’s crypto trading prediction challenge and finished in the **top 11%**.

Using anonymized trade features (X1–X780) and the target label, I predicted future market movement. I applied extensive **feature engineering** (nonlinear transforms, row-wise aggregations, interaction features), performed **feature selection** with **LightGBM + SHAP**, and maximized performance via **Optuna** hyperparameter tuning. The final predictor **ensembles** **LightGBM**, **XGBoost**, and **CatBoost**.

The competition metric was the **Pearson correlation coefficient**. My final score was **0.08230**. For context, the **average score of the top 10 teams** was **0.11433** (see leaderboard below), suggesting a ceiling with the public feature set. Still, the exercise yielded solid practical insights by pushing multiple ML techniques end-to-end.

Leaderboard: [https://www.kaggle.com/competitions/drw-crypto-market-prediction/leaderboard](https://www.kaggle.com/competitions/drw-crypto-market-prediction/leaderboard)

---

## Tech Stack

* **Language:** Python
* **ML Models:** XGBoost, LightGBM, CatBoost
* **Hyperparameter Tuning:** Optuna
* **Feature Selection:** SHAP
* **Data Manipulation & Analysis:** pandas, NumPy, itertools
* **Visualization:** Matplotlib, Seaborn, Optuna Dashboard
* **Cross-Validation:** `TimeSeriesSplit` (scikit-learn)
* **OS:** Linux (Ubuntu Desktop 24.04 LTS)
* **IDE:** VS Code, Jupyter Notebook

---

## Problem

Crypto markets are fast-moving and noisy; price dynamics depend on liquidity, order flow, investor sentiment, and structural inefficiencies. Extracting stable predictive signals is therefore challenging.

This competition’s goal is to build a model that predicts **short-term crypto futures price movement**, combining DRW’s production-style data characteristics with public market features. Participants must integrate hundreds of high-dimensional features (X1–X780) and volume-related indicators to generate a robust signal that captures directionality under heavy noise.

---

## Dataset

Data provided by **DRW Trading** via Kaggle ([https://www.kaggle.com/competitions/drw-crypto-market-prediction/data](https://www.kaggle.com/competitions/drw-crypto-market-prediction/data)).

**`train.parquet`** — Historical market data and labels.

| Column          | Description                                                                  |
| --------------- | ---------------------------------------------------------------------------- |
| `timestamp`     | Minute-level timestamp index for each row.                                   |
| `bid_qty`       | Total quantity buyers are willing to purchase at the best bid at that time.  |
| `ask_qty`       | Total quantity sellers are willing to sell at the best ask at that time.     |
| `buy_qty`       | Total executed buy quantity at the best ask during the minute.               |
| `sell_qty`      | Total executed sell quantity at the best bid during the minute.              |
| `volume`        | Total traded volume during the minute.                                       |
| `X_{1,...,780}` | Anonymized set of proprietary market features.                               |
| `label`         | Anonymized target representing future market price movement to be predicted. |

**`test.parquet`** — Same feature structure as train but with differences:

* **`timestamp`** masked and shuffled, replaced with unique IDs (to prevent look-ahead).
* **`label`** set to 0 for all rows.

**`sample_submission.csv`** — Template showing the required submission format (must match rows and structure to be valid).

---

## Methodology & Approach

I designed a comprehensive ML pipeline covering preprocessing, feature engineering, modeling, tuning, and ensembling.

### 1) Preprocessing & Feature Pruning

* Remove constant/duplicate/low-variance columns
* For pairs with **Pearson > 0.9**, drop one of the features

### 2) Feature Selection & Engineering

* **SHAP + LightGBM + TimeSeriesSplit** for importance-guided selection
* Center feature engineering around the **top 20** features (row-wise, nonlinear, interactions)
* Final selection of **top 200** features
  *(See `/drw2025/1_feature_engineering_and_selection_lgb.ipynb` for details.)*

![SHAP feature importance](images/feature_importance.png)
*Figure 1. SHAP-based feature importance*

### 3) Model Training & Validation

* Gradient-boosted trees: **XGBoost**, **LightGBM**, **CatBoost**
* Validation with **`TimeSeriesSplit`**; monitor **RMSE** during training
* Regularization and **early stopping** to mitigate overfitting
* **Train loss:** RMSE; **Validation/Tuning metric:** custom **Pearson** (to align with LB metric)

![Learning Curve](images/learning_curve.png)
*Figure 2. Training & validation learning curves*

### 4) Hyperparameter Tuning

* Automated search with **Optuna**
* Treat **feature count** as a hyperparameter to jointly find the best subset

![Optuna result](images/LGB_Optuna_20250720_1.png)
*Figure 3. LightGBM + Optuna hyperparameter tuning (Optuna Dashboard)*

### 5) Ensemble (Blending)

* Weighted average of model predictions based on CV performance
* Improvement over best single CV (**0.1509**) by **+0.0067 absolute** (**+4.4%** relative)

![Blending Result](images/blending_result.png)
*Figure 4. Ensemble results*

| Model        | CV (Pearson) | LB (Pearson) |
| ------------ | ------------ | ------------ |
| LightGBM     | 0.1169       | —            |
| XGBoost      | 0.1315       | —            |
| CatBoost     | 0.1509       | —            |
| **Blending** | **0.1576**   | **0.0823**   |

*LB for single models was not computed due to submission/management cost; the final submission used the blended model.*

---

## Results & Key Observations

* **Final Pearson:** **0.08230** on the public test set (**top 11%**). Given the **top-10 average = 0.11433**, the public feature set appears to impose a limit.
* **[Key Observation 1]** A large CV–LB gap suggests **generalization issues** and room for improvement.
* **[Key Observation 2]** With non-anonymized (or richer) data, more advanced feature engineering would likely yield further gains.
* Despite using `TimeSeriesSplit`, the CV→LB drop hints at **leakage risk** in validation or preprocessing. Next iteration: **walk-forward CV + embargo**, strictly in-fold preprocessing/selection, and Pearson-based early stopping.

```
Split1: [ Train .......... ][ Val1 ]
Split2: [ Train .................. ][ Val2 ]
Split3: [ Train .......................... ][ Val3 ]
(Next: consider an n-step embargo near split boundaries)
```

---

## Conclusion & Next Steps

This project demonstrates a complete time-series ML workflow for **crypto price movement prediction**, highlighting both strengths and pitfalls of modeling under heavy noise.

**Planned improvements:**

* Integrate external data (on-chain, sentiment) and data augmentation
* Strengthen **XAI** for production (feature attribution, risk monitoring)
* Investigate the **CV (0.1576) → LB (0.0823)** gap (potential overfitting)
* Explore stronger **ensembles**
* Design a more robust **CV strategy**

---

## How to Reproduce

*Reproducibility:* `seed=42` fixed; identical seeds for all splits/tuning. See `requirements.txt` for the environment.

1. **Clone the repository**

   ```bash
   git clone https://github.com/madmax0404/kaggle-drw-crypto-market-prediction-2025.git
   cd kaggle-drw-crypto-market-prediction-2025
   ```

2. **Download the dataset**

   * Join the competition: **[DRW — Crypto Market Prediction](https://www.kaggle.com/competitions/drw-crypto-market-prediction/)**
   * Download and place the data in the appropriate directory.

3. **Create a virtual environment & install dependencies**

   ```bash
   conda create -n DRW python=3.12  # or venv
   conda activate DRW
   pip install -r requirements.txt
   ```

4. **Run Jupyter Notebook**

   ```bash
   jupyter notebook drw2025
   ```

   Follow the notebooks to run preprocessing, training, and evaluation.

---

## Acknowledgements

Thanks to **DRW Trading** and **Kaggle** for the dataset and competition platform.

This project was supported by: **LightGBM**, **XGBoost**, **CatBoost**, **Optuna**, **SHAP**, **scikit-learn**, **pandas**, **numpy**, **matplotlib**, **seaborn**, **Jupyter**.

All data usage complies with the competition rules and licenses.

---

## License

Code © 2025 **Jongyun Han (Max)**. Released under the **MIT License**.
See the LICENSE file for details.

**Note:** Datasets are **NOT** redistributed in this repository.
Please download them from the official Kaggle competition page and comply with the competition rules/EULA.
