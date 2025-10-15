# ⚾ Swing Probability in Baseball  
**Author:** Danilo Arruda, PhD  

---

## 📘 Project Overview
This project develops a **Swing Probability Model** in baseball using historical data from **three Major League Baseball (MLB) seasons**.  
The model predicts the likelihood that a batter will swing at a given pitch, based on pitch characteristics, count context, and player information.  
Beyond prediction, the project also explores **feature importance and interpretability** using SHAP values to understand what drives swing decisions.

---

## 🧠 Motivation
Understanding swing behavior is valuable for:
- **Teams and coaches**, to tailor pitching strategies and player matchups.  
- **Analysts and scouts**, to identify players with disciplined or aggressive swing tendencies.  
- **Sports scientists**, to study decision-making and motor behavior under time constraints.

---

## 📂 Table of Contents
1. [Project and Data Description](#project-and-data-description)  
2. [Data Preparation](#data-preparation)  
3. [Modeling](#modeling)  
4. [Model Calibration](#model-calibration)  
5. [Model Diagnostics](#model-diagnostics)  
6. [Predicting Swing Probability for Year 3](#predicting-swing-probability-for-year-3)  
7. [Feature Exploration](#feature-exploration)  
8. [How to Run the Notebook](#how-to-run-the-notebook)  
9. [Dependencies](#dependencies)  
10. [Author](#author)

---

## ⚙️ Project and Data Description
The dataset is retrieved using the **`pybaseball`** package and includes pitch-level information from three MLB seasons:
- 2022 season (April–October)
- 2023 season (March–October)
- 2024 season (March–June)

Each pitch includes detailed features such as:
- **Pitch characteristics**: type, velocity, spin rate, spin axis, and release position.  
- **Pitch trajectory**: movement (pfx_x, pfx_z), plate location (plate_x, plate_z), and strike zone metrics (sz_top, sz_bot).  
- **Game context**: balls, strikes, outs, inning, and score.  
- **Player information**: batter, pitcher, handedness (stand and p_throws).

The **target variable** is a binary indicator representing whether the batter **swung (1)** or **did not swing (0)**.

---

## 🧹 Data Preparation
Key preprocessing steps:
- Data fetched from Statcast via `pybaseball.statcast()`.
- Selection of relevant features for swing behavior.
- Handling missing values (rows with few NaNs dropped).
- Creation of the binary target variable (`swing`).
- Conversion of categorical features into dummy variables for modeling.

---

## 🤖 Modeling
Four classification models were initially trained and compared:
- **LightGBM**
- **Gradient Boosting**
- **Decision Tree**
- **Random Forest**

LightGBM achieved the best performance (≈ **85% accuracy**) and was selected for further optimization using **Bayesian hyperparameter tuning** via `skopt.BayesSearchCV`.

---

## 🔧 Model Calibration
Since the final goal is to predict **probabilities**, model calibration was performed using **isotonic regression** with cross-validation (`cv=20`) to improve probability reliability.  
The calibrated model was then used to evaluate accuracy and Brier score.

**Performance summary:**
- Accuracy: ~0.85  
- Brier Score: ~0.106  
- ROC-AUC: High discrimination between swing/no-swing classes.

---

## 📊 Model Diagnostics
- **Confusion Matrix** and **Classification Report** confirmed balanced accuracy between classes.  
- **ROC Curve** showed strong separability between swing outcomes.  
- **Calibration Curve** verified that predicted probabilities align closely with observed outcomes.

---

## 🧩 Predicting Swing Probability for Year 3
The trained and calibrated LightGBM model was applied to **2024 (Year 3)** data to predict swing probabilities.  
Predictions correctly identified actual swing decisions **~80% of the time**, consistent with training performance.

A threshold-based evaluation (0.5 cutoff) illustrated how often model predictions matched real outcomes.

---

## 🔍 Feature Exploration
To interpret the model, **SHAP (SHapley Additive exPlanations)** was used.

### SHAP Beeswarm Plot
- **Strikes:** More strikes increase swing probability.  
- **Release speed:** Slower pitches slightly increase swing tendency.  
- **Strike zone top/bottom (sz_top, sz_bot):** Influence batter’s decision boundaries.  
- **Pitch type (e.g., sinkers)** tends to reduce swing probability.

### SHAP Dependence Plots
- **Strikes × Balls:** Swing probability increases sharply at two strikes, moderated by the number of balls.  
- **Release speed × Plate height:** Relationship is nonlinear (U-shaped); mid-range velocities increase swing likelihood.

These insights can guide **coaching strategies**, highlighting how pitch type, speed, and count context jointly shape swing behavior.

---

## ▶️ How to Run the Notebook
1. Clone this repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/swing-probability-baseball.git
   cd swing-probability-baseball
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
4. Open and execute the notebook:
   ```
   swing_probability.ipynb
   ```

---

## 📦 Dependencies
Key Python packages:
- `pandas`, `numpy`, `matplotlib`
- `scikit-learn`
- `lightgbm`
- `shap`
- `skopt`
- `pybaseball`
- `tqdm`

Make sure your environment supports **Python ≥ 3.9**.

---

## 👨‍🔬 Author
**Danilo Arruda, PhD**  
- Exercise Science researcher specializing in **motor behavior** and **performance analytics**.  
- Experienced in **data analysis**, **machine learning**, and **sports biomechanics**.  

📫 [LinkedIn](https://www.linkedin.com/in/danilo-arruda/) • [Google Scholar](https://scholar.google.com/citations?user=On_20uoAAAAJ&hl=en)

---

## 🏁 Summary
This notebook demonstrates a **data-driven approach to understanding swing behavior** in baseball using machine learning.  
By combining model calibration, probabilistic interpretation, and explainable AI, it provides actionable insights that can inform both **coaching decisions** and **player development** strategies.

---

> “Whether you think you can or think you can't, either way you are right.” — *Henry Ford*
