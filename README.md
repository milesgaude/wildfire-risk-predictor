# 🔥 Wildfire Risk Predictor

A machine learning web application that predicts wildfire risk levels across the United States based on location, time of year, and environmental conditions. Built with a Random Forest classifier trained on 1.88 million historical wildfire records.

**[→ Try the Live App](https://wildfire-risk-predictor-h6menbn8yhwkycpnjt8z2a.streamlit.app/)**

---

## Overview

Wildfires cause billions of dollars in damage annually across the US. This tool uses historical data from the US Forest Service (1992–2015) to predict wildfire risk levels - Low, Medium, High, or Extreme — based on geographic and temporal features.

Users input a state, coordinates, time of year, and fire cause, and the model returns a risk prediction with a confidence breakdown across all four levels.

---

## How It Works

**Data Pipeline**
- Loaded 1.88 million wildfire records from the US Forest Service FPA-FOD dataset
- Selected key features: location (state, latitude, longitude), time (year, day of year), and fire cause
- Engineered additional features: month, fire season flag, western US indicator, and latitude bands
- Encoded categorical variables using Label Encoding

**Model**
- Algorithm: Random Forest Classifier (50 trees, max depth 15)
- Addressed class imbalance using custom class weights - High fires weighted 10x, Extreme fires weighted 25x
- Trained on 400,000 samples, tested on 100,000 holdout samples
- Accuracy: 71.6%

**Feature Importance**

The model identified these as the most predictive features:

| Feature | Importance |
|---|---|
| Longitude | 0.284 |
| Latitude | 0.236 |
| Day of Year | 0.147 |
| Fire Year | 0.108 |
| Fire Cause | 0.067 |
| State | 0.048 |

Geographic location (longitude + latitude) accounts for over 50% of the model's decisions, reflecting the significant difference in wildfire behavior between western and eastern US.

---

## Tech Stack

- **Python** — core language
- **Pandas & NumPy** — data manipulation and processing
- **Scikit-learn** — Random Forest model, preprocessing, evaluation
- **Streamlit** — interactive web application
- **Plotly** — confidence visualization charts
- **Joblib** — model serialization

---

## Key Takeaways

- **Class imbalance is real**: 97% of fires are Low or Medium. Without custom class weights the model ignored dangerous fires entirely. Applying weights of 10x for High and 25x for Extreme significantly improved detection of severe fires.
- **Feature engineering matters**: Adding derived features like fire season flags and regional indicators improved model interpretability, though the raw location and temporal features carried the most predictive power.
- **Location dominates**: Longitude alone was the single strongest predictor, aligning with domain knowledge that western US fires behave fundamentally differently due to climate, terrain, and vegetation.

---

## Limitations & Future Improvements

- **No weather data**: Temperature, humidity, wind speed, and drought indices are the strongest real-world predictors of catastrophic wildfires but are absent from this dataset. Integrating NOAA weather APIs would likely push accuracy above 85%.
- **Historical data only**: The dataset covers 1992–2015. A production model would need continuous data updates.
- **Rare event detection**: Despite class weighting, precision on Extreme fires remains low (0.20). Techniques like ensemble stacking or gradient boosting could improve this.

---

## Data Source

[US Forest Service Fire Program Analysis Fire-Occurrence Database (FPA-FOD)](https://www.kaggle.com/datasets/rtatman/188-million-us-wildfires) — 1.88 million georeferenced wildfire records from 1992–2015.

---

## Run Locally

```bash
git clone https://github.com/milesgaude/wildfire-risk-predictor.git
cd wildfire-risk-predictor
pip install -r requirements.txt
streamlit run app.py
```

---

## Author

**Miles Gaude** — [GitHub](https://github.com/milesgaude) | [LinkedIn](https://linkedin.com/in/milesgaude)
