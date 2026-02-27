#  Airbnb NYC Market Analysis & Revenue Predictor

An end-to-end Machine Learning project to optimize rental revenue in the New York City market using **Logistic Regression** and **K-Means Clustering**.

---

##  1. Business Problem Statement

Hosts on Airbnb NYC face a complex challenge: how to balance **pricing** and **availability** to maximize revenue. The problem is two-fold:
1. **Revenue Prediction**: Predicting if a listing's configuration will result in "High Revenue" (above-median performance).
2. **Market Segmentation**: Identifying which market "segment" a listing belongs to so hosts can compare themselves against the right competitors.

---

##  2. Economic & Financial Concepts Applied

This project bridges the gap between raw data and economic theory:
- **Price Optimization**: Analyzing the trade-off between nightly rates and occupancy (availability) to reach the revenue frontier.
- **Trust Economy (Demand Proxy)**: Using "Number of Reviews" as a financial signal for customer trust and historical demand.
- **Market Segmentation**: Applying K-Means to identify distinct "tribes" of listings, allowing for segmented marketing and pricing strategies.
- **Risk Analysis**: Predicting "Low Revenue" performance to help hosts mitigate the risk of vacant or underpriced properties.

---

##  3. AI & Data Science Techniques

The system implements a structured ML pipeline:
- **StandardScaler**: Normalizes feature scales for fair comparison.
- **K-Means Clustering (Unsupervised)**: Segments the NYC market into 3 distinct operational clusters based on price-availability-demand dynamics.
- **Logistic Regression (Supervised)**: A classification model that achieves high accuracy in predicting revenue performance.
- **Data Preprocessing**: Handling ~102k records, removing outliers, and engineering the `high_revenue` target.

---

##  4. Project Deliverables

###  Google Colab / Jupyter Notebooks
- [Data_Inspection.ipynb](notebook/Data_Inspection.ipynb): Advanced cleaning and preprocessing.
- [eda_analysis.ipynb](notebook/eda_analysis.ipynb): Exploratory Data Analysis and visual insights.
- [modeling.ipynb](notebook/modeling.ipynb): K-Means & Logistic Regression implementation with business interpretation.

### Streamlit Deployment
The app is fully functional with an interactive dashboard and real-time inference.

**[Launch Predictor Dashboard]**
> `streamlit run app/app.py`

###  Model Output Screenshots

#### Interpretable Prediction Interface
![Dashboard Overview](/Users/rajkoli/.gemini/antigravity/brain/22e133f1-9518-47ef-9c15-778612ae8a70/streamlit_analysis_results_1772168246069.png)

#### Market Position & Segmentation
![Market Analysis](/Users/rajkoli/.gemini/antigravity/brain/22e133f1-9518-47ef-9c15-778612ae8a70/streamlit_app_prediction_result_1772167562735.png)

---

##  5. Dataset Information
- **Source**: [Kaggle - Airbnb NYC Open Data](https://www.kaggle.com/datasets/arianazmoudeh/airbnbopendata)

---

##  6. Installation & Execution

1. **Clone Repo**: `git clone https://github.com/Rajkoli145/Airbnb-NYC-Market-Analysis.git`
2. **Environment**: `python -m venv .venv && source .venv/bin/activate`
3. **Dependencies**: `pip install -r requirements.txt`
4. **Run App**: `.venv/bin/streamlit run app/app.py`

---

##  7. Team Contributions
- **Data Cleaning**: [Rajkoli145](https://github.com/Rajkoli145)
- **Feature Engineering**: [Harshal7506](https://github.com/Harshal7506)
- **Modeling & Deployment**: [Parth](https://github.com/Indianworldruler)

---
*Developed for the AI in Finance/Business Mini-Project.*
