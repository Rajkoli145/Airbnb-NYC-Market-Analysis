#  Airbnb NYC Market Analysis

This project analyzes Airbnb listings in New York City to understand pricing behavior and predict whether a listing will generate high revenue.

The project is built step by step — starting from raw data cleaning, moving into feature engineering, and preparing a dataset ready for machine learning modeling.

This is a structured ML pipeline, not just a notebook experiment.

---

##  Project Objective

The main goals of this project are:

- Clean and preprocess Airbnb NYC dataset  
- Remove outliers and handle missing values  
- Create meaningful features from raw data  
- Build a dataset ready for machine learning  
- Predict whether a listing generates **high revenue**

---

##  Dataset Information

- Source: Kaggle – Airbnb Open Data  
- Raw dataset size: ~102,000 listings  
- Cleaned dataset size: **69,477 listings**  
- Final engineered dataset size: **69,477 rows × 29 columns**

---

##  Project Workflow

### 1. Data Cleaning

Notebook: `notebook/Data_Inspection.ipynb`

Steps performed:

- Removed unnecessary columns (`id`, `host name`, `license`, etc.)
- Converted `price` and `service fee` to numeric format
- Removed extreme outliers:
  - price > 1000  
  - minimum nights > 365  
- Removed invalid values (price ≤ 0, nights ≤ 0)
- Handled missing values
- Dropped rows with null values
- Exported cleaned dataset to:
  `data/processed/airbnb_cleaned.csv`
  
Final cleaned dataset shape:
`(69477, 20)`

---

### 2️⃣ Feature Engineering

Notebook: `notebook/feature_engineering.ipynb`

Steps performed:

- Created new feature:
  `high_revenue = 1 if revenue > median_revenue
high_revenue = 0 otherwise`

- Applied one-hot encoding to:
- `room type`
- `neighbourhood group`
- `cancellation_policy`

- Exported final dataset to:
  `data/processed/airbnb_featured.csv`

Final dataset shape:
`(69477, 29)`

Target distribution:

- 0 → 34,747  
- 1 → 34,730  

The dataset is well balanced and ready for classification models.

---

### 3️⃣ Modeling (Next Phase)

The engineered dataset is prepared for:

- Logistic Regression  
- KMeans Clustering  
- Model evaluation and comparison  

This phase focuses on predicting whether a listing will generate high revenue.

---

## 📂 Project Structure
```Airbnb-NYC-Market-Analysis/
│
├── data/
│   ├── raw/
│   │   └── airbnb_raw.csv
│   └── processed/
│       ├── airbnb_cleaned.csv
│       └── airbnb_featured.csv
│
├── notebook/
│   ├── Data_Inspection.ipynb
│   └── feature_engineering.ipynb
│
├── images/
├── requirements.txt
└── README.md
```

---

##  Installation & Setup

### 1. Clone the repository
```
git clone https://github.com//Airbnb-NYC-Market-Analysis.git

```

```
cd Airbnb-NYC-Market-Analysis

```

### 2. Create virtual environment
```
python -m venv myenv

```
```
source myenv/bin/activate

```
### 3. Install dependencies
```
pip install -r requirements.txt

```
### 4. Run Jupyter Notebook
```
jupyter notebook

```

Open the notebooks inside the `notebook/` folder.

---

##  Key Learning Outcomes

- Real-world data cleaning process  
- Handling missing values and outliers  
- Feature engineering for machine learning  
- Creating classification targets  
- Preparing structured datasets for modeling  
- Team-based Git workflow using branches and pull requests  

---

##  Team Contributions

- **Data Cleaning:** Raj  
- **Feature Engineering:** Harshal  
- **Modeling:** Parth (Next Phase)

---

##  Final Note

This project focuses on building a strong foundation in data preprocessing and feature engineering before applying machine learning models.

Clean data → Good features → Better models.

That is the approach followed throughout this project.

