# E-Commerce Product Recommendation System

This is my machine learning project where I built a recommendation system using real e-commerce data. The idea is simple — based on how a user has behaved on the platform (what they viewed, carted or purchased), we predict which products they are most likely to buy next.

---

## About the Dataset

The dataset contains around 1 million user interaction events from an e-commerce platform for the month of November 2019. Each row is one event — a user either viewed, added to cart, or purchased a product.

---

## What I did

- **EDA** — explored the data to understand user behavior, popular categories, brands, price distribution and session patterns
- **Feature Engineering** — created meaningful features from raw events like user level stats, product level stats and category/brand conversion rates
- **Modelling** — compared multiple models (Logistic Regression, Random Forest, Gradient Boosting, XGBoost) and selected the best one based on ROC-AUC score
- **Hyperparameter Tuning** — used GridSearchCV with cross validation to improve model performance
- **Deployment** — built a simple Flask web app where you can enter a user ID and get top 10 product recommendations

---

## Project Structure

```
├── Notebooks/
│   ├── EDA.ipynb
│   ├── Feature_Engineering.ipynb
│   └── Modelling.ipynb
├── Models/
│   └── Final_model.pkl
├── templates/
│   └── index.html
├── app.py
└── requirements.txt
```

---

## Tech Used

Python, Pandas, Matplotlib, Seaborn, Scikit-learn, XGBoost, Flask

---

## Results

The final model gave a ROC-AUC score of 0.99 on the test set. Since the dataset is heavily imbalanced (97% views, only 3% purchases), ROC-AUC was used instead of accuracy as the evaluation metric.

---

## How to Run

1. Install the required libraries
```
pip install -r requirements.txt
```

2. Run the Flask app
```
python app.py
```

3. Open your browser and go to
```
http://localhost:5000
```

4. Enter any user ID from the dataset and get recommendations
