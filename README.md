# Movie Genre Classification Using Traditional Machine Learning

A machine learning pipeline that classifies movie plot summaries into **Comedy**, **Drama**, or **Documentary**, comparing three traditional models to see which one handles raw text best.

## Dataset

Started from a raw catalog of 108,414 titles across many genres. After removing rows with missing values or an unknown release year, and keeping only the three genres this project targets (Comedy, Drama, Documentary), **63,687 titles** remained — this is the set actually used for feature engineering, training, and evaluation.

## Approach

Each title's name and plot description are combined into one text field. Three models are compared on the same train/validation/test split (70/15/15, stratified by genre):

| Model | Text features | Test accuracy | Macro F1 |
|---|---|---|---|
| **CatBoost** | native text handling (no manual vectorization) + year/length features | **82.9%** (83.25% avg. over 5-fold cross-validation) | 0.81 |
| Logistic Regression | TF-IDF, 1,000 features | 77.9% | 0.75 |
| XGBoost | TF-IDF, 1,000 features + year/length features | 77.1% | 0.73 |

**CatBoost wins because it isn't limited to a fixed-size bag-of-words.** Logistic Regression and XGBoost both depend on a 1,000-feature TF-IDF vector, which caps how much of the plot text they can actually use. CatBoost's built-in text-feature handling reads the full description directly, which is the main reason it comes out roughly 5 points ahead of the other two.

## Results by genre (CatBoost, held-out test set)

| Genre | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Documentary | 0.90 | 0.92 | 0.91 | 3,770 |
| Drama | 0.81 | 0.81 | 0.81 | 3,692 |
| Comedy | 0.74 | 0.69 | 0.71 | 2,091 |

Comedy is the hardest of the three to classify — it's also the smallest class by a wide margin, which is the most likely reason. Documentary, the largest class, is the easiest.

## Files

- `movie_classification.ipynb` — the full pipeline: loading, cleaning, feature engineering, and training/evaluation for all three models.

## How to run

```bash
pip install pandas numpy scikit-learn catboost xgboost
```

Open `movie_classification.ipynb` and run it top to bottom. The notebook keeps its executed outputs, so the results above can be reviewed without retraining anything.
