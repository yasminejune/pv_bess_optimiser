# Agentic Data Scientist Report

> **Dataset:** `{dataset}`
> **Target:** `{target}`

---

## 1. Dataset Profile

| Property | Value |
|:--|:--|
| Rows | **{n_rows}** |
| Initial columns | **{initial_n_cols}** |
| Final columns | **{final_n_cols}** |
| Task | {task_type_display} |
| Learning | {learning} |
| Imbalance ratio | {imbalance_ratio_display} |

### 1.1 Initial Feature Types

- **Boolean** ({n_bool}): {bool_features}
- **Numeric** ({n_numeric}): {numeric_features}
- **Date** ({n_date}): {date_features}
- **Text** ({n_text}): {text_features}
- **Categorical** ({n_cat}): {cat_features}

### 1.2 Final Feature Types

- **One-Hot Encoded** ({n_one_hot}): {one_hot_features}
- **Embedded** ({n_embedded}): {embedded_features}
- **Scaled** ({n_scaled}): {scaled_features}
- **Unchanged** ({n_unchanged}): {unchanged_features}

---

## 2. Data Quality

| Check | Result |
|:--|:--|
| Missing values | {missing_values} |
| Systematic missingness | {systematic_missingness_summary} |
| Skewed columns | {skewed_columns} |
| Duplicates | {duplicates} |
| Outliers | {outlier_summary} |

---

## 3. Statistical Properties

{numeric_summary_table}

| Property | Details |
|:--|:--|
| Normality | {normality_summary} |
| Scaling applied | {scaling_summary} |
| Multicollinearity | {multicollinearity_summary} |
| Feature importance | {feature_importance_mi} |
| Train/Test split | {split_summary} |

---

## 4. Preprocessing

| Step | Details |
|:--|:--|
| Binary-encoded variables | {binary_encoded} |
| One-hot encoded variables | {one_hot_encoded} |
| Label-encoded target | {label_encoded} |
| Log-transformed variables | {log_transformed} |
| Standard scaling | {standard_scaling} |
| MinMax scaling | {minmax_scaling} |

---

## 5. Modelling Concerns

{modelling_concerns}

---

## 6. Processing Notes

{notes}

---

## 7. EDA Visualisations

### Boxplots — Numeric Features

![Boxplots — Numeric Features](boxplots_all_numeric.png)

### Histograms — Numeric Features

![Histograms — Numeric Features](histograms_all_numeric.png)

### Correlation Heatmap

![Correlation Heatmap](correlation_heatmap.png)

### Countplots — Categorical Features

![Countplots — Categorical Features](countplots_all_categorical.png)
