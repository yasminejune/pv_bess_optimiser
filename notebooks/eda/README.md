# Data Exploration and Planning

The aim of this EDA is to demonstrate an understanding of any dataset and my readiness to design an agentic data science system. The `eda.ipynb` does this in six steps:
1. Data loading and basic inspection
2. Exploratory data analysis (EDA) quality
3. Data cleaning and preprocessing decisions 
4. Dataset understanding and challenge identification 
5. Agentic planning proposal for the final project 
6. Creation of a summary `report.md` report and a `eda_summary.json` file

This script will form the basis for the final Agentic Data Scientist. The code will be added to the existing skeleton structure, while the Plan will guide the creation of the remaining elements for the Agent.

The remaining sections of this document will detail and explain the steps taken in the notebook.

## 1. Data Loading and Basic Inspection
In a first step, the agent loads the data, and performs basic inspection on it. Basic inspection includes feature type, label, and ID column detection, as well as an initial datasize assessment.

### 1.1 File loading 
The agent reads the file into a pandas DataFrame, as this facilitates the inspection and editing of the data.

### 1.2 Ensure Variable Type Consistency
Before detecting each variables feature type, the agent checks each object-type column for mixed types (e.g. a column containing both strings and numbers). If the majority of values are numeric, the column is converted to numeric and the non-numeric entries become NaN. Otherwise, the column is kept as string.

### 1.3 Feature type detection
The agent classifies each column as one of: `bool`, `numeric`, `datetime`, `text`, or `categorical`.
* Numeric covers all integer and float subtypes. The agent groups these types as they are cleaned and preprocessed in the same way. 
* Since datetime columns may not be in datetime format, the agent tests object columns for datetime parseability. If more than 80% of the values are parseable, the column is classified and converted to datetime. This percentage is arbitrary, and can be adjusted by the "Reflector" if needed.
* Object columns are split into text vs. categorical. If more than 5% of the values are unique and there are more than 10 unique values, the column is classified as text. This percentage is arbitrary, and can be adjusted by the "Reflector" if needed.

### 1.4 Detect target column
The agent determines whether the dataset contains ground truth labels. The agent determines a list of possible labels, based on whether they meet the following three criteria:
* **Exact name match** - If a single column matches any of the following labels, it is considered the target column: `["target", "label", "class", "y", "outcome"]`. If multiple columns match this criterion, the agent will do the following steps ONLY for these columns.
* **Disqualify** - The agent disqualifies columns that are of datetime or text format, as well as columns that contain hint words: `["id", "name", "date", "time", "timestamp", "index", "path", "file", "description", "comment", "note", "address", "phone", "email", "code", "key", "uuid"]`. These are words that - in my experience - are not used as labels. It is an assumption, and can be adjusted by the "Reflector" if needed. Additionally, categorical columns with less than 2 or more than 20 unique values are disqualified, as they are unlikely to be meaningful classification targets.
* **Score** - Remaining candidates are scored using semantic similarity, keyword bonuses, and position. The semantic similarity is calculated using a pre-trained 300-dimensional spaCy model (en_core_web_md) and compares the semantic similarity of each column-name token to each token parsed from the dataset filename. The keyword bonuses are given for the following keywords: `["data", "final", "result", "output", "response", "prediction"]`. The position bonus is given for the last column, since excel sheets tend to be organised as such. 

If any column has a score above 1.5, then the agent assumes that the highest scoring column is the target column. If no column has a score above 1.5, then the agent assumes that the dataset does not contain a target column. This threshold can be adjusted by the "Reflector" if needed.

If the agent identifies a target, it decides whether the task is a regression or classification task. If the target type is an object/category (categorical strings), OR numeric with <= 20 unique values (discrete classes), then the task is a classification task. If the target is numeric with > 20 unique values (continuous), then the task is a regression task. This threshold can be adjusted by the "Reflector" if needed.

### 1.5 Identifying ID column
The agent detects and drops columns that serve as row identifiers (e.g. names containing "id", "index", or "idx", or columns with sequential integer values). These carry no predictive information.

### 1.6 Determine dataset size
The agent flags small datasets (<1,000 rows) and high-dimensional datasets (>40 columns), as both require adjusted modelling strategies.

## 2. Exploratory Data Analysis
This section explores the data before conducting any data cleaning. The type of EDA depends largely on whether a target has been found (supervised learning), and whether the target is categorical or numeric:
* If supervised learning and classification task:
    * Class imbalance detection
    * Grouped countplot
    * Feature importance
* If supervised learning and regression task:
    * Countplot
    * Feature importance
* If unsupervised learning:
    * Countplot
* For any numeric feature:
    * Boxplot
    * Histogram
    * Correlation heatmap
* For any categorical feature:
    * Countplot

### 2.1 Class imbalance detection (classification only)
The agent checks whether the target classes are balanced and computes the imbalance ratio. If the imbalance ratio is above 3, the agent will flag the dataset as imbalanced. This has implications on the later modelling plans.

### 2.2 Boxplots (numeric features only)
The agent generates boxplots for each numeric feature to visualise the spread. It also flags potential outliers for later removal.

### 2.3 Histograms (numeric features only)
The agent generates histograms for each numeric feature to visualise the distribution.

### 2.4 Correlation heatmap and feature importance (numeric features only)
The agent generates a correlation heatmap for each numeric feature to visualise the correlation between features. If any two feature have a correlation coefficient above 0.90 in absolute value, then the agent will flag the dataset as having multicollinearity. This threshold can be adjusted by the "Reflector" if needed.
If a target was identified, then the agent ranks the importance of each numeric feature for the target.

### 2.5 Countplots (categorical features only)
If a categorical target was identified, the agent creates stacked countplots grouped by class. Otherwise, it creates a simple countplot for each categorical feature.

### 2.6 Summary statistics
The agent computes summary statistics for both numeric and categorical features - easily visualised in a table. The statistics include: ['mean', 'median', 'std', 'min', 'max', 'q1', 'q3', 'skewness', 'is_skewed', 'kurtosis', 'is_kurtosis'] - which once more highlights any non-normality and outliers.

## 3. Data cleaning and preprocessing
The agent prepares the data for model training. All cleaning steps up to §3.6 operate on the full `df`. After the train/test split (§3.7), remaining transformations are fit on training data only. This is to prevent any data leakage.

### 3.1 Remove single-variable columns
Single-variable columns provide no predictive information. The agent removes these.

### 3.2 Multicollinearity handling
If multicollinear pairs were flagged in section 2.4, the agent resolves them by dropping redundant features. It first builds an adjacency graph from all flagged pairs and identifies connected components (groups of mutually correlated features).

* Groups of 3+ features: The agent keeps the feature with the highest variance and drops the rest. Variance is used as the selection criterion because it indicates information spread across observations.
* Pairs (groups of 2): If a numeric target exists, the agent keeps the feature with the stronger absolute correlation to the target, as it carries more predictive signal. If both features have equal target correlation, it falls back to variance. If no numeric target is available (unsupervised or categorical target), variance is used directly.

All dropped columns are recorded, and the surviving numeric column list is updated accordingly.

### 3.3 Datetime conversion
Models cannot consume raw datetime values. The agent decomposes each into numeric components (year, month, day, day-of-week) and drops the original column.

### 3.4 LabelEncode target
If there is a target, and it is categorical, the agent label encodes it. This is necessary for the model to consume it.

### 3.5 Remove or address missing variables

The agent computes the percentage of missing values per column, tests whether missingness is systematic (MAR/MNAR) or random (MCAR), and then applies the appropriate imputation or removal strategy:
* **Missing Completely at Random (MCAR).** This type of missing value is unrelated to any variable, which means the agent can delete the row with missing values without biasing the model if there is sufficient data.   
* **Missing at Random (MAR).** Variables that are MAR depend on other observed variables, which means that if all rows with missing values were to be deleted, it would bias the results.
* **Missing Not at Random (MNAR).** Variables that are MNAR are dependent on themselves, which once more would bias the results if deleted. However, these are tricky to detect. The agent therefore flags that the dataset could have MNAR variables when addressing missing values.

The agent is designed to prefer deletion over imputation when feasible, as this introduces no artificial data, preserving the true distribution. It handles missing values using the following decision logic:

1. **>40% missing** — drop the column entirely, since this provides too little information to reliably impute.
2. **Assess missingness type** — test whether the missing pattern is systematic (MAR) or random (MCAR) based on the Mann Whitney U test (for numeric variables) or Chi-square test (for categorical variables). If more than 25% of the tested columns are significant at a standard 5% level, then the agent considers the missingness systematic. Otherwise, it is considered random.
3. **Random <10% + dataset >= 50 rows** — drop the affected rows, since this should not bias the model or remove too much information from it.
4. **Categorical column** — impute with the mode (most frequent value).
5. **Numeric + symmetric** — impute with the mean.
6. **Numeric + skewed** — impute with the median.

### 3.6 Remove duplicates
The agent removes exact duplicate rows and duplicate columns (identical content under different names) to avoid redundancy in the training data.

### 3.7 Train/test split
The agent splits the data into training and testing sets using a random split, with a default test size of 30%. From here on, the agent only uses the training set to fit models and the testing set to evaluate them to avoid any data leakage.

### 3.8 Remove outliers
The agent removes outliers from the training data based on the interquartile range (IQR). This test was chosen over a z-score test for its robustness on skewed distributions. 

### 3.9 Categorical encoding
The agent encodes the categorical features for the model to consume it. The strategy depends on the number of unique values in each column:
* **2 unique values** — binary label encoding (0/1).
* **3-5 unique values** — one-hot encoding (drop-first to avoid multicollinearity).
* **>5 unique values** — target-encode or ordinal-encode.

All encoders are fit on training data only and applied to both train and test.

### 3.10 Normality testing, log transformation and numeric scaling
For each numeric feature column, the agent conducts the following tests and transformations (excluding target and booleans): 
The feature is first tested for normality (if more than 20 values, since otherwise it is not reliable), using a D'Agostino-Pearson test to a N(0,1) distribution where the alpha is adjusted to the size of the dataset: If the agent faces less than 500 rows, it uses a standard alpha = 0.05. If there are between 500 to 5000, then a more liberal alpha = 0.01 is applied, and finally, if there are over 5000 rows, an alpha = 0.001 is applied. Any feature that passes the normality test can be scaled using StandardScaler. If the feature does not pass the normality test, the agent checks whether it is skewed at an absolute degree of > 0.5. If so, the agent applies a log transformation, and then rechecks for normality. For the log transformation, the agent ensures all values are positive, and then applies the log transformation. The log transformation is applied to the training and test data, but based on the training set only. This means that the test set could have values more negative than the training set, in which case, the transformation results in NaN values, which are flagged. If the feature is still not normal, the agent applies a MinMaxScaler(−1, 1) to ensure higher predictability in the transformed non-normal data, and models benefit from features that have all been scaled similarily.

### 3.11 Dimensionality reduction
The agent applies PCA-based dimensionality reduction if the number of features exceeds 1/2 of the number of rows. The statistical literature on high-dimensional inference generally identifies the problematic regime as p approaching n. The Marchenko-Pastur law from random matrix theory shows that sample covariance matrices become unreliable estimators as p/n approaches 1. A threshold of n/2 is a pragmatic early-warning trigger that catches genuinely high-dimensional problems without aggressively reducing feature spaces that are perfectly manageable. The agent builds a PCA model such that the components that remain still explain at least 95% of the variance. The agent then applies the PCA model to both the training and test data. 