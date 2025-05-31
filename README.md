# SSHbruteforsedetection_ml

SSH Brute-Force Attack Detection using Machine Learning
This project provides a machine learning-based solution for detecting SSH brute-force attacks. By analyzing system log data and training a classifier, it aims to automatically identify and distinguish between normal and malicious login behavior on a server.

### Overview
The notebook brute_force_attack_classifier.ipynb demonstrates the following:

Parsing and preprocessing server log files (auth.log).

Extracting useful features such as IP address frequency and timestamps.

Labeling data based on brute-force attempt patterns.

Training and evaluating a machine learning model to classify login attempts as either "attack" or "normal".

### Features
Log file parsing from auth.log.

Detection of suspicious IP activity based on failed login attempts.

Binary classification using machine learning algorithms.

Visualization of feature importance and classification performance.

### Technologies Used
- Python 3
- pandas
- scikit-learn
- matplotlib
- seaborn
- re (regex for log parsing)

### Model Overview
The classifier is trained on features extracted from the logs, such as:
- Number of failed attempts per IP
- Time intervals between attempts
- Unique IP addresses per session
After training, it can classify each login attempt as either (0 - Normal, 1 - Brute-force attack)

### Function Descriptions
```
read_auth_log(file_path)
```
Purpose: Reads the SSH authentication log file.
- Input: Path to auth.log.
- Output: List of log lines.

```
extract_failed_login_attempts(log_lines)
```
Purpose: Parses log lines to find failed SSH login attempts.
- Output: List of dictionaries with IPs, timestamps, and usernames.

```
create_feature_dataframe(attempts)
```
Purpose: Converts extracted attempts into a DataFrame with features.
- Output: pandas DataFrame with engineered features (e.g. number of attempts, time between attempts).

```
label_data(df)
```
Purpose: Adds labels to data (e.g. 0 for normal, 1 for attack) based on rules.
- Output: DataFrame with an added label column.

```
train_model(X, y)
```
Purpose: Trains a Random Forest classifier. 
- Output: Trained model and accuracy score.

```
evaluate_model(model, X_test, y_test)
```
Purpose: Evaluates the model using accuracy and confusion matrix.
- Output: Prints evaluation metrics and shows visualization.
  
```
plot_feature_importance(model, X)
```
Purpose: Visualizes the most important features used by the classifier.
