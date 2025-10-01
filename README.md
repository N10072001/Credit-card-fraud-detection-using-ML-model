Credit Card Fraud Detection using Machine Learning

📌 Project Overview

Credit card fraud is a significant issue in today’s digital world, causing financial losses to both individuals and institutions. This project focuses on detecting fraudulent transactions using Machine Learning (ML) models.

The dataset is highly imbalanced, with fraud cases being much fewer than non-fraudulent ones. To overcome this, various data balancing techniques and hyperparameter tuning methods are applied to build robust models.


---

📂 Project Structure

The repository contains the following files and notebooks:

README.md → Documentation of the project.

balanced_dataset_oversampling.ipynb → Fraud detection using oversampling techniques.

balanced_dataset_undersampling.ipynb → Fraud detection using undersampling methods.

credit_card_fraud_detection_using_ensemble.ipynb → Fraud detection using ensemble models (Random Forest, XGBoost, etc.).

hybrid_under_sampling.ipynb → Hybrid sampling techniques for class balancing.

hyperparameter_tuning.ipynb → Hyperparameter tuning using manual grid search.

hyperparameter_tuning_RandomizedSearchCV.ipynb → Hyperparameter tuning with RandomizedSearchCV.

hyperparameter_tuning_BayesianOptimization.ipynb → Hyperparameter tuning using Bayesian Optimization.

unbalanced_dataset.ipynb → Model training on the original unbalanced dataset for comparison.



---

⚙ Technologies Used

Python 3.x

Jupyter Notebook

Libraries:

NumPy

Pandas

Scikit-learn

Matplotlib & Seaborn (for visualization)

Imbalanced-learn (SMOTE, undersampling, etc.)

XGBoost / LightGBM (for ensemble learning)




---

🚀 Workflow

1. Data Preprocessing

Handling missing values

Feature scaling

Splitting dataset into train & test sets



2. Handling Imbalanced Dataset

Oversampling (SMOTE)

Undersampling

Hybrid approaches



3. Model Training

Logistic Regression

Decision Tree

Random Forest

XGBoost / LightGBM

Ensemble methods



4. Hyperparameter Tuning

Manual tuning

RandomizedSearchCV

Bayesian Optimization



5. Evaluation Metrics

Accuracy

Precision, Recall, F1-Score

ROC-AUC Curve





---

📊 Results

Models trained on balanced datasets perform significantly better than those trained on unbalanced data.

Ensemble models (Random Forest, XGBoost) achieved the highest ROC-AUC scores.

Hyperparameter tuning further improved model performance.



---

📌 Key Insights

Fraud detection is a highly imbalanced classification problem.

Balancing techniques + Ensemble learning yield the most reliable results.

Precision and Recall are more important than accuracy in fraud detection.



---

💡 Future Improvements

Use Deep Learning models (ANN, LSTM) for fraud detection.

Deploy the best model using Flask / FastAPI.

Build a real-time fraud detection system.



---

📜 License

This project is open-source and available under the MIT License.


---

