# Customer Churn Prediction in Telecom Dataset with SVM

### This project explores customer churn in a telecom dataset using Support Vector Machines (SVM).

## Project Overview
The project involves data exploration, cleaning, visualization, feature engineering, and model building using SVM to predict customer churn.

 ## Data Exploration and Cleaning
**1. Libraries:** Imports pandas, seaborn, NumPy, matplotlib, and scikit-learn libraries for data manipulation, visualization, and modeling.\
 **2. Data Loading:** Reads the CSV file containing customer data.\
**3. Data Cleaning:** Handles missing values, removes customer IDs, and converts the target variable 'Churn' to numerical format.\
**4. Feature Engineering:** Creates dummy variables for categorical features to prepare data for SVM.\
**5. Correlation Analysis:** Analyzes the correlation between 'Churn' and other features to identify potentially relevant factors.
## Data Visualization
The project utilizes Seaborn plots to visualize various aspects of the data:

* Distribution of features like gender, senior citizens, dependents, partners, contract types, tenure, monthly charges, total charges, etc.
* Relationship between features and churn rate.
* Distribution of charges for churning and non-churning customers.
## Modeling with SVM
**1. Data Split:** Splits the data into training and testing sets for model evaluation.\
**2.Scaling (Not Included):** It's recommended to scale the features before feeding them to the SVM model for better performance.\
**3. Model Training:** Trains an SVM model using sklearn.svm.SVC.
**4. Evaluation (Not Included):** Evaluates the model's performance using metrics like accuracy score. You can explore other metrics like precision, recall, F1-score, or ROC AUC.
## Further Exploration
* **Hyperparameter Tuning:** Experiment with different SVM hyperparameters like kernel functions and regularization parameters.
* **Feature Importance:** Analyze which features contribute most to the SVM model's predictions.
* **Model Comparison:** Compare the performance of SVM with other classification algorithms.
* **Imbalanced Class Handling:** If the churn rate is skewed, consider techniques to address class imbalance.
## Dependencies
* pandas
* seaborn
* NumPy
* matplotlib
* scikit-learn
## Getting Started
1. Clone this repository.
2. Install required libraries using pip install -r requirements.txt.
3. Run the Python script (e.g., main.py) to execute the analysis and model training.
This README provides a high-level overview of the project. Refer to the code for detailed implementation.