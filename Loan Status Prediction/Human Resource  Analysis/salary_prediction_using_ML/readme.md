#  Salary Prediction using Machine Learning

The salary prediction project utilizes machine learning models to forecast salaries based on various input parameters such as age, gender, education level, job title, and years of experience. The goal is to provide users with accurate salary estimates derived from a trained predictive model.

## Data Set

The below csv dataset from kaggle is used as reference which contains nearly 300+ rows (salary_prediction.ipynb) on which processing is performed to obtained a  processed data , all this processing is performed in first notebook (salary-prediction.ipynb) file.

The dataset link is are as follows :-
https://www.kaggle.com/datasets/rkiattisak/salaly-prediction-for-beginer

on this dataset, below processing are performed :
1) featue scaling and column reinitialization
2) errors and outliers removal using box plot
3) remove na,missing values , regularization etc
4) Drop duplicates , normalization , column dropping

(all this works ar depicted in salary_prediction file)

The model is trained on processed data after data processing and feature engineering  and all works associated with it are depicted in salary_prediction.ipynb file.

## Methodology

The project follows the below structured methodology ranging from data preprocessing pipeline to feature engineering model training, evaluation and deployment :-

1. **Data Preprocessing and feature enginnering**

2. **Exploratory Data Analysis (EDA)**:
    after Data preprocessing the next step is Exploratory  data analysis using different plotting libraries like matplotlib,pandas,seaborn and plotly.following plots were plotted in this step:-
    1) Pie charts
    2) violen plots
    3) box plot of numerical features
    4) count plots
    5) histogram
    6) model comparison graphs
    7) Actual vs predicted regression lines
    (refer images folder for this images and graph observation)


4. **Model Training and evaluation**: 
     The four machine learning model random forest ,XgBoost ,multiple linear regression, gradient boosting machine are selected for model training over the inputed processed data:
     random forest accuracy : 90 %
     GBM accuracy : 91 %
     XGBOOST accuracy : 92 %
     Multiple linear regression accuracy : 89 %

     The 10 fold cross validation is then performed on XGBOOST model to obtained a final average cross validated accuracy of 86 % .

     This  XGBOOST model is then loaded into streamlit application after installing and using joblib library.

5. **Inference**: 
      Deployed the model with the help streamlit web application to predict the estimated salary of the individual.

## Libraries Used

1. **Joblib**: For downloading the KNN model
2. **Scikit learn**: For machine learning processing  and operations
3. **Matplotlib**: For plotting and visualizing the detection results.
4. **Pandas**: For Data manipulation.
5. **NumPy**: For efficient numerical operations.
6. **Seaborn** : for advanced data visualizations
7. **plotly** : for 3D data visualizations .
8. **Streamlit** : for creating gui of the web application.
9. **requests** : requests for creating Htttp requests

## How to Use

1. **Clone the Repository**: 
    ```sh
    git clone url_to_this_repository
    ```

2. **Install Dependencies**: 
    ```sh
    pip install -r requirements.txt
    ```


3. **Run the Model**: 
    ```python
    streamlit run app.py
    ```

4. **View Results**:  The script will allow you to predict or estimate the salary of a men/women based on various factors like age , year of experience ,education level and post held at office using four different machine learning techniques. The final web application model using streamlit package from python is fitted with the model having highest accuracy (XGBOOST 92%) to predict the salary of different users.

## Demo 
