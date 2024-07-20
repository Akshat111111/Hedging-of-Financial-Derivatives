#  Loan interest rate prediction using Machine learning

This project involves developing a Streamlit application for predicting loan interest rates based on user-provided inputs. The application utilizes a trained XGBoost machine learning model loaded from a saved file (xgboost_model.pkl). Users input various factors such as loan amount, work experience, home ownership status, income verification, purpose of loan, gender, annual income, debt-to-income ratio, and other financial metrics. The input data undergoes preprocessing using LabelEncoder to transform categorical variables into numerical values suitable for model prediction. Upon clicking the prediction button, the application displays the estimated loan interest rate in a styled success box with a green background, white bold text, and bordered layout. This project integrates basic CSS styling within Streamlit's Markdown functionality to enhance the visual presentation of the prediction results, aiming for a user-friendly and informative interface for loan interest rate prediction.

## Data Set

The below csv dataset from kaggle is used as reference which contains nearly 150000+ rows (loan_prediction.csv) on which porcessing is performed to obtained a  processed data in .ipynb file.

The dataset link is are as follows :-
https://www.kaggle.com/datasets/shravankoninti/janatahack-machine-learning-for-banking/code

- Dataset columns :

Loan_ID	
Loan_Amount_Requested	
Length_Employed	
Home_Owner	
Annual_Income	
Income_Verified	
Purpose_Of_Loan	
Debt_To_Income	
Inquiries_Last_6Mo	
Months_Since_Deliquency	
Number_Open_Accounts	
Total_Accounts	
Gender	
Interest_Rate

## Methodology

The project follows the below structured methodology ranging from data preprocessing pipeline to model training, evaluation and deployment :-

1. **Data Preprocessing and feature enginnering**: 
2. **Exploratory Data Analysis (EDA)**:
    after Data preprocessing the next step is Exploratory  data analysis using different plotting libraries like matplotlib,pandas,seaborn and plotly.following plots were plotted in this step:-
    1) Pie charts
    2) violen plots
    3) box plots
    4) count plots
    5) heatmaps
    6) model comparison graphs
    7) coreelation heatmaps
    8) Histograms

    (refer images folder for this images and graph observation)
  

4. **Model Training and evaluation**: 
     The four machine learning model random forest ,linear regression, gradient boosting machine and XGBOOST are selected for model training over the inputed processed data:
    
     The  XGBOOST machine model is then loaded into streamlit application after installing and using joblib library.

5. **Inference**: 
      Deployed the model with the help streamlit web application to predict the loan interests.


## Libraries Used

1. **Joblib**: For downloading the random forest model
2. **Scikit learn**: For machine learning processing  and operations
3. **Matplotlib**: For plotting and visualizing the detection results.
4. **Pandas**: For image manipulation.
5. **NumPy**: For efficient numerical operations.
6. **Seaborn** : for advanced data visualizations
7. **plotly** : for 3D data visualizations .
8. **Streamlit** : for creating gui of the web application.


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

4. **View Results**: The script will allow you to predict rates on the loans.