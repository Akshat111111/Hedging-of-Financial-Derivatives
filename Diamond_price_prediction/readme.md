#  Car ownership prediction using Machine learning

This Streamlit application predicts diamond prices based on user-input diamond characteristics and provides dynamic information about diamonds fetched from a website. It incorporates a machine learning model trained to predict diamond prices using features like carat weight, cut, color, clarity, and dimensions. Users can interactively input these attributes through a user-friendly interface and receive predicted price estimates instantly.

The application includes three main functionalities accessed through a sidebar navigation:

Diamond Price Prediction: Users input diamond characteristics such as carat weight, cut type, color, clarity, depth, table, and dimensions. Upon clicking the "Predict Price" button, the app utilizes a pre-trained machine learning model to compute and display the predicted diamond price in lakhs.

Dataset View: This section allows users to explore a dataset (diamonds.csv in this example) containing historical diamond data. The dataset is displayed in a tabular format using Streamlit's st.dataframe function, providing insights into various attributes and prices of diamonds.

## Methodology

The project follows the below structured methodology ranging from data preprocessing pipeline to model training, evaluation and deployment :-

1. **Data Preprocessing and feature enginnering**: 
2. **Exploratory Data Analysis (EDA)**:
    after Data preprocessing the next step is Exploratory  data analysis using different plotting libraries like matplotlib,pandas,seaborn and plotly.following plots were plotted in this step:-
    1) Pie charts
    2) violin plots
    3) box plots
    4) count plots
    5) heatmap or confusion matrix for four different models of machine learning
    6) model comparison graphs
    (refer images folder for this images and graph observation)

 **Model Training and evaluation**: 
     The four machine learning model random forest ,decision trees linear regression , support vector machine are selected for model training over the inputed processed data:
     random forest accuracy : 98 %
     support vector machine accuracy : 87 %
     decision trees accuracy : 97 %
     linear regression accuracy : 90 %

     The 10 fold cross validation is then performed on  Random forest model to obtained a final average cross validated accuracy of 98.25 %.

 **Inference**: 
      Deployed the model with the help streamlit web application to predict the price of diamonds.


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

3. **Download the model in drive and Run the Model**: 
    
    drive link : https://drive.google.com/file/d/1OQ3WwkFdhlTmOptbfgudzCx6nS5QggQH/view?usp=sharing (keep it in same direcotry as app.py)

    ```python
    streamlit run app.py
    ```

4. **View Results**: The script will allow you to predict the estimated price of diamonds.