# **Bank Customer Churn Prediction**

## Overview
****
In the banking industry, it is important to predict and understand when customers might decide to end their relationship with the bank, known as customer churn. When customers leave, it can lead to financial losses and impact the bank's reputation.By identifying customers who are likely to churn, the bank can proactively take measures to retain them and minimize revenue loss.Therefore, the goal of this project is to develop a system that can accurately predict customer churn in order to take proactive steps to retain these customers.

## Data
***
The data used in this project is from <a href="churn.csv"> `Churn for Bank Customers`</a>.

## Data Understanding
***
        i. Visualizing customer Churn Distribution
<img src="images\Customer Churn Distribution.jpg"  width="650" height="400"> 

        ii. Visualizing Churn distribution based on gender
<img src="images\Churn distribution based on gender.jpg"  width="650" height="400"> 

        iii. Visualizing Churn distribution based on customer age
<img src="images\Churn distribution based on Age.jpg"  width="650" height="400"> 

        iv. Visualizing Churn distribution based on Location
<img src="images\Distribution of Exited based on Location.jpg"  width="650" height="400"> 


## Modelling
***
Different models were evaluated and the best performing model was picked to be the final model.Precision was used as the main evaluation metric to avoid wasting resources on customers who would not have churned. The Random Forest Classifier was picked as the final model, with its parameters being the best parameters found through grid search. 

                        Models
 <img src="images\models.jpg"  width="650" height="400"> 


## Evaluation
***
The model achieves an accuracy of 86% in identifying potential churners. Key features influencing churn prediction include estimated salary, active membership status, credit card presence, number of products purchased, and account balance. 

                        Classification report
 <img src="images\classification report.jpg"  width="650" height="400"> 

                        Confusion Matrix
 <img src="images\confusion matrix.jpg"  width="650" height="400"> 

                        Feature importance
<img src="images\feature importance.jpg"  width="650" height="400">



## Conclusion
***
The accuracy score of the model is 86%. That means that the model predicts whether a customer will churn or not 86% of the time. By leveraging the predictive power of the model, the bank can proactively take actions to prevent customer churn and improve overall customer retention rates. By focusing resources on customers identified as likely to churn, the bank can implement targeted retention strategies, such as personalized offers, improved customer service, or loyalty programs, to keep these customers engaged and satisfied. Ultimately, this approach can help the bank optimize its retention efforts and minimize customer churn.

## Limitations
***
* **Data availability:** The accuracy of the churn prediction model depends on the quality and availability of historical customer data. 
* **Changing factors:** The factors that contribute to customer churn can evolve over time. Therefore, the churn prediction model requires regular maintenance.
* **External factors:** The model may not account for external factors such as economic conditions, regulatory changes, or competitive landscape, which can influence customer churn. These factors should be considered when interpreting the predictions and implementing retention strategies. 

