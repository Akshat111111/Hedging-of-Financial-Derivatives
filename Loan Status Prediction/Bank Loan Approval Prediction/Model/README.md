## **BANK LOAN APPROVAL PREDICTION**

### 🎯 **Goal**

 The main goal of this project is to come up with Deep Learning multi-layer neural network model for predicting approval for personal bank loans on the basis of customer's information which includes their age, experience, income, geographical information and many more.

### 🧵 **Dataset**

The Universal Bank dataset is taken from [Kaggle](https://www.kaggle.com/datasets/vinod00725/svm-classification?select=UniversalBank.csv) and can be found [here](https://github.com/abhisheks008/DL-Simplified/tree/main/Bank%20Loan%20Approval%20Prediction/Dataset). The dataset for this project consists of labeled data. The target column is called 'Personal Loan' which is used to predict whether a customer gets approved for loan or not.

### 🧾 **Description**

For training the model, different Deep Learning approches are considered. These are the deep learning algorithms which are considered.

* Feedforward Neural-Network
* Feedforward Neural Network with k-Fold validation
* TabNet model with k-Fold validation
* Wide & Deep neural network architecture

### 🧾 Data Preprocessing

These are the observations which are made on dataset.

* The minimum value of Experience is -3 and it also contains numeric values which are less than 0 which is not possible. It is observed that this field has 52 negative values. Further it was observed that minimum age and experience diffrence is 23. So wherever the experience was less than 0, it was replaced with their age minus 23.
* ZIP Code was initially represented as a numeric data. But it is a nominal data. Out of 5000 records, there are only 467 unique ZIP codes. Thus this represents that the dataset is restricted to a particular region. So this was converted to appropriate nominal data format.
* Education was also initially represented as a numeric data having 3 unique values {1: Bachelor, 2: Masters, 3: Advanced Degree}. So this is again not a numeric data. It is ordinal data and was converted to appropriate data format.
* Personal Loan (Target Variable) is either 0 or 1. {0: Loan not approved, 1: Loan approved}. So this is binary data,
* Securities Account is binary data representing {0: doesn't have security account, 1: has security account}
* CD Account is binary data representing {0: doesn't have CD Account, 1: has CD Account}
* Online is binary data representing {0: doesn't use online banking, 1: uses online banking}
* Credit Card is binary data representing {0: doesn't have credit card, 1: has credit card}

All these binary data were initally numeric data, so these were changed to boolean data format. Rest are numeric data.

### 🚀 **Models Implemented**

Three deep learning algorithms are implemented which give more than 90% validation accuracy. These models are described as follows:

#### Feedforward Neural Network with k-Fold validation

Here we implement a feedforward neural network for binary classification using TensorFlow and Keras. It uses K-Fold Cross-Validation to evaluate the model's performance, ensuring that the results are reliable and generalize well to unseen data. Each fold involves training a new model and applying early stopping to prevent overfitting, with the best epoch's weights restored for evaluation.

Layers:
* The first dense layer has 64 neurons and uses the ReLU activation function.
* The second dense layer has 32 neurons and also uses the ReLU activation function.
* The output layer has 1 neuron and uses the sigmoid activation function to output a probability for binary classification.

Compilation:
* The loss function is binary_crossentropy, suitable for binary classification.
* The optimizer is adam, an adaptive learning rate optimizer.
* The metric is accuracy.

K-Fold Cross-Validation:
* The dataset is split into 5 parts (folds).

Accuracies over all folds

| Fold                 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|----------------------|--------|--------|--------|--------|--------|
| **Best Epoch**       | 47     | 45     | 25     | 47     | 45     |
| **Final Validation Loss** | 0.1204 | 0.0833 | 0.1053 | 0.1113 | 0.0882 |
| **Final Validation Accuracy** | 0.9549 | 0.9620 | 0.9660 | 0.9679 | 0.9710 |

* Overall Average Validation Loss: 0.1017
* Overall Average Validation Accuracy: 0.964

| ![image](https://github.com/theiturhs/DL-Simplified/assets/96874023/9352f641-2a02-4d11-b177-18a9c6b2a2f4) | ![image](https://github.com/theiturhs/DL-Simplified/assets/96874023/92325127-5511-41cb-8237-ec2da884e6f5) |
| ---- | ---- |
| Training vs Validation Accuracy : FNN Model | Training vs Validation Loss : FNN Model |


#### TabNet Model

In this code, we implement a TabNet-based classifier for binary classification using PyTorch. The model's performance is evaluated using K-Fold Cross-Validation, ensuring that the results are reliable and generalize well to unseen data. Each fold involves training a new model and applying early stopping to prevent overfitting, with the best epoch's weights restored for evaluation.

Components:

* Model Architecture: TabNet is a deep learning model specifically designed for tabular data, with capabilities for feature selection and interpretability.
* Optimizer: Adam, an adaptive learning rate optimizer.
* Learning Rate Scheduler: Reduces the learning rate by a factor of 0.9 every 10 epochs.
* Evaluation Metrics: Accuracy and logloss are used to evaluate the model's performance.
* K-Fold Cross-Validation: The dataset is split into 5 folds to ensure robust evaluation. Each fold involves training a new model and storing the best validation loss.

Accuracies over all folds

| Fold                         | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|------------------------------|--------|--------|--------|--------|--------|
| **Best Epoch**               | 35     | 45     | 46     | 41     | 24     |
| **Final Validation LogLoss** | 0.0438 | 0.0623 | 0.0626 | 0.0466 | 0.0651 |
| **Final Validation Accuracy**| 0.980  | 0.985  | 0.978  | 0.972  | 0.982  |

The parameters that yield better accuracy are selected.

| ![image](https://github.com/theiturhs/DL-Simplified/assets/96874023/3da4eb3c-07b8-4d6b-85a9-192f8d58d397) | ![image](https://github.com/theiturhs/DL-Simplified/assets/96874023/2957e428-68d7-4208-b435-a68ed23dea38) |
| ---- | ----|
| Training vs Validation Accuracy : TabNet Model | Training vs Validation Loss : TabNet Model |

#### Wide & Deep neural network architecture

This implements a Wide & Deep neural network architecture using TensorFlow's Keras API for binary classification tasks.

Components of the Model:
* Normalization of Data: The input data is normalized using mean and standard deviation calculated from the training data. This step helps in stabilizing the training process and improving convergence.
* Wide Component: The wide component is a linear model that directly connects the input features to the output layer without any non-linear transformations. It is represented by a single Dense layer.
* Deep Component: The deep component is a neural network consisting of multiple layers. Each layer is followed by Batch Normalization, LeakyReLU activation, and Dropout for regularization. It comprises three Dense layers with 128, 64, and 32 units, respectively.
* Combining Wide and Deep Components: The outputs from the wide and deep components are concatenated using the Concatenate layer. This allows the model to learn both low-level and high-level feature representations simultaneously.
* Final Output Layer: The concatenated output is passed through a final Dense layer with a sigmoid activation function, which outputs the predicted probability of the positive class (binary classification).

Metrics
* Training Accuracy: 0.9715
* Training Loss: 0.0752
* val_accuracy: 0.9760
* val_loss: 0.0531

| ![image](https://github.com/theiturhs/DL-Simplified/assets/96874023/39126a2b-c038-4678-a346-936604bc8f1e) | ![image](https://github.com/theiturhs/DL-Simplified/assets/96874023/0d4d3df9-5e95-41f9-9058-0f8f92e77146) |
| ---- | ---- |
| Training vs Validation Accuracy : WDNN Model | Training vs Validation Loss : WDNN Model |

### 📚 **Libraries Needed**

* pandas
* numpy
* matplotlib
* seaborn
* tensorflow
* joblib
* pytorch_tabnet
* sklearn

### 📊 **Exploratory Data Analysis Results**

| ![Age Distribution](https://github.com/theiturhs/DL-Simplified/assets/96874023/17709677-b86a-4d5a-8595-ac9a419de225) | ![Box Plot of income](https://github.com/theiturhs/DL-Simplified/assets/96874023/64ecfa44-e3be-4f35-9105-9aaed87bd940) |
| --- | --- |
| Distribution of Age | Box Plot of Income |
| ![CCAvg Distribution by Personal Loan](https://github.com/theiturhs/DL-Simplified/assets/96874023/0be9aab1-de5e-4170-a042-8e9620c6db15) | ![Distribution of Education](https://github.com/theiturhs/DL-Simplified/assets/96874023/36bb6796-132f-4c8c-9f1d-52c157222ff3) |
| CCAvg Distribution by Personal Loan | Distribution of Education |

### 📈 **Performance of the Models based on the Accuracy Scores**

Summary of model and their accuracy scores

| Models | ANN | FNN | TabNet Model | WDNN Model |
| --- | --- | --- | --- | --- |
| Accuracy | 0.9820 | 0.9710 | 0.985 | 0.9760 |

### 📢 **Conclusion**

Concluding, this project aimed to classifies Bank Loan Approval using Deep Learning models. Among the models developed, the TabNet model achieved the highest validation score of 0.985. Using K-Fold Cross-Validation, it ensured that the results are reliable and generalize well to unseen data.

