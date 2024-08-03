import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load training histories
nn_history = pd.read_csv('nn_history.csv')
cnn_history = pd.read_csv('cnn_history.csv')
rnn_history = pd.read_csv('rnn_history.csv')

st.title("IPO Performance Prediction")

st.header("Exploratory Data Analysis (EDA)")
# Load dataset
df = pd.read_csv('ipo_data.csv')

# Display dataset
st.write("Dataset Overview:")
st.write(df.head())

# EDA: Distribution of IPO Return
st.write("Distribution of IPO Return")
fig, ax = plt.subplots()
sns.histplot(df['IPO_Return'], bins=30, kde=True, ax=ax)
ax.set_title('Distribution of IPO Return')
ax.set_xlabel('IPO Return')
ax.set_ylabel('Frequency')
st.pyplot(fig)

# EDA: Pairplot of numerical features
st.write("Pairplot of Numerical Features")
fig = sns.pairplot(df, diag_kind='kde')
st.pyplot(fig)

# EDA: Correlation heatmap
st.write("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(12, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
ax.set_title('Correlation Heatmap')
st.pyplot(fig)

# EDA: Boxplot of IPO Return by Sector
st.write("IPO Return by Sector")
fig, ax = plt.subplots()
sns.boxplot(x='Sector', y='IPO_Return', data=df, ax=ax)
ax.set_title('IPO Return by Sector')
ax.set_xlabel('Sector')
ax.set_ylabel('IPO Return')
st.pyplot(fig)

st.header("Basic Neural Network (NN)")
st.write("Mean Absolute Error (MAE) over Epochs")
fig, ax = plt.subplots()
ax.plot(nn_history['epoch'], nn_history['mae'], label='Training MAE')
ax.plot(nn_history['epoch'], nn_history['val_mae'], label='Validation MAE')
ax.set_xlabel('Epochs')
ax.set_ylabel('MAE')
ax.legend()
st.pyplot(fig)

st.header("Convolutional Neural Network (CNN)")
st.write("Mean Absolute Error (MAE) over Epochs")
fig, ax = plt.subplots()
ax.plot(cnn_history['epoch'], cnn_history['mae'], label='Training MAE')
ax.plot(cnn_history['epoch'], cnn_history['val_mae'], label='Validation MAE')
ax.set_xlabel('Epochs')
ax.set_ylabel('MAE')
ax.legend()
st.pyplot(fig)

st.header("Recurrent Neural Network (RNN)")
st.write("Mean Absolute Error (MAE) over Epochs")
fig, ax = plt.subplots()
ax.plot(rnn_history['epoch'], rnn_history['mae'], label='Training MAE')
ax.plot(rnn_history['epoch'], rnn_history['val_mae'], label='Validation MAE')
ax.set_xlabel('Epochs')
ax.set_ylabel('MAE')
ax.legend()
st.pyplot(fig)
