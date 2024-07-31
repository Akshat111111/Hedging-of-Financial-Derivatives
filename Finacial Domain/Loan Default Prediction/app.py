import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Sample data for the demonstration
epochs = list(range(1, 11))
mlp_loss = [0.5130, 0.5036, 0.5009, 0.4999, 0.4987, 0.4970, 0.4967, 0.4953, 0.4942, 0.4938]
mlp_val_loss = [0.5149, 0.5129, 0.5143, 0.5140, 0.5132, 0.5139, 0.5137, 0.5169, 0.5148, 0.5135]
mlp_accuracy = 0.7910

cnn_loss = [0.5124, 0.5050, 0.5033, 0.5015, 0.5005, 0.4997, 0.4990, 0.4982, 0.4975, 0.4957]
cnn_val_loss = [0.5167, 0.5141, 0.5138, 0.5163, 0.5148, 0.5158, 0.5175, 0.5138, 0.5172, 0.5184]
cnn_accuracy = 0.7910

rnn_loss = [0.5141, 0.5025, 0.5026, 0.5026, 0.5020, 0.5014, 0.5022, 0.5015, 0.5020, 0.5019]
rnn_val_loss = [0.5135, 0.5130, 0.5128, 0.5133, 0.5137, 0.5227, 0.5143, 0.5128, 0.5129, 0.5131]
rnn_accuracy = 0.7910

# Streamlit app
st.title("Model Performance Comparison")

st.header("MLP Model")
st.write(f"Accuracy: {mlp_accuracy * 100:.2f}%")
fig, ax = plt.subplots()
ax.plot(epochs, mlp_loss, label='Training Loss')
ax.plot(epochs, mlp_val_loss, label='Validation Loss')
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.set_title('MLP Loss')
ax.legend()
st.pyplot(fig)

st.header("CNN Model")
st.write(f"Accuracy: {cnn_accuracy * 100:.2f}%")
fig, ax = plt.subplots()
ax.plot(epochs, cnn_loss, label='Training Loss')
ax.plot(epochs, cnn_val_loss, label='Validation Loss')
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.set_title('CNN Loss')
ax.legend()
st.pyplot(fig)

st.header("RNN Model")
st.write(f"Accuracy: {rnn_accuracy * 100:.2f}%")
fig, ax = plt.subplots()
ax.plot(epochs, rnn_loss, label='Training Loss')
ax.plot(epochs, rnn_val_loss, label='Validation Loss')
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.set_title('RNN Loss')
ax.legend()
st.pyplot(fig)
