import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
import os
import numpy as np

# Define the path to the CSV file
csv_file_path = 'Coca-Cola_stock_history.csv'

def visualize_eda(start_date, end_date):
    # Create a directory to save plots
    if not os.path.exists('eda_plots'):
        os.makedirs('eda_plots')

    # Initialize output paths
    line_plot_path = 'eda_plots/line_plot.png'
    bar_plot_path = 'eda_plots/bar_plot.png'
    hist_plot_path = 'eda_plots/hist_plot.png'
    scatter_plot_path = 'eda_plots/scatter_plot.png'

    # Load the data from the CSV file
    try:
        df = pd.read_csv(csv_file_path, parse_dates=True, index_col=0)
    except Exception as e:
        return [None, None, None, None, f"Error loading CSV file: {e}"]

    # Filter the data by the given date range
    try:
        df = df.loc[start_date:end_date]
    except Exception as e:
        return [None, None, None, None, f"Error filtering data: {e}"]

    # Check for and handle non-numeric columns
    df_numeric = df.select_dtypes(include=['float64', 'int64'])
    if df_numeric.empty:
        return [None, None, None, None, "Error: No numeric data found in CSV file."]

    # Plot 1: Line plot of all numerical features
    try:
        plt.figure(figsize=(12, 6))
        for column in df_numeric.columns:
            plt.plot(df_numeric.index, df_numeric[column], label=column)
        plt.title('Line Plot of Numerical Features')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig(line_plot_path)
        plt.close()
    except Exception as e:
        return [None, None, None, None, f"Error generating line plot: {e}"]

    # Plot 2: Bar plot of average values per month
    try:
        plt.figure(figsize=(12, 6))
        monthly_avg = df_numeric.resample('M').mean()
        monthly_avg.plot(kind='bar', figsize=(15, 7))
        plt.title('Monthly Average of Numerical Features')
        plt.xlabel('Month')
        plt.ylabel('Average Value')
        plt.savefig(bar_plot_path)
        plt.close()
    except Exception as e:
        return [None, None, None, None, f"Error generating bar plot: {e}"]

    # Plot 3: Histogram of numerical features
    try:
        plt.figure(figsize=(12, 6))
        df_numeric.hist(bins=30, figsize=(15, 7))
        plt.suptitle('Histogram of Numerical Features')
        plt.savefig(hist_plot_path)
        plt.close()
    except Exception as e:
        return [None, None, None, None, f"Error generating histogram: {e}"]

    # Plot 4: Scatter plot matrix of numerical features
    try:
        from pandas.plotting import scatter_matrix
        plt.figure(figsize=(12, 12))
        scatter_matrix(df_numeric, alpha=0.2, figsize=(15, 15), diagonal='kde')
        plt.suptitle('Scatter Plot Matrix of Numerical Features')
        plt.savefig(scatter_plot_path)
        plt.close()
    except Exception as e:
        return [None, None, None, None, f"Error generating scatter plot matrix: {e}"]

    # Return paths to generated plots
    return [line_plot_path, bar_plot_path, hist_plot_path, scatter_plot_path, None]

# Define the Gradio interface
iface = gr.Interface(
    fn=visualize_eda,
    inputs=[
        gr.Textbox(label="Start Date (YYYY-MM-DD)", value="2002-01-02"),
        gr.Textbox(label="End Date (YYYY-MM-DD)", value="2022-10-10")
    ],
    outputs=[
        gr.Image(type="filepath", label="Line Plot"),
        gr.Image(type="filepath", label="Bar Plot"),
        gr.Image(type="filepath", label="Histogram"),
        gr.Image(type="filepath", label="Scatter Plot Matrix"),
        gr.Textbox(label="Error Message", type="text")  # Add a textbox for error messages
    ],
    live=False  # This will add an explicit "Submit" button
)

# Launch the Gradio app
iface.launch(share=True, inbrowser=True)
