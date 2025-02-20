import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr

# Load the CSV file
data_path = 'PH_Housing.csv'

def load_data():
    df = pd.read_csv(data_path)
    df.dropna(subset=['Price', 'Bedrooms', 'Floor Area'], inplace=True)  # Drop rows with missing values in key columns
    return df

def visualize_data(max_price):
    # Load data
    df = load_data()

    # Filter data based on max price
    filtered_df = df[df['Price'] <= max_price]

    if filtered_df.empty:
        return "Error: No data available for the selected price range.", None

    # Plot bar chart for average property prices by number of bedrooms
    plt.figure(figsize=(12, 8))
    avg_prices = filtered_df.groupby('Bedrooms')['Price'].mean().reset_index()
    avg_prices = avg_prices.sort_values(by='Price', ascending=False)  # Sort by price
    sns.barplot(x='Bedrooms', y='Price', data=avg_prices, ci=None, palette='viridis')
    plt.title('Average Property Prices by Number of Bedrooms')
    plt.xlabel('Number of Bedrooms')
    plt.ylabel('Average Price (in $)')
    plt.tight_layout()

    # Save plot to a file
    plot_path = 'bar_plot.png'
    plt.savefig(plot_path)
    plt.close()

    # Generate EDA summary
    eda_summary = filtered_df.describe(include='all').to_string()

    return eda_summary, plot_path

iface = gr.Interface(
    fn=visualize_data,
    inputs=gr.Number(label="Max Price (in $)"),
    outputs=[gr.Textbox(label="EDA Summary"), gr.Image(type="filepath", label="Bar Plot")]
)

iface.launch(share=True)
