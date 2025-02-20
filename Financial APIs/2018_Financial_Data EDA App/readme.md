

#  stocks Visualization App

<img width="896" alt="Screenshot 2024-08-07 at 1 35 18 PM" src="https://github.com/user-attachments/assets/eb7e51c1-4b5b-4749-8b60-9ab75572a768">



This application provides interactive visualizations of numerical data from a CSV file using Gradio. The application allows users to view line plots and bar charts for the specified date range.

## Features
- **Line Plot**: Displays trends for all numerical features over time.
- **Bar Plot**: Shows the average values of numerical features per month.

## Requirements
- Python 3.x
- `pandas`
- `matplotlib`
- `gradio`

You can install the required packages using:
```bash
pip install pandas matplotlib gradio
```

## Usage
1. **Prepare Your Data**: Ensure your CSV file is in the same directory as this script. The CSV should have a date column as the index and numerical columns for plotting.

2. **Run the App**: Execute the script to launch the Gradio interface:
    ```bash
    python your_script_name.py
    ```
   Replace `your_script_name.py` with the name of your Python script.

3. **Interact with the App**:
   - Enter the **Start Date** and **End Date** in the format `YYYY-MM-DD`.
   - Click **Submit** to generate the plots.
   - View the generated **Line Plot** and **Bar Plot**.

## Example
For a CSV file with the following columns: `Date`, `Feature1`, `Feature2`, the app will generate:
- **Line Plot**: Trends of `Feature1` and `Feature2` over the selected date range.
- **Bar Plot**: Monthly averages of `Feature1` and `Feature2`.

## Troubleshooting
- **No Numeric Data Error**: Ensure your CSV contains numeric data and the index column is of date type.
- **Plot Generation Error**: Check if the date range covers the data present in the CSV file.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to adjust the content as needed!
