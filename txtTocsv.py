import pandas as pd

# Path to your input txt file
input_txt_file = "C:\\Users\jonat\Downloads\individual+household+electric+power+consumption\household_power_consumption.txt"

# Path to your output csv file
output_csv_file = "C:\\Users\jonat\Downloads\individual+household+electric+power+consumption\household_power_consumption.csv"



# Load using pandas with proper separator and handle missing values
df = pd.read_csv(
    input_txt_file,
    sep=';',
    header=0,
    na_values='?',
    low_memory=False
)

# Optional: combine Date + Time into single datetime column (very useful for time series)
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
df = df.drop(columns=['Date', 'Time'])
df = df.set_index('Datetime')

# Save to CSV
df.to_csv(output_csv_file)

print(f"Conversion complete. CSV saved to: {output_csv_file}")
