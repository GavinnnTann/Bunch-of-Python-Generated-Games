"""
Pandas filter test script to demonstrate filtering where:
- Column A contains "National"
- Column D contains "EPD"
"""
import pandas as pd

# Create a sample DataFrame with test data
data = {
    'Column A': ['National Park', 'International Site', 'National Monument', 'Local Area', 'National Historic'],
    'Column B': ['Info 1', 'Info 2', 'Info 3', 'Info 4', 'Info 5'],
    'Column C': [100, 200, 300, 400, 500],
    'Column D': ['EPD 123', 'ABC 456', 'EPD 789', 'XYZ 123', 'DEF 456'],
    'Column E': [True, False, True, False, True]
}

# Create DataFrame
df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)
print("\n" + "-"*50 + "\n")

# Method 1: Using pandas query() function
print("Method 1: Using pandas query function:")
filtered_df1 = df.query('`Column A`.str.contains("National") & `Column D`.str.contains("EPD")')
print(filtered_df1)
print("\n" + "-"*50 + "\n")

# Method 2: Using boolean indexing
print("Method 2: Using boolean indexing:")
filtered_df2 = df[df['Column A'].str.contains('National') & df['Column D'].str.contains('EPD')]
print(filtered_df2)
print("\n" + "-"*50 + "\n")

# Method 3: Using filter method with lambda
print("Method 3: Using filter with lambda:")
filtered_df3 = df[lambda x: x['Column A'].str.contains('National') & x['Column D'].str.contains('EPD')]
print(filtered_df3)
print("\n" + "-"*50 + "\n")

# Method 4: Case-insensitive filtering
print("Method 4: Case-insensitive filtering:")
filtered_df4 = df[
    df['Column A'].str.contains('national', case=False) & 
    df['Column D'].str.contains('epd', case=False)
]
print(filtered_df4)

# How to use this in the Excel Filter app's script section:
print("\n" + "="*60)
print("To use in Excel Filter app's Advanced Filter section:")
print("="*60)
print("\nPandas query (select 'pandas query' radio button):")
print('`Column A`.str.contains("National") & `Column D`.str.contains("EPD")')
print("\nPython lambda (select 'Python lambda' radio button):")
print('lambda df: df["Column A"].str.contains("National") & df["Column D"].str.contains("EPD")')
