import pandas as pd

data = {
    'Name': ['John', 'Jane', 'Jim', 'Joan'],
    'Age': [30, 29, 31, 32],
    'Country': ['USA', 'Canada', 'UK', 'Australia']
}
df = pd.DataFrame(data)
print("DataFrame:")
print(df)

second_df = df[['Country']]. copy ()
print("\nSecond DataFrame:")
print(second_df)

print("\nFirst 5 rows of the DataFrame:")
print(df.head(5))

print("\nSummary statistics of the DataFrame:")
print(df.describe())

print("\nSecond row, second column element using iloc:")
print(df.iloc[1,1])

print("\nSecond row, 'Name' column element using loc:")
print(df.loc[1, 'Name'])
