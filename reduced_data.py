import pandas as pd

# Load data
df = pd.read_csv("final_df.csv")

test = pd.read_csv("patient_data.csv")

unique_counts = test["DIAGNOSIS"].value_counts()
print(unique_counts)

filtered_counts = unique_counts[unique_counts < 1000]
print(filtered_counts)

rare_df = test[test['DIAGNOSIS'].isin(['STROKE', 'RENAL FAILURE', 'WOUND INFECTION'])]
print(rare_df)

# Save to csv
rare_df.to_csv('rare_df.csv', index=False)



# Get reduced data set for submission

# Set the target sample size per category
sample_size = 200

# Create a new reduced DataFrame by sampling 200 rows for each diagnosis
reduced_df = pd.concat([
    df[df['DIAGNOSIS'] == diagnosis].sample(n=sample_size, random_state=42)
    for diagnosis in df['DIAGNOSIS'].unique()
])

# Shuffle the resulting dataset (optional)
reduced_df = reduced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Check the reduced dataset
print(reduced_df['DIAGNOSIS'].value_counts())
print(len(reduced_df))

# Save to CSV
reduced_df.to_csv('reduced_final_df.csv', index=False)

