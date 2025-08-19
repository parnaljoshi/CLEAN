import pandas as pd
from sklearn.model_selection import train_test_split

# # Load the TSV file
# df = pd.read_csv('EC7_combined_output_deduplicated.tsv', sep='\t')

# # Split into 80% train and 20% test
# train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

# # Save to TSV files
# train_df.to_csv('EC7_train_set.tsv', sep='\t', index=False)
# test_df.to_csv('EC7_test_set.tsv', sep='\t', index=False)

# Load the TSV file
df = pd.read_csv('EC1_combined_output_deduplicated.tsv', sep='\t')

# Split into 80% train and 20% test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

# Save to TSV files
train_df.to_csv('EC1_train_set.tsv', sep='\t', index=False)
test_df.to_csv('EC1_test_set.tsv', sep='\t', index=False)

# Load the TSV file
df = pd.read_csv('EC2_combined_output_deduplicated.tsv', sep='\t')

# Split into 80% train and 20% test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

# Save to TSV files
train_df.to_csv('EC2_train_set.tsv', sep='\t', index=False)
test_df.to_csv('EC2_test_set.tsv', sep='\t', index=False)

# Load the TSV file
df = pd.read_csv('EC3_combined_output_deduplicated.tsv', sep='\t')

# Split into 80% train and 20% test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

# Save to TSV files
train_df.to_csv('EC3_train_set.tsv', sep='\t', index=False)
test_df.to_csv('EC3_test_set.tsv', sep='\t', index=False)

# Load the TSV file
df = pd.read_csv('EC4_combined_output_deduplicated.tsv', sep='\t')

# Split into 80% train and 20% test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

# Save to TSV files
train_df.to_csv('EC4_train_set.tsv', sep='\t', index=False)
test_df.to_csv('EC4_test_set.tsv', sep='\t', index=False)

# Load the TSV file
df = pd.read_csv('EC5_combined_output_deduplicated.tsv', sep='\t')

# Split into 80% train and 20% test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

# Save to TSV files
train_df.to_csv('EC5_train_set.tsv', sep='\t', index=False)
test_df.to_csv('EC5_test_set.tsv', sep='\t', index=False)

# Load the TSV file
df = pd.read_csv('EC6_combined_output_deduplicated.tsv', sep='\t')

# Split into 80% train and 20% test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

# Save to TSV files
train_df.to_csv('EC6_train_set.tsv', sep='\t', index=False)
test_df.to_csv('EC6_test_set.tsv', sep='\t', index=False)