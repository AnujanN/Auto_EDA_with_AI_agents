import pandas as pd
import numpy as np

# Create a sample dataset for testing
np.random.seed(42)

# Generate sample data
n_samples = 1000

data = {
    'age': np.random.normal(35, 10, n_samples).astype(int),
    'income': np.random.normal(50000, 15000, n_samples),
    'experience': np.random.normal(8, 4, n_samples),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples, p=[0.3, 0.4, 0.25, 0.05]),
    'department': np.random.choice(['Sales', 'Engineering', 'Marketing', 'HR', 'Finance'], n_samples),
    'satisfaction_score': np.random.uniform(1, 10, n_samples),
    'performance_rating': np.random.choice(['Poor', 'Average', 'Good', 'Excellent'], n_samples, p=[0.1, 0.3, 0.4, 0.2])
}

# Add some missing values
missing_indices = np.random.choice(n_samples, size=int(0.05 * n_samples), replace=False)
data['income'][missing_indices[:20]] = np.nan
data['satisfaction_score'][missing_indices[20:40]] = np.nan

# Add some correlations
data['salary'] = data['income'] + data['experience'] * 2000 + np.random.normal(0, 5000, n_samples)

# Create target variable
data['promoted'] = (
    (data['performance_rating'] == 'Excellent') | 
    ((data['performance_rating'] == 'Good') & (data['satisfaction_score'] > 7))
).astype(int)

# Create DataFrame
df = pd.DataFrame(data)

# Ensure age is reasonable
df['age'] = np.clip(df['age'], 22, 65)
df['experience'] = np.clip(df['experience'], 0, df['age'] - 22)

# Save to CSV
df.to_csv('sample_employee_data.csv', index=False)
print("Sample dataset created: sample_employee_data.csv")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nFirst few rows:")
print(df.head())
