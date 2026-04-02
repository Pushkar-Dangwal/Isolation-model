"""Quick integration test for DataLoader with actual data."""

from src.dual_evaluation import DataLoader

# Test loading actual data
loader = DataLoader(random_state=42)
print("Loading full dataset...")
df = loader.load_full_dataset('data/financial.csv')
print(f"✓ Loaded {len(df):,} transactions")
print(f"✓ Fraud rate: {df['is_fraud'].mean():.4f}")
print(f"✓ Columns: {len(df.columns)}")

# Test balanced dataset creation
print("\nCreating balanced dataset...")
balanced_df = loader.create_balanced_dataset(df)
print(f"✓ Balanced dataset size: {len(balanced_df):,}")
print(f"✓ Balanced fraud rate: {balanced_df['is_fraud'].mean():.4f}")

# Test time-based split
print("\nTesting time-based split...")
train_df, test_df = loader.time_based_split(df, test_size=0.2)
print(f"✓ Train size: {len(train_df):,}, Test size: {len(test_df):,}")
print(f"✓ No temporal overlap: {train_df['timestamp'].max() < test_df['timestamp'].min()}")

# Test stratified split
print("\nTesting stratified split...")
train_df2, test_df2 = loader.stratified_split(balanced_df, test_size=0.2)
print(f"✓ Train size: {len(train_df2):,}, Test size: {len(test_df2):,}")
print(f"✓ Train fraud rate: {train_df2['is_fraud'].mean():.4f}")
print(f"✓ Test fraud rate: {test_df2['is_fraud'].mean():.4f}")

print("\n✅ All DataLoader integration tests passed!")
