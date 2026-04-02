"""
Check financial.csv size and generate exactly 300k dataset
"""
import pandas as pd

print("Checking financial.csv...")
try:
    # Try to count rows without loading all
    df = pd.read_csv('data/financial.csv')
    total_rows = len(df)
    print(f"Total rows in financial.csv: {total_rows:,}")
    print(f"Fraud: {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.4f}%)")
    
    # Calculate what we need for 300k
    print("\n" + "=" * 60)
    print("OPTIONS TO GET 300K DATASET:")
    print("=" * 60)
    
    print("\nOption 1: 240k train + 60k test = 300k")
    print("  - Load 75,000 rows from financial.csv")
    print("  - Split: 60k train + 15k test")
    print("  - SMOTENC train to 240k (120k fraud + 120k legit)")
    print("  - Keep test at 60k (original distribution)")
    print("  - Total: 300k")
    
    print("\nOption 2: 290k train + 10k test = 300k")
    print("  - Load 50,000 rows from financial.csv")
    print("  - Split: 40k train + 10k test")
    print("  - SMOTENC train to 290k (145k fraud + 145k legit)")
    print("  - Keep test at 10k (original distribution)")
    print("  - Total: 300k")
    
    print("\nOption 3: Use ALL data")
    print(f"  - Load all {total_rows:,} rows")
    print("  - Generate to desired size")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATION:")
    print("=" * 60)
    print("Use Option 1 for better test set size (60k)")
    print("Larger test set = more reliable metrics")
    
except Exception as e:
    print(f"Error: {e}")
