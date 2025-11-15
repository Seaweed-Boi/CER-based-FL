# Loading Your Own Dataset

This guide shows you how to use your own dataset instead of the synthetic data.

## Requirements

Your dataset must:
1. Be in **CSV format**
2. Have **numeric features** (any number of columns)
3. Have a **binary target column** (0/1 labels)
4. Optionally handle missing values (will be filled with column means)

## Option 1: Load Custom Dataset

### Step 1: Prepare Your CSV File

Example format:
```csv
feature_0,feature_1,feature_2,...,is_fraud
0.5,1.2,0.8,...,0
0.3,0.9,1.5,...,1
0.7,1.1,0.6,...,0
...
```

### Step 2: Load Your Dataset

```bash
# Activate virtual environment
source venv/bin/activate

# Load your dataset (replace with your file path and target column name)
python -m data.generate_synthetic --dataset path/to/your/data.csv --target is_fraud

# Example with credit card fraud dataset:
python -m data.generate_synthetic --dataset ~/datasets/creditcard.csv --target Class

# Example with custom test/train split ratio:
python -m data.generate_synthetic --dataset data.csv --target label --test-size 0.3
```

**Parameters:**
- `--dataset`: Path to your CSV file
- `--target`: Name of your target/label column (default: "label")
- `--test-size`: Fraction for test set (default: 0.2 = 20%)

### Step 3: Split into Client Shards

```bash
python -m data.split_shards
```

This creates non-IID client shards in `data/shards/`.

### Step 4: Run FL Training

```bash
./scripts/run_all_locally.sh --config baseline --num-clients 3
```

## Option 2: Keep Using Synthetic Data

```bash
# Generate synthetic data (default behavior)
python -m data.generate_synthetic --synthetic
```

## What Happens Behind the Scenes

1. **Data Loading**: Your CSV is loaded and split into features (X) and target (y)
2. **Missing Values**: Any NaN values are filled with column means
3. **Train/Test Split**: Data is split (default 80/20) with stratification
4. **Standardization**: Saves to `data/raw/train.csv` and `data/raw/test.csv`
5. **Feature Names**: Columns are auto-named as `feature_0`, `feature_1`, etc.

## Adjusting Model Input Dimension

If your dataset has a different number of features than the default (20), the model will automatically adjust. However, you can verify the input dimension in training logs:

```
Loaded training data: 2400 samples
Model input dimension: 25  # <-- Your feature count
```

## Example: Credit Card Fraud Dataset

```bash
# Download Kaggle credit card fraud dataset
# https://www.kaggle.com/mlg-ulb/creditcardfraud

# Load it (284,807 samples, 30 features)
python -m data.generate_synthetic \
    --dataset ~/Downloads/creditcard.csv \
    --target Class \
    --test-size 0.2

# Split into 3 non-IID client shards
python -m data.split_shards

# Run FL training
./scripts/run_all_locally.sh --config baseline --num-clients 3
```

## Example: Custom Medical Dataset

```bash
# Assume you have heart_disease.csv with columns:
# age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target

python -m data.generate_synthetic \
    --dataset data/heart_disease.csv \
    --target target \
    --test-size 0.25

# Continue with FL training as normal
```

## Troubleshooting

### Error: "Target column not found"
Make sure your target column name matches the `--target` parameter exactly (case-sensitive).

```bash
# Check column names in your CSV
head -1 your_data.csv
```

### Error: "ValueError: could not convert string to float"
Your dataset contains non-numeric columns. You need to:
1. Remove non-numeric columns before loading
2. Or encode them (one-hot encoding, label encoding)

### Different Number of Features
The model automatically detects the input dimension from `train.csv`. No changes needed!

### Imbalanced Classes
The system is designed for imbalanced data (uses Focal Loss). Your dataset can have any class distribution.

## Next Steps

After loading your dataset:
1. ✅ Verify `data/raw/train.csv` and `data/raw/test.csv` were created
2. ✅ Check the printed feature count and class distribution
3. ✅ Run `python -m data.split_shards` to create client shards
4. ✅ Train with `./scripts/run_all_locally.sh`
5. ✅ View results in Streamlit dashboard: `streamlit run dashboard/app.py`

---

**Note**: The current implementation assumes binary classification (2 classes). For multi-class problems, you'll need to modify `clients/model.py` to change the output dimension and loss function.
