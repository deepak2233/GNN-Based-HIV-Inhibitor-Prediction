# %% Load raw dataset
import pandas as pd

data = pd.read_csv("data/split_data/HIV_train.csv")
data.index = data["index"] if "index" in data.columns else data.index
print("Original class distribution:")
print(data["HIV_active"].value_counts())
start_index = data.index[0]

# %% Apply oversampling

# Check how many additional samples we need
neg_class = data["HIV_active"].value_counts()[0]
pos_class = data["HIV_active"].value_counts()[1]
multiplier = int(neg_class / pos_class) - 1

print(f"\nNegative class: {neg_class}")
print(f"Positive class: {pos_class}")
print(f"Replication multiplier: {multiplier}")

# Replicate the dataset for the positive class
replicated_pos = [data[data["HIV_active"] == 1]] * multiplier

# Concatenate replicated data (fixed: pd.DataFrame.append is deprecated)
data = pd.concat([data] + replicated_pos, ignore_index=True)
print(f"\nAfter oversampling: {data.shape}")

# Shuffle dataset
data = data.sample(frac=1).reset_index(drop=True)

# Re-assign index (This is our ID later)
index = range(start_index, start_index + data.shape[0])
data.index = index
if "index" in data.columns:
    data["index"] = data.index

print("\nOversampled class distribution:")
print(data["HIV_active"].value_counts())

# %% Save
data.to_csv("data/split_data/HIV_train_oversampled.csv")
print("\nSaved to data/split_data/HIV_train_oversampled.csv")

# %%
