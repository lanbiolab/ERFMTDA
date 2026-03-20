'''
@Author: Wang Dong
@Date: 2025.10.27
@Description: Data preprocessing and encodeing
@Negative sampling strategy: motif similarity based
'''


import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from collections import Counter
from tqdm import tqdm
from itertools import product

# -------------------------------
# configuration parameters
# -------------------------------
RAW_DATA_PATH = 'tsRNA-disease.xlsx'        # Original Excel file path
OUTPUT_DIR = './data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 1983
np.random.seed(RANDOM_STATE)

# -------------------------------
# 1. Read raw data
# -------------------------------
print("Start preprocessing...")
df_raw = pd.read_excel(RAW_DATA_PATH)
print(f"Raw data loading completed，with a total of {len(df_raw)} records")

before_dedup = len(df_raw)
df_raw = df_raw.drop_duplicates()
print(f"De duplication completed：Remove {before_dedup - len(df_raw)} duplicate records，leaving {len(df_raw)} records")
df_raw['Seq Length'] = df_raw['Seq Length'].astype(str)

# -------------------------------
# 4. Category feature encoding
# -------------------------------
le_tsRNA_id = LabelEncoder()
le_tsRNA_type = LabelEncoder()
le_tsRNA_type_1 = LabelEncoder()
le_tsRNA_type_2 = LabelEncoder()
le_Isotype_Anticodon = LabelEncoder()
le_Seq_Length = LabelEncoder()
le_disease = LabelEncoder()
le_organ = LabelEncoder()
le_icd = LabelEncoder()
le_icd_1 = LabelEncoder()
le_icd_2 = LabelEncoder()

df_encoded = pd.DataFrame()
df_encoded['tsrna_id_id'] = le_tsRNA_id.fit_transform(df_raw['tsRNA ID'])
df_encoded['tsrna_type_id'] = le_tsRNA_type.fit_transform(df_raw['tsRNA type'])
df_encoded['tsrna_type_1_id'] = le_tsRNA_type_1.fit_transform(df_raw['tsRNA type_1'])
df_encoded['tsrna_type_2_id'] = le_tsRNA_type_2.fit_transform(df_raw['tsRNA type_2'])
df_encoded['Isotype_Anticodon_id'] = le_Isotype_Anticodon.fit_transform(df_raw['Isotype_Anticodon'])
df_encoded['Seq_Length_id'] = le_Seq_Length.fit_transform(df_raw['Seq Length'])
df_encoded['disease_id'] = le_disease.fit_transform(df_raw['disease'])
df_encoded['organ_id'] = le_organ.fit_transform(df_raw['organ'])
df_encoded['icd_id'] = le_icd.fit_transform(df_raw['ICD'].astype(str))
df_encoded['icd_1_id'] = le_icd_1.fit_transform(df_raw['ICD_1'].astype(str))
df_encoded['icd_2_id'] = le_icd_2.fit_transform(df_raw['ICD_2'].astype(str))


# -------------------------------
# 5. Merge all features
# -------------------------------
processed_df = df_encoded.copy()
processed_df['sequence'] = df_raw['tsRNA Sequence'].values
processed_df['sequence_norm'] = processed_df['sequence'].str.upper().str.replace("T", "U")

id_cols = [
    'tsrna_id_id', 'tsrna_type_id', 'tsrna_type_1_id', 'tsrna_type_2_id', 
    'Isotype_Anticodon_id', 'Seq_Length_id', 
    'disease_id', 'organ_id', 'icd_id', 'icd_1_id', 'icd_2_id'
]
processed_df[id_cols] = processed_df[id_cols].astype(int)

print(f" Preprocessing completed：{len(processed_df)} positive samples")


# -------------------------------
# motif statistics
# -------------------------------
def extract_motif_binary(sequences, L_range=(12, 14), min_count=3):
    """
    Extract all motifs and generate a motif occurrence (0/1) matrix for each tsRNA.
    """
    motif_counter = Counter()
    tsrna_motif_list = []

    # 1.Count all motifs and their global occurrences
    for seq in tqdm(sequences, desc="Counting motifs"):
        s = seq.upper().replace("T", "U")
        seen = set()
        for L in range(L_range[0], L_range[1] + 1):
            for i in range(len(s) - L + 1):
                seen.add(s[i:i+L])
        motif_counter.update(seen)
        tsrna_motif_list.append(seen)

    # 2.Filter out low-frequency motifs and reduce dimensionality (adjustable as needed)
    motif_vocab = [m for m, c in motif_counter.items() if c >= min_count]
    print(f"✅ motif 种类数（过滤后）: {len(motif_vocab)}")

    # 3.construct a 0/1 matrix
    motif_index = {m: i for i, m in enumerate(motif_vocab)}
    motif_matrix = np.zeros((len(sequences), len(motif_vocab)), dtype=np.float32)

    for i, seen in enumerate(tsrna_motif_list):
        for m in seen:
            if m in motif_index:
                motif_matrix[i, motif_index[m]] = 1

    return motif_matrix, motif_vocab


# -------------------------------
# 5+. Motif feature generation + PCA dimensionality reduction
# -------------------------------

# Construct unique tsRNA → sequence mapping (in the order of LabelEncoder)
tsrna_seq_list = []
for tsrna_name in le_tsRNA_id.classes_:
    seqs = processed_df.loc[processed_df['tsrna_id_id'] ==
                            le_tsRNA_id.transform([tsrna_name])[0], 'sequence_norm']
    # If the same tsRNA appears multiple times, take the first one
    if len(seqs) > 0:
        tsrna_seq_list.append(seqs.iloc[0])
    else:
        tsrna_seq_list.append("")

motif_matrix, motif_vocab = extract_motif_binary(
    tsrna_seq_list,
    L_range=(8, 12),
    min_count=3  # Adjustable: Filter motifs that appear less than 3 times
)
# print(f"motif_matrix 形状: {motif_matrix.shape}")

# PCA
pca_dim = 32
svd = TruncatedSVD(n_components=pca_dim, random_state=RANDOM_STATE)
motif_pca_features = svd.fit_transform(motif_matrix)
motif_pca_cols = [f"motif_pca_{i}" for i in range(pca_dim)]

for i in range(pca_dim):
    processed_df[f"motif_pca_{i}"] = 0.0

tsrna_ids = processed_df['tsrna_id_id'].values 
motif_pca_array = np.zeros((len(processed_df), pca_dim), dtype=np.float32)

for idx, tsrna_id in enumerate(tsrna_ids):
    motif_pca_array[idx] = motif_pca_features[tsrna_id]

for i in range(pca_dim):
    processed_df[f"motif_pca_{i}"] = motif_pca_array[:, i]


# -------------------------------
# 6. Constructing a dictionary of tsRNA and disease feature
# -------------------------------
tsrna_features = {}
tsrna_cols = ['tsrna_type_id', 'tsrna_type_1_id', 'tsrna_type_2_id', 'Isotype_Anticodon_id', 'Seq_Length_id']
# tsrna_num_cols = ['seq_len']
for _, row in processed_df.iterrows():
    tid = row['tsrna_id_id']
    if tid in tsrna_features:
        continue
    tsrna_features[tid] = row[tsrna_cols].values.astype(np.int64)

# tsRNA -> sequence mapping (used for generating negative samples)
tsrna_seq_map = processed_df.groupby('tsrna_id_id')['sequence_norm'].first().to_dict()

disease_features = {}
disease_cols = ['icd_id', 'icd_1_id', 'icd_2_id', 'organ_id']
for _, row in processed_df.iterrows():
    did = row['disease_id']
    if did in disease_features:
        continue
    disease_features[did] = row[disease_cols].values.astype(np.int64)

# print(f"✅ tsRNA types: {len(tsrna_features)}")
# print(f"✅ disease types: {len(disease_features)}")

# -------------------------------
# 7. Association matrix
# -------------------------------
num_tsrnas = len(le_tsRNA_id.classes_)   
num_diseases = len(le_disease.classes_)  

adj_matrix = np.zeros((num_tsrnas, num_diseases), dtype=np.int8)

for _, row in processed_df.iterrows():
    ts_idx = int(row['tsrna_id_id'])    
    dis_idx = int(row['disease_id'])    
    adj_matrix[ts_idx, dis_idx] = 1

num_positive = adj_matrix.sum()

# -------------------------------
# 7+. Perform PCA dimensionality reduction on the tsRNA/disease association matrix
# -------------------------------
pca_dim = 32

# tsRNA PCA
pca_tsRNA = PCA(n_components=pca_dim, random_state=RANDOM_STATE)
tsrna_pca_features = pca_tsRNA.fit_transform(adj_matrix)  # shape: (num_tsrnas, pca_dim)

# disease PCA
pca_disease = PCA(n_components=pca_dim, random_state=RANDOM_STATE)
disease_pca_features = pca_disease.fit_transform(adj_matrix.T)  # shape: (num_diseases, pca_dim)

tsrna_pca_cols = [f'tsrna_pca_{i}' for i in range(pca_dim)]
disease_pca_cols = [f'disease_pca_{i}' for i in range(pca_dim)]


# ======================================================
# 🚀 Negative sampling strategy
# ======================================================

def compute_gip_similarity(adj_matrix, gamma=1.0):
    profiles = adj_matrix
    norm_sq = np.sum(profiles**2, axis=1)
    dist_sq = norm_sq[:, None] + norm_sq[None, :] - 2 * np.dot(profiles, profiles.T)
    dist_sq = np.clip(dist_sq, 0, None)
    gamma = gamma / np.mean(dist_sq[dist_sq != 0])  
    sim = np.exp(-gamma * dist_sq)
    return sim

tsRNA_sim = compute_gip_similarity(adj_matrix)
seq_sim_matrix = cosine_similarity(motif_matrix)

alpha = 0  
fused_sim_matrix = alpha * tsRNA_sim + (1 - alpha) * seq_sim_matrix


neg_pairs = set()
tsRNA_idx_neg, disease_idx_neg = [], []
top_k = 20 
print(f"top_k = {top_k}")

print("🔹 Start negative sample sampling...")
for ts_idx in tqdm(range(num_tsrnas), desc="Round 1: Negative Sampling"):
    if len(neg_pairs) >= num_positive:
        break

    sim_scores = fused_sim_matrix[ts_idx, :]
    top_sim_indices = np.argsort(sim_scores)[::-1][:top_k+1]

    forbidden_diseases = set()
    for sim_ts in top_sim_indices:
        linked = np.where(adj_matrix[sim_ts] == 1)[0]
        forbidden_diseases.update(linked)
    
    current_linked = np.where(adj_matrix[ts_idx] == 1)[0]
    forbidden_diseases.update(current_linked)

    candidates = list(set(range(num_diseases)) - forbidden_diseases)
    np.random.shuffle(candidates)

    for dis_id in candidates:
        if (ts_idx, dis_id) not in neg_pairs:
            neg_pairs.add((ts_idx, dis_id))
            tsRNA_idx_neg.append(ts_idx)
            disease_idx_neg.append(dis_id)
            break
        

current_neg_count = len(tsRNA_idx_neg)
remaining_count = int(num_positive - current_neg_count)

if remaining_count > 0:
    print(f"Starting Round 2: Need {remaining_count} more unique negative samples.")

    all_ts_indices = np.arange(num_tsrnas)
    np.random.shuffle(all_ts_indices)
    selected_ts_indices = np.random.choice(all_ts_indices, size=max(remaining_count * 3, num_tsrnas), replace=True)

    added = 0
    for ts_idx in selected_ts_indices:
        if added >= remaining_count:
            break

        sim_scores = fused_sim_matrix[ts_idx, :]
        top_sim_indices = np.argsort(sim_scores)[::-1][:top_k + 1]

        forbidden_diseases = set()
        for similar_ts in top_sim_indices:
            linked_diseases = np.where(adj_matrix[similar_ts] == 1)[0]
            forbidden_diseases.update(linked_diseases)
        
        current_linked = np.where(adj_matrix[ts_idx] == 1)[0]
        forbidden_diseases.update(current_linked)

        candidate_diseases = list(set(range(num_diseases)) - forbidden_diseases)
        if len(candidate_diseases) == 0:
            continue

        np.random.shuffle(candidate_diseases)
        for chosen_disease in candidate_diseases:
            pair = (int(ts_idx), int(chosen_disease))
            if pair not in neg_pairs:
                neg_pairs.add(pair)
                tsRNA_idx_neg.append(int(ts_idx))
                disease_idx_neg.append(int(chosen_disease))
                added += 1
                break

    print(f"Round 2 added {added} unique negative samples.")

else:
    print("No additional samples needed.")

assert len(tsRNA_idx_neg) == num_positive, f"Negative sample count {len(tsRNA_idx_neg)} != positive {num_positive}"
assert len(set(zip(tsRNA_idx_neg, disease_idx_neg))) == num_positive, "Duplicate negative pairs found!"


# -------------------------------
# 8. Negative samples DataFrame
# -------------------------------
neg_data = []
for ts_id, dis_id in zip(tsRNA_idx_neg, disease_idx_neg):
    ts_feat = tsrna_features[ts_id]
    ts_pca_feat = tsrna_pca_features[ts_id]
    motif_feat = motif_pca_features[ts_id]   
    dis_feat = disease_features[dis_id]
    dis_pca_feat = disease_pca_features[dis_id]

    row = [ts_id] + ts_feat.tolist() + ts_pca_feat.tolist() + motif_feat.tolist() + [dis_id] + dis_feat.tolist() + dis_pca_feat.tolist() + [0]
    neg_data.append(row)

neg_df = pd.DataFrame(neg_data, columns=[
    'tsrna_id_id', 'tsrna_type_id', 'tsrna_type_1_id', 'tsrna_type_2_id',
    'Isotype_Anticodon_id', 'Seq_Length_id'
] + tsrna_pca_cols + motif_pca_cols + [
    'disease_id', 'icd_id', 'icd_1_id', 'icd_2_id', 'organ_id'
] + disease_pca_cols + ['label'])


# -------------------------------
# 9. Combine positive and negative samples
# -------------------------------
# 1) tsRNA PCA mapping
tsrna_ids = processed_df['tsrna_id_id'].values
tsrna_pca_array = tsrna_pca_features[tsrna_ids]  # shape: (num_samples, pca_dim)
tsrna_pca_df = pd.DataFrame(tsrna_pca_array, columns=tsrna_pca_cols)

# 2) disease PCA mapping
disease_ids = processed_df['disease_id'].values
disease_pca_array = disease_pca_features[disease_ids]  # shape: (num_samples, pca_dim)
disease_pca_df = pd.DataFrame(disease_pca_array, columns=disease_pca_cols)

# 3) concat
processed_df = pd.concat([processed_df, tsrna_pca_df, disease_pca_df], axis=1)

# 4) Positive and negative sample construction
pos_df = processed_df.copy()
pos_df['label'] = 1

full_df = pd.concat([pos_df, neg_df], ignore_index=True)
full_df = full_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

full_df = full_df.copy()

# Uniform column order
feature_cols = [
    'tsrna_id_id', 'tsrna_type_id', 'tsrna_type_1_id', 'tsrna_type_2_id', 'Isotype_Anticodon_id', 'Seq_Length_id',
    'disease_id', 'organ_id', 'icd_id', 'icd_1_id', 'icd_2_id'
] + tsrna_pca_cols + motif_pca_cols + disease_pca_cols
full_df = full_df[feature_cols + ['label']]

cat_cols = ['tsrna_id_id', 'tsrna_type_id', 'tsrna_type_1_id', 'tsrna_type_2_id', 'Isotype_Anticodon_id', 'Seq_Length_id',
            'disease_id', 'organ_id', 'icd_id', 'icd_1_id', 'icd_2_id']
full_df[cat_cols] = full_df[cat_cols].astype(np.int64)

# -------------------------------
# 10. Save results
# -------------------------------
output_path = os.path.join(OUTPUT_DIR, 'full_dataset.csv')
full_df.to_csv(output_path, index=False)
print(f"✅ Successfully saved: {output_path}")

assert len(le_tsRNA_id.classes_) == len(tsrna_features), "The number of tsRNA encodings is inconsistent with the feature dictionary!"
assert len(le_disease.classes_) == len(disease_features), "The number of disease codes does not match the feature dictionary!"

# Save label encoder
encoders = {
    'tsrna_id': le_tsRNA_id,
    'tsrna_type': le_tsRNA_type,
    'tsrna_type_1': le_tsRNA_type_1,
    'tsrna_type_2': le_tsRNA_type_2,
    'Isotype_Anticodon': le_Isotype_Anticodon,
    'Seq_Length': le_Seq_Length,
    'disease': le_disease,
    'organ': le_organ,
    'icd': le_icd,
    'icd_1': le_icd_1,
    'icd_2': le_icd_2,
}
joblib.dump(encoders, os.path.join(OUTPUT_DIR, 'encoders.pkl'))

# Save vocab sizes (for model embedding layer)
vocab_sizes = {
    'tsrna_id': int(len(le_tsRNA_id.classes_)),
    'tsrna_type': int(len(le_tsRNA_type.classes_)),
    'tsrna_type_1': int(len(le_tsRNA_type_1.classes_)),
    'tsrna_type_2': int(len(le_tsRNA_type_2.classes_)),
    'Isotype_Anticodon': int(len(le_Isotype_Anticodon.classes_)),
    'Seq_Length': int(len(le_Seq_Length.classes_)),
    'disease': int(len(le_disease.classes_)),
    'organ': int(len(le_organ.classes_)),
    'icd': int(len(le_icd.classes_)),
    'icd_1': int(len(le_icd_1.classes_)),
    'icd_2': int(len(le_icd_2.classes_))
}


#  PCA feature dimension adjustment
vocab_sizes["tsrna_pca_dim"] = len(tsrna_pca_cols)
vocab_sizes["disease_pca_dim"] = len(disease_pca_cols)
# vocab_sizes["motif_pca_dim"] = len(motif_pca_cols)
# vocab_sizes["tsrna_pca_dim"] = 0
# vocab_sizes["disease_pca_dim"] = 0
vocab_sizes["motif_pca_dim"] = 0
with open(os.path.join(OUTPUT_DIR, 'vocab_sizes.json'), 'w') as f:
    json.dump(vocab_sizes, f, indent=2)

# Save field configuration
field_config = {
    "fields": [
        {"name": "tsrna_id", "type": "categorical", "col": "tsrna_id_id"},
        {"name": "tsrna_type", "type": "categorical", "col": "tsrna_type_id"},
        {"name": "tsrna_type_1", "type": "categorical", "col": "tsrna_type_1_id"},
        {"name": "tsrna_type_2", "type": "categorical", "col": "tsrna_type_2_id"},
        {"name": "Isotype_Anticodon", "type": "categorical", "col": "Isotype_Anticodon_id"},
        {"name": "Seq_Length", "type": "categorical", "col": "Seq_Length_id"},
        {"name": "disease", "type": "categorical", "col": "disease_id"},
        {"name": "organ", "type": "categorical", "col": "organ_id"},
        {"name": "icd", "type": "categorical", "col": "icd_id"},
        {"name": "icd_1", "type": "categorical", "col": "icd_1_id"},
        {"name": "icd_2", "type": "categorical", "col": "icd_2_id"},
    ] + [
        {"name": col, "type": "numerical", "col": col}
        for col in tsrna_pca_cols + disease_pca_cols
    ] + [
        {"name": col, "type": "numerical", "col": col}
        for col in motif_pca_cols
    ]
}
with open(os.path.join(OUTPUT_DIR, 'field_config.json'), 'w') as f:
    json.dump(field_config, f, indent=2)

# Save other auxiliary data
np.save(os.path.join(OUTPUT_DIR, 'adj_matrix.npy'), adj_matrix)
np.save(os.path.join(OUTPUT_DIR, 'tsrna_ids.npy'), np.array(le_tsRNA_id.classes_))
np.save(os.path.join(OUTPUT_DIR, 'disease_ids.npy'), np.array(le_disease.classes_))

# Save the mapping dictionary from tsRNA to sequence
import pickle
with open(os.path.join(OUTPUT_DIR, 'tsrna_seq_map.pkl'), 'wb') as f:
    pickle.dump(tsrna_seq_map, f)

print(f"Collum name: {list(full_df.columns)}")
print("Dataset generation completed!")

# -------------------------------
# 11. Save as raw_data.json
# -------------------------------

# Convert full_df to a list of dictionaries, with each field processed separately
raw_data_list = []
for _, row in full_df.iterrows():
    sample = {}

    # Categorical fields
    sample['tsrna_id_id'] = int(row['tsrna_id_id'])
    sample['tsrna_type_id'] = int(row['tsrna_type_id'])
    sample['tsrna_type_1_id'] = int(row['tsrna_type_1_id'])
    sample['tsrna_type_2_id'] = int(row['tsrna_type_2_id'])
    sample['Isotype_Anticodon_id'] = int(row['Isotype_Anticodon_id'])
    sample['Seq_Length_id'] = int(row['Seq_Length_id'])
    sample['disease_id'] = int(row['disease_id'])
    sample['organ_id'] = int(row['organ_id'])
    sample['icd_id'] = int(row['icd_id'])
    sample['icd_1_id'] = int(row['icd_1_id'])
    sample['icd_2_id'] = int(row['icd_2_id'])

    # Label
    sample['label'] = int(row['label'])

    # tsRNA PCA features
    for col in tsrna_pca_cols:
        sample[col] = float(row[col])

    # disease PCA features
    for col in disease_pca_cols:
        sample[col] = float(row[col])

    # tsrna_motif
    for col in motif_pca_cols:
        sample[col] = float(row[col])

    raw_data_list.append(sample)

# Saved as JSON
raw_data_path = os.path.join(OUTPUT_DIR, 'raw_data.json')
with open(raw_data_path, 'w', encoding='utf-8') as f:
    json.dump(raw_data_list, f, indent=2, ensure_ascii=False)


