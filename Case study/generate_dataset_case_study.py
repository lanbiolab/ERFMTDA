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
from collections import Counter
from tqdm import tqdm
from itertools import product


RAW_DATA_PATH = 'tsRNA-disease.xlsx'        
OUTPUT_DIR = './data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 1983
np.random.seed(RANDOM_STATE)

# raw data
df_raw = pd.read_excel(RAW_DATA_PATH)
before_dedup = len(df_raw)
df_raw = df_raw.drop_duplicates()
df_raw['Seq Length'] = df_raw['Seq Length'].astype(str)

# label encoding
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

# Merge features
processed_df = df_encoded.copy()
processed_df['sequence'] = df_raw['tsRNA Sequence'].values
processed_df['sequence_norm'] = processed_df['sequence'].str.upper().str.replace("T", "U")

id_cols = [
    'tsrna_id_id', 'tsrna_type_id', 'tsrna_type_1_id', 'tsrna_type_2_id', 
    'Isotype_Anticodon_id', 'Seq_Length_id', 
    'disease_id', 'organ_id', 'icd_id', 'icd_1_id', 'icd_2_id'
]
processed_df[id_cols] = processed_df[id_cols].astype(int)

# motif statistics
def extract_motif_binary(sequences, L_range=(12, 14), min_count=3):
    motif_counter = Counter()
    tsrna_motif_list = []

    # Count all motifs and their global occurrences
    for seq in tqdm(sequences, desc="Counting motifs"):
        s = seq.upper().replace("T", "U")
        seen = set()
        for L in range(L_range[0], L_range[1] + 1):
            for i in range(len(s) - L + 1):
                seen.add(s[i:i+L])
        motif_counter.update(seen)
        tsrna_motif_list.append(seen)

    # Filter out low-frequency motifs and reduce dimensionality (adjustable as needed)
    motif_vocab = [m for m, c in motif_counter.items() if c >= min_count]

    motif_index = {m: i for i, m in enumerate(motif_vocab)}
    motif_matrix = np.zeros((len(sequences), len(motif_vocab)), dtype=np.float32)

    for i, seen in enumerate(tsrna_motif_list):
        for m in seen:
            if m in motif_index:
                motif_matrix[i, motif_index[m]] = 1

    return motif_matrix, motif_vocab

# motif feature generation
motif_matrix, motif_vocab = extract_motif_binary(
    processed_df['sequence_norm'].tolist(),
    L_range=(8, 12),
    min_count=3 
)

# PCA dimensionality reduction
pca_dim = 16
pca = PCA(n_components=pca_dim, random_state=RANDOM_STATE)
motif_pca_features = pca.fit_transform(motif_matrix)

motif_pca_cols = [f"motif_pca_{i}" for i in range(pca_dim)]
motif_pca_df = pd.DataFrame(motif_pca_features, columns=motif_pca_cols)

processed_df = pd.concat([processed_df.reset_index(drop=True), motif_pca_df], axis=1)

# Construct tsRNA and disease feature dictionaries
tsrna_features = {}
tsrna_cols = ['tsrna_type_id', 'tsrna_type_1_id', 'tsrna_type_2_id', 'Isotype_Anticodon_id', 'Seq_Length_id']
for _, row in processed_df.iterrows():
    tid = row['tsrna_id_id']
    tsrna_features[tid] = row[tsrna_cols].values.astype(np.float32)
    '''
    tsrna_features[tid] = {
        "cat": row[tsrna_cat_cols].values.astype(np.int64),
        "num": row[tsrna_num_cols].values.astype(np.float32),
    }
    '''

# tsRNA -> sequence mapping (for generating negative samples)
tsrna_seq_map = processed_df.groupby('tsrna_id_id')['sequence_norm'].first().to_dict()

disease_features = {}
disease_cols = ['icd_id', 'icd_1_id', 'icd_2_id', 'organ_id']
for _, row in processed_df.iterrows():
    did = row['disease_id']
    disease_features[did] = row[disease_cols].values.astype(np.int64)

# Association matrix construction
num_tsrnas = len(le_tsRNA_id.classes_)  
num_diseases = len(le_disease.classes_) 

adj_matrix = np.zeros((num_tsrnas, num_diseases), dtype=np.int8)

for _, row in processed_df.iterrows():
    ts_idx = int(row['tsrna_id_id'])   
    dis_idx = int(row['disease_id'])    
    adj_matrix[ts_idx, dis_idx] = 1

num_positive = adj_matrix.sum()

# Perform PCA dimensionality reduction on the tsRNA/disease association matrix
pca_dim = 32 

# tsRNA PCA
pca_tsRNA = PCA(n_components=pca_dim, random_state=RANDOM_STATE)
tsrna_pca_features = pca_tsRNA.fit_transform(adj_matrix)  # shape: (num_tsrnas, 16)

# disease PCA
pca_disease = PCA(n_components=pca_dim, random_state=RANDOM_STATE)
disease_pca_features = pca_disease.fit_transform(adj_matrix.T)  # shape: (num_diseases, 16)

tsrna_pca_cols = [f'tsrna_pca_{i}' for i in range(pca_dim)]
disease_pca_cols = [f'disease_pca_{i}' for i in range(pca_dim)]

for idx, ts_id in enumerate(range(num_tsrnas)):
    tsrna_features[ts_id] = np.concatenate([tsrna_features[ts_id], tsrna_pca_features[idx]])

for idx, dis_id in enumerate(range(num_diseases)):
    disease_features[dis_id] = np.concatenate([disease_features[dis_id], disease_pca_features[idx]])


# Negative sampling strategy
def compute_gip_similarity(adj_matrix, gamma=1.0):
    profiles = adj_matrix
    norm_sq = np.sum(profiles**2, axis=1)
    dist_sq = norm_sq[:, None] + norm_sq[None, :] - 2 * np.dot(profiles, profiles.T)
    dist_sq = np.clip(dist_sq, 0, None)
    gamma = gamma / np.mean(dist_sq[dist_sq != 0])  
    sim = np.exp(-gamma * dist_sq)
    return sim

tsRNA_sim = compute_gip_similarity(adj_matrix)

tsrna_seq_list = []
for tsrna_name in le_tsRNA_id.classes_:
    seqs = processed_df.loc[processed_df['tsrna_id_id'] ==
                            le_tsRNA_id.transform([tsrna_name])[0], 'sequence_norm']
    if len(seqs) > 0:
        tsrna_seq_list.append(seqs.iloc[0])
    else:
        tsrna_seq_list.append("")

motif_matrix_unique, motif_vocab_unique = extract_motif_binary(
    tsrna_seq_list,
    L_range=(8, 12),
    min_count=3
)

seq_sim_matrix = cosine_similarity(motif_matrix_unique)

alpha = 0  
fused_sim_matrix = alpha * tsRNA_sim + (1 - alpha) * seq_sim_matrix

neg_pairs = set()
tsRNA_idx_neg, disease_idx_neg = [], []
top_k = 20  
print(f"top_k = {top_k}")

for ts_idx in tqdm(range(num_tsrnas), desc="Round 1: Negative Sampling"):
    if len(neg_pairs) >= num_positive:
        break

    sim_scores = fused_sim_matrix[ts_idx, :]
    top_sim_indices = np.argsort(sim_scores)[::-1][:top_k+1]

    forbidden_diseases = set()
    for sim_ts in top_sim_indices:
        linked = np.where(adj_matrix[sim_ts] == 1)[0]
        forbidden_diseases.update(linked)

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
print(f"✅ Final unique negative sample count: {len(tsRNA_idx_neg)} (target: {num_positive})")

# Negative samples DataFrame
neg_data = []
for ts_id, dis_id in zip(tsRNA_idx_neg, disease_idx_neg):
    ts_feat = tsrna_features[ts_id]
    motif_feat = motif_pca_features[ts_id]  
    dis_feat = disease_features[dis_id]

    row = [ts_id] + ts_feat.tolist() + motif_feat.tolist() + [dis_id] + dis_feat.tolist() + [0]
    neg_data.append(row)

neg_df = pd.DataFrame(neg_data, columns=[
    'tsrna_id_id', 'tsrna_type_id', 'tsrna_type_1_id', 'tsrna_type_2_id',
    'Isotype_Anticodon_id', 'Seq_Length_id'
] + tsrna_pca_cols + motif_pca_cols + [
    'disease_id', 'icd_id', 'icd_1_id', 'icd_2_id', 'organ_id'
] + disease_pca_cols + ['label'])


# Combine positive and negative samples
for idx, col in enumerate(tsrna_pca_cols):
    processed_df[col] = processed_df['tsrna_id_id'].map(lambda x: tsrna_pca_features[x][idx])

for idx, col in enumerate(disease_pca_cols):
    processed_df[col] = processed_df['disease_id'].map(lambda x: disease_pca_features[x][idx])
pos_df = processed_df.copy()
pos_df['label'] = 1

full_df = pd.concat([pos_df, neg_df], ignore_index=True)
full_df = full_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

# Uniform column order
feature_cols = [
    'tsrna_id_id', 'tsrna_type_id', 'tsrna_type_1_id', 'tsrna_type_2_id', 'Isotype_Anticodon_id', 'Seq_Length_id',
    'disease_id', 'organ_id', 'icd_id', 'icd_1_id', 'icd_2_id'
] + tsrna_pca_cols + motif_pca_cols + disease_pca_cols
full_df = full_df[feature_cols + ['label']]

cat_cols = ['tsrna_id_id', 'tsrna_type_id', 'tsrna_type_1_id', 'tsrna_type_2_id', 'Isotype_Anticodon_id', 'Seq_Length_id',
            'disease_id', 'organ_id', 'icd_id', 'icd_1_id', 'icd_2_id']
full_df[cat_cols] = full_df[cat_cols].astype(np.int64)

# Save results
output_path = os.path.join(OUTPUT_DIR, 'full_dataset.csv')
full_df.to_csv(output_path, index=False)

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

# PCA feature dimension adjustment
vocab_sizes["tsrna_pca_dim"] = len(tsrna_pca_cols)
vocab_sizes["disease_pca_dim"] = len(disease_pca_cols)
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
print("Dataset generation completed！")

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

    # PCA features
    for col in tsrna_pca_cols + disease_pca_cols:
        sample[col] = float(row[col])

    # Label
    sample['label'] = int(row['label'])

    # Numerical_motif
    for col in motif_pca_cols:
        sample[col] = float(row[col])

    raw_data_list.append(sample)

raw_data_path = os.path.join(OUTPUT_DIR, 'raw_data.json')
with open(raw_data_path, 'w', encoding='utf-8') as f:
    json.dump(raw_data_list, f, indent=2, ensure_ascii=False)


# ======================================================
# 🎯 12. Generate Case Study dataset
# ======================================================

# 1️⃣ Find the code of the target disease ID
target_disease_name = "Hepatocellular Carcinoma (HCC)"
if target_disease_name not in le_disease.classes_:
    raise ValueError(f"Disease not found {target_disease_name} ，Please check if the name is consistent with the original table!")

target_disease_id = int(le_disease.transform([target_disease_name])[0])
print(f"Target disease ID: {target_disease_id}")

# 2️⃣ Obtain the feature vector of the target disease
target_dis_feat = disease_features[target_disease_id]
target_dis_pca = disease_pca_features[target_disease_id]

# 3️⃣ Traverse all tsRNAs and construct samples related to the disease
case_samples = []
for ts_id in range(len(tsrna_features)):
    sample = {}
    # tsRNA category features
    sample["tsrna_id_id"] = int(ts_id)
    sample["tsrna_type_id"] = int(processed_df.loc[processed_df["tsrna_id_id"] == ts_id, "tsrna_type_id"].iloc[0])
    sample["tsrna_type_1_id"] = int(processed_df.loc[processed_df["tsrna_id_id"] == ts_id, "tsrna_type_1_id"].iloc[0])
    sample["tsrna_type_2_id"] = int(processed_df.loc[processed_df["tsrna_id_id"] == ts_id, "tsrna_type_2_id"].iloc[0])
    sample["Isotype_Anticodon_id"] = int(processed_df.loc[processed_df["tsrna_id_id"] == ts_id, "Isotype_Anticodon_id"].iloc[0])
    sample["Seq_Length_id"] = int(processed_df.loc[processed_df["tsrna_id_id"] == ts_id, "Seq_Length_id"].iloc[0])

    # disease category features
    sample["disease_id"] = target_disease_id
    # organ / icd / icd_1 / icd_2 
    sample["organ_id"] = int(processed_df.loc[processed_df["disease_id"] == target_disease_id, "organ_id"].iloc[0])
    sample["icd_id"] = int(processed_df.loc[processed_df["disease_id"] == target_disease_id, "icd_id"].iloc[0])
    sample["icd_1_id"] = int(processed_df.loc[processed_df["disease_id"] == target_disease_id, "icd_1_id"].iloc[0])
    sample["icd_2_id"] = int(processed_df.loc[processed_df["disease_id"] == target_disease_id, "icd_2_id"].iloc[0])

    # tsRNA PCA features
    for i, col in enumerate(tsrna_pca_cols):
        sample[col] = float(tsrna_pca_features[ts_id][i])

    # motif PCA features
    for i, col in enumerate(motif_pca_cols):
        sample[col] = float(motif_pca_features[ts_id][i])

    # disease PCA features
    for i, col in enumerate(disease_pca_cols):
        sample[col] = float(target_dis_pca[i])

    # No label is required here (no label is needed for prediction)
    sample["label"] = -1  # Mark as unknown sample

    case_samples.append(sample)

# 4️⃣ saved as JSON
case_study_path = os.path.join(OUTPUT_DIR, "case_study.json")
with open(case_study_path, "w", encoding="utf-8") as f:
    json.dump(case_samples, f, indent=2, ensure_ascii=False)

print(f"case_study.json has saved: {case_study_path}")
print(f"The number of samples: {len(case_samples)}")
