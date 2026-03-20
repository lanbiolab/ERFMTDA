'''
@Author: Wang Dong
@Date: 2025.10.27
@Description: Data preprocessing and encodeing
@Negative sampling strategy: random sampling
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


RAW_DATA_PATH = 'tsRNA-disease.xlsx'       
OUTPUT_DIR = './data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 1983
np.random.seed(RANDOM_STATE)


print("🚀 开始预处理...")
df_raw = pd.read_excel(RAW_DATA_PATH)
print(f"✅ 原始数据加载完成，共 {len(df_raw)} 条记录")
before_dedup = len(df_raw)
df_raw = df_raw.drop_duplicates()
print(f"🧹 去重完成：去掉 {before_dedup - len(df_raw)} 条重复记录，剩余 {len(df_raw)} 条")
df_raw['Seq Length'] = df_raw['Seq Length'].astype(str)


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



processed_df = df_encoded.copy()
processed_df['sequence'] = df_raw['tsRNA Sequence'].values
processed_df['sequence_norm'] = processed_df['sequence'].str.upper().str.replace("T", "U")
id_cols = [
    'tsrna_id_id', 'tsrna_type_id', 'tsrna_type_1_id', 'tsrna_type_2_id', 
    'Isotype_Anticodon_id', 'Seq_Length_id', 
    'disease_id', 'organ_id', 'icd_id', 'icd_1_id', 'icd_2_id'
]
processed_df[id_cols] = processed_df[id_cols].astype(int)



def extract_motif_binary(sequences, L_range=(12, 14), min_count=3):
    motif_counter = Counter()
    tsrna_motif_list = []

    for seq in tqdm(sequences, desc="Counting motifs"):
        s = seq.upper().replace("T", "U")
        seen = set()
        for L in range(L_range[0], L_range[1] + 1):
            for i in range(len(s) - L + 1):
                seen.add(s[i:i+L])
        motif_counter.update(seen)
        tsrna_motif_list.append(seen)

    motif_vocab = [m for m, c in motif_counter.items() if c >= min_count]

    motif_index = {m: i for i, m in enumerate(motif_vocab)}
    motif_matrix = np.zeros((len(sequences), len(motif_vocab)), dtype=np.float32)

    for i, seen in enumerate(tsrna_motif_list):
        for m in seen:
            if m in motif_index:
                motif_matrix[i, motif_index[m]] = 1

    return motif_matrix, motif_vocab



tsrna_seq_list = []
for tsrna_name in le_tsRNA_id.classes_:
    seqs = processed_df.loc[processed_df['tsrna_id_id'] ==
                            le_tsRNA_id.transform([tsrna_name])[0], 'sequence_norm']
    if len(seqs) > 0:
        tsrna_seq_list.append(seqs.iloc[0])
    else:
        tsrna_seq_list.append("")

motif_matrix, motif_vocab = extract_motif_binary(
    tsrna_seq_list,
    L_range=(8, 14),
    min_count=3
)

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


tsrna_features = {}
tsrna_cols = ['tsrna_type_id', 'tsrna_type_1_id', 'tsrna_type_2_id', 'Isotype_Anticodon_id', 'Seq_Length_id']
for _, row in processed_df.iterrows():
    tid = row['tsrna_id_id']
    if tid in tsrna_features:
        continue
    tsrna_features[tid] = row[tsrna_cols].values.astype(np.int64)

tsrna_seq_map = processed_df.groupby('tsrna_id_id')['sequence_norm'].first().to_dict()

disease_features = {}
disease_cols = ['icd_id', 'icd_1_id', 'icd_2_id', 'organ_id']
for _, row in processed_df.iterrows():
    did = row['disease_id']
    if did in disease_features:
        continue
    disease_features[did] = row[disease_cols].values.astype(np.int64)



num_tsrnas = len(le_tsRNA_id.classes_)   
num_diseases = len(le_disease.classes_)  

adj_matrix = np.zeros((num_tsrnas, num_diseases), dtype=np.int8)

for _, row in processed_df.iterrows():
    ts_idx = int(row['tsrna_id_id'])    
    dis_idx = int(row['disease_id'])    
    adj_matrix[ts_idx, dis_idx] = 1

num_positive = adj_matrix.sum()


pca_dim = 32 

# tsRNA PCA
pca_tsRNA = PCA(n_components=pca_dim, random_state=RANDOM_STATE)
tsrna_pca_features = pca_tsRNA.fit_transform(adj_matrix)  # shape: (num_tsrnas, 16)

# disease PCA
pca_disease = PCA(n_components=pca_dim, random_state=RANDOM_STATE)
disease_pca_features = pca_disease.fit_transform(adj_matrix.T)  # shape: (num_diseases, 16)

tsrna_pca_cols = [f'tsrna_pca_{i}' for i in range(pca_dim)]
disease_pca_cols = [f'disease_pca_{i}' for i in range(pca_dim)]


neg_positions = np.where(adj_matrix == 0)
neg_ts_idx = neg_positions[0]
neg_dis_idx = neg_positions[1]

sampled = np.random.choice(len(neg_ts_idx), size=num_positive, replace=False)
neg_ts_sampled = neg_ts_idx[sampled]
neg_dis_sampled = neg_dis_idx[sampled]

neg_data = []
for i in tqdm(range(num_positive), desc="Generating negative samples"):
    ts_id = neg_ts_sampled[i]
    dis_id = neg_dis_sampled[i]
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


# 1) tsRNA PCA 
tsrna_ids = processed_df['tsrna_id_id'].values
tsrna_pca_array = tsrna_pca_features[tsrna_ids]  # shape: (num_samples, pca_dim)
tsrna_pca_df = pd.DataFrame(tsrna_pca_array, columns=tsrna_pca_cols)

# 2) disease PCA 
disease_ids = processed_df['disease_id'].values
disease_pca_array = disease_pca_features[disease_ids]  # shape: (num_samples, pca_dim)
disease_pca_df = pd.DataFrame(disease_pca_array, columns=disease_pca_cols)


processed_df = pd.concat([processed_df, tsrna_pca_df, disease_pca_df], axis=1)

pos_df = processed_df.copy()
pos_df['label'] = 1

full_df = pd.concat([pos_df, neg_df], ignore_index=True)
full_df = full_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

full_df = full_df.copy()

feature_cols = [
    'tsrna_id_id', 'tsrna_type_id', 'tsrna_type_1_id', 'tsrna_type_2_id', 'Isotype_Anticodon_id', 'Seq_Length_id',
    'disease_id', 'organ_id', 'icd_id', 'icd_1_id', 'icd_2_id'
] + tsrna_pca_cols + motif_pca_cols + disease_pca_cols
full_df = full_df[feature_cols + ['label']]

cat_cols = ['tsrna_id_id', 'tsrna_type_id', 'tsrna_type_1_id', 'tsrna_type_2_id', 'Isotype_Anticodon_id', 'Seq_Length_id',
            'disease_id', 'organ_id', 'icd_id', 'icd_1_id', 'icd_2_id']
full_df[cat_cols] = full_df[cat_cols].astype(np.int64)


output_path = os.path.join(OUTPUT_DIR, 'full_dataset.csv')
full_df.to_csv(output_path, index=False)
print(f"Saved: {output_path}")


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


vocab_sizes["tsrna_pca_dim"] = len(tsrna_pca_cols)
vocab_sizes["disease_pca_dim"] = len(disease_pca_cols)
# vocab_sizes["motif_pca_dim"] = len(motif_pca_cols)
# vocab_sizes["tsrna_pca_dim"] = 0
# vocab_sizes["disease_pca_dim"] = 0
vocab_sizes["motif_pca_dim"] = 0
with open(os.path.join(OUTPUT_DIR, 'vocab_sizes.json'), 'w') as f:
    json.dump(vocab_sizes, f, indent=2)

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

np.save(os.path.join(OUTPUT_DIR, 'adj_matrix.npy'), adj_matrix)
np.save(os.path.join(OUTPUT_DIR, 'tsrna_ids.npy'), np.array(le_tsRNA_id.classes_))
np.save(os.path.join(OUTPUT_DIR, 'disease_ids.npy'), np.array(le_disease.classes_))

import pickle
with open(os.path.join(OUTPUT_DIR, 'tsrna_seq_map.pkl'), 'wb') as f:
    pickle.dump(tsrna_seq_map, f)

print(f"Collum names: {list(full_df.columns)}")
print("Dataset generation completed")

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

raw_data_path = os.path.join(OUTPUT_DIR, 'raw_data.json')
with open(raw_data_path, 'w', encoding='utf-8') as f:
    json.dump(raw_data_list, f, indent=2, ensure_ascii=False)

print(f"✅ raw_data.json 已保存至: {raw_data_path}")

