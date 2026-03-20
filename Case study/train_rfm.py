import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import os
from rfm_pure import RFM  
from evaluate import evaluate
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset


class TsRNADiseaseDataset(Dataset):
    def __init__(self, json_path, tsrna_pca_dim=0, disease_pca_dim=0, motif_pca_dim=32):
        
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.categorical_fields = {
            "tsrna_id_id": "tsrna_id",
            "tsrna_type_1_id": "tsrna_type_1",
            "tsrna_type_2_id": "tsrna_type_2",
            "tsrna_type_id": "tsrna_type",
            "Isotype_Anticodon_id": "Isotype_Anticodon",
            "Seq_Length_id": "Seq_Length",
            "disease_id": "disease",
            "organ_id": "organ",
            "icd_id": "icd",
            "icd_1_id": "icd_1",
            "icd_2_id": "icd_2"
        }

        self.tsrna_pca_fields = [f"tsrna_pca_{i}" for i in range(tsrna_pca_dim)]
        self.disease_pca_fields = [f"disease_pca_{i}" for i in range(disease_pca_dim)]
        self.motif_pca_fields = [f"motif_pca_{i}" for i in range(motif_pca_dim)]

        self.all_pca_fields = self.tsrna_pca_fields + self.disease_pca_fields + self.motif_pca_fields

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        batch = {}

        for old_name, new_name in self.categorical_fields.items():
            batch[new_name] = torch.tensor(row[old_name], dtype=torch.long)

        for f in self.all_pca_fields:
            if f in row:
                batch[f] = torch.tensor(row[f], dtype=torch.float)
            else:
                batch[f] = torch.tensor(0.0, dtype=torch.float)

        # label
        batch["label"] = torch.tensor(row["label"], dtype=torch.float)

        return batch


def collate_fn(batch):
    collated = {}
    for key in batch[0]:
        collated[key] = torch.stack([item[key] for item in batch], dim=0)
    return collated


