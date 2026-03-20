'''
@Author: Wang Dong
@Date: 2025.10.27
@Description: Data preprocessing and encodeing
@Negative sampling strategy: motif similarity based
'''

import torch
from torch.utils.data import DataLoader
import json, os, random, glob
import numpy as np
import pandas as pd
from datetime import datetime

from rfm_pure import RFM
from train_rfm import TsRNADiseaseDataset, collate_fn

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    seed = 42
    set_seed(seed)
    data_dir = "./data"
    train_path = os.path.join(data_dir, "raw_data.json")
    case_path = os.path.join(data_dir, "case_study.json")
    vocab_path = os.path.join(data_dir, "vocab_sizes.json")
    model_dir = "./saved_model"

    # Loading the tsRNA and disease names
    tsrna_names = np.load(os.path.join(data_dir, "tsrna_ids.npy"), allow_pickle=True)
    disease_names = np.load(os.path.join(data_dir, "disease_ids.npy"), allow_pickle=True)

    with open(vocab_path, "r") as f:
        vocab_sizes = json.load(f)

    tsrna_pca_dim = vocab_sizes.get("tsrna_pca_dim", 0)
    disease_pca_dim = vocab_sizes.get("disease_pca_dim", 0)
    motif_pca_dim = vocab_sizes.get("motif_pca_dim", 0)

    batch_size = 32
    embedding_size = 32
    hidden_units = 32
    att_layers = 2
    head_num = 4
    mlp_list = [64]
    group_hidden_dimension = 8
    drop_rate_att = 0.1
    dp_rate_amp = 0.1
    lr = 1e-3
    weight_decay = 1e-5
    epochs = 200

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    model = RFM(
        vocab_sizes=vocab_sizes,
        embedding_size=embedding_size,
        hidden_units=hidden_units,
        att_layers=att_layers,
        head_num=head_num,
        mlp_list=mlp_list,
        group_hidden_dimension=group_hidden_dimension,
        drop_rate_att=drop_rate_att,
        dp_rate_amp=dp_rate_amp,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Check if there is already a model file
    os.makedirs(model_dir, exist_ok=True)
    model_files = glob.glob(os.path.join(model_dir, "RFM_case_full_*.pt"))

    if len(model_files) > 0:
        # Load the latest model
        latest_model = max(model_files, key=os.path.getmtime)
        state_dict = torch.load(latest_model, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"Existing model detected, loading: {latest_model}")
    else:
        print("No saved model detected, starting to train model on full training set ...")
        train_dataset = TsRNADiseaseDataset(
            train_path,
            tsrna_pca_dim=tsrna_pca_dim,
            disease_pca_dim=disease_pca_dim,
            motif_pca_dim=motif_pca_dim
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0
            for batch in train_loader:
                for k in batch:
                    batch[k] = batch[k].to(device)
                optimizer.zero_grad()
                loss = model.calculate_loss(batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f}")

        # save the model
        model_path = f"{model_dir}/RFM_case_full_{datetime.now().strftime('%Y%m%d-%H%M%S')}.pt"
        torch.save(model.state_dict(), model_path)

    print("\n Start predicting the case study dataset ...")
    case_dataset = TsRNADiseaseDataset(
        case_path,
        tsrna_pca_dim=tsrna_pca_dim,
        disease_pca_dim=disease_pca_dim,
        motif_pca_dim=motif_pca_dim
    )
    case_loader = DataLoader(case_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model.eval()
    all_probs, all_pairs = [], []

    with torch.no_grad():
        for batch in case_loader:
            for k in batch:
                batch[k] = batch[k].to(device)
            logits = model.forward(batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            for i in range(len(probs)):
                tsrna_id = batch["tsrna_id"][i].item()
                disease_id = batch["disease"][i].item()
                all_pairs.append((tsrna_id, disease_id))

    # Read the tsRNA sequence mapping table
    import pickle
    tsrna_seq_path = os.path.join(data_dir, "tsrna_seq_map.pkl")
    if os.path.exists(tsrna_seq_path):
        with open(tsrna_seq_path, "rb") as f:
            tsrna_seq_map = pickle.load(f)
        print(f"{len(tsrna_seq_map)} tsRNA sequence mappings have been loaded.")
    else:
        print("'tsrna_seq_map.pkl' not found, sequence information will not be added.")
        tsrna_seq_map = {}

    # Restore the code to its actual name
    decoded_results = []
    for (tsrna_id, disease_id), score in zip(all_pairs, all_probs):
        tsrna_name = tsrna_names[tsrna_id] if tsrna_id < len(tsrna_names) else f"unknown_{tsrna_id}"
        disease_name = disease_names[disease_id] if disease_id < len(disease_names) else f"unknown_{disease_id}"
        seq = tsrna_seq_map.get(tsrna_id, "N/A")  
        decoded_results.append((tsrna_name, seq, disease_name, score))

    # Output and save the sorting results
    results = pd.DataFrame(decoded_results, columns=["tsRNA_name", "sequence", "disease_name", "score"])
    results = results.sort_values(by="score", ascending=False).reset_index(drop=True)

    os.makedirs("./results", exist_ok=True)
    csv_path = f"./results/case_study_prediction_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    results.to_csv(csv_path, index=False)
    print(f"The prediction results have been saved to: {csv_path}")
    print("\n Top 10：")
    print(results.head(10))


if __name__ == "__main__":
    main()
