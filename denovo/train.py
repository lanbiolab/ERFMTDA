'''
@Author: Wang Dong
@Date: 2025.10.27
@Description: Data preprocessing and encodeing
@Negative sampling strategy: motif similarity based
'''

import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
import json, os, random
import numpy as np
from datetime import datetime
from sklearn.metrics import auc
from rfm_pure import RFM
from train_rfm import TsRNADiseaseDataset, collate_fn
from evaluate import evaluate

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_logger(log_dir="./logs"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(log_dir, f"train_{timestamp}.log")

    def log(msg):
        print(msg)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    return log

def average_curves(fold_metrics):
    '''Calculate the average of interpolated ROC and PR curves'''
    mean_fpr = np.linspace(0, 1, 100)
    mean_recall = np.linspace(0, 1, 100)

    tprs = []
    precisions = []

    for m in fold_metrics:
        tpr_interp = np.interp(mean_fpr, m["fpr"], m["tpr"])
        tprs.append(tpr_interp)
        recall_curve = np.array(m["recall_curve"])
        precision_curve = np.array(m["precision_curve"])
        order = np.argsort(recall_curve)
        recall_curve = recall_curve[order]
        precision_curve = precision_curve[order]
        prec_interp = np.interp(mean_recall, recall_curve, precision_curve)
        precisions.append(prec_interp)

    mean_tpr = np.mean(tprs, axis=0)
    mean_precision = np.mean(precisions, axis=0)

    mean_auc = auc(mean_fpr, mean_tpr)
    mean_aupr = auc(mean_recall, mean_precision)

    return {
        "mean_fpr": mean_fpr,
        "mean_tpr": mean_tpr,
        "mean_recall": mean_recall,
        "mean_precision": mean_precision,
        "mean_AUC": mean_auc,
        "mean_AUPR": mean_aupr
    }


def train_one_fold(model, train_loader, val_loader, optimizer, scheduler, device,
                   epochs=50, log_file=None, hparams=None):
    last_metrics = None

    if log_file and hparams is not None:
        log_file.write("===== Hyperparameters =====\n")
        for k, v in hparams.items():
            log_file.write(f"{k}: {v}\n")
        log_file.write("===========================\n\n")

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

        metrics = evaluate(model, val_loader, device)
        val_auc = metrics["AUC"]
        last_metrics = metrics  

        if scheduler is not None:
            scheduler.step(val_auc)
            current_lr = scheduler.optimizer.param_groups[0]['lr']
        else:
            current_lr = optimizer.param_groups[0]['lr']

        log_text = (f"Epoch {epoch} | Loss: {avg_loss:.4f} | "
                    f"AUC: {metrics['AUC']:.4f}, "
                    f"AUPR: {metrics['AUPR']:.4f}, "
                    f"Acc: {metrics['Accuracy']:.4f}, "
                    f"Prec: {metrics['Precision']:.4f}, "
                    f"Rec: {metrics['Recall']:.4f}, "
                    f"F1: {metrics['F1']:.4f} | "
                    f"LR: {current_lr:.6f}\n")
        
        print(log_text.strip())
        if log_file:
            log_file.write(log_text)
        
    if log_file and last_metrics is not None:
        log_file.write("\n===== Final Epoch Metrics =====\n")
        for k, v in last_metrics.items():
            if isinstance(v, (int, float)):
                log_file.write(f"{k}: {v:.4f}\n")
        log_file.write("========================\n")

    return last_metrics


def main(use_scheduler=False):
    seed = 42
    set_seed(seed) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    data_dir = "./data/de_novo_disease"
    vocab_path = "./data/vocab_sizes.json"
    with open(vocab_path, "r") as f:
        vocab_sizes = json.load(f)

    # Extract PCA feature dimensions
    tsrna_pca_dim = vocab_sizes.get("tsrna_pca_dim", 0)
    disease_pca_dim = vocab_sizes.get("disease_pca_dim", 0)
    motif_pca_dim = vocab_sizes.get("motif_pca_dim",0)

    train_batch_size = 32
    val_batch_size = 32
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

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"./logs/de_novo_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    all_results = []

    # Traverse the cold start dataset for each disease
    disease_dirs = sorted(
        [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    )
    num_diseases = len(disease_dirs)
    print(f"A total of {num_diseases} diseases were detected in the De novo dataset")

    for disease_id in disease_dirs:
        train_path = os.path.join(data_dir, disease_id, "train.json")
        test_path = os.path.join(data_dir, disease_id, "test.json")

        print(f"\n===== Cold-Disease {disease_id} / {num_diseases} =====")

        train_dataset = TsRNADiseaseDataset(train_path,
                                            tsrna_pca_dim=tsrna_pca_dim,
                                            disease_pca_dim=disease_pca_dim,
                                            motif_pca_dim=motif_pca_dim)
        test_dataset = TsRNADiseaseDataset(test_path,
                                           tsrna_pca_dim=tsrna_pca_dim,
                                           disease_pca_dim=disease_pca_dim,
                                           motif_pca_dim=motif_pca_dim)
            
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False, collate_fn=collate_fn)

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

        if use_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.5,
                patience=3,
                min_lr=1e-5,
            )
        else:
            scheduler = None

        # hyperparameter dictionary    
        hparams = dict(
            seed=seed,
            tsrna_pca_dim=tsrna_pca_dim,
            disease_pca_dim=disease_pca_dim,
            motif_pca_dim=motif_pca_dim,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            embedding_size=embedding_size,
            hidden_units=hidden_units,
            att_layers=att_layers,
            head_num=head_num,
            mlp_list=mlp_list,
            group_hidden_dimension=group_hidden_dimension,
            drop_rate_att=drop_rate_att,
            dp_rate_amp=dp_rate_amp,
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
        )

        log_path = os.path.join(log_dir, f"disease_{disease_id}.log")
        with open(log_path, "w") as log_file:
            metrics = train_one_fold(
                model,
                train_loader,
                test_loader,
                optimizer,
                scheduler,
                device,
                epochs=epochs,
                log_file=log_file,
                hparams=hparams
            )

        all_results.append(metrics)

    # Summary results
    print("\n===== Cold-Disease Average Results =====")
    summary_path = os.path.join(log_dir, "summary.log")
    with open(summary_path, "w") as summary_file:
        summary_file.write("===== cold_disease de novo Results =====\n")

        scalar_metrics = ['AUC', 'AUPR', 'Accuracy', 'Precision', 'Recall', 'F1']

        for metric in scalar_metrics:
            values = [res[metric] for res in all_results if metric in res]
            mean_val = np.mean(values)
            line = f"{metric}: {mean_val:.4f} ({values})\n"
            print(line.strip())
            summary_file.write(line)

    print(f"\n✅ Summary results saved to {summary_path}")


    avg_metrics = average_curves(all_results)
    print("Mean AUC:", avg_metrics["mean_AUC"])
    print("Mean AUPR:", avg_metrics["mean_AUPR"])
    np.savez(os.path.join(log_dir, "ERFMTDA_denovo_avg_curves.npz"),
            fpr=avg_metrics["mean_fpr"],
            tpr=avg_metrics["mean_tpr"],
            recall=avg_metrics["mean_recall"],
            precision=avg_metrics["mean_precision"],
            mean_AUC=avg_metrics["mean_AUC"],
            mean_AUPR=avg_metrics["mean_AUPR"])


if __name__ == "__main__":
    # main(use_scheduler=True)
    main(use_scheduler=False)
