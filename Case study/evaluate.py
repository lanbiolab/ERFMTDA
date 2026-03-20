import torch
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, precision_recall_curve

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    for batch in dataloader:
        for k in batch:
            batch[k] = batch[k].to(device)

        logits = model.forward(batch)                
        probs = torch.sigmoid(logits).detach().cpu() 
        preds = (probs >= 0.5).long()                

        labels = batch["label"].detach().cpu()

        all_labels.extend(labels.tolist())
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())

    
    y_true = all_labels
    y_pred = all_preds
    y_prob = all_probs

    
    metrics = {}
    metrics["AUC"] = roc_auc_score(y_true, y_prob)
    metrics["AUPR"] = average_precision_score(y_true, y_prob)
    metrics["Accuracy"] = accuracy_score(y_true, y_pred)
    metrics["Precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["Recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["F1"] = f1_score(y_true, y_pred, zero_division=0)

    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    metrics["fpr"] = fpr
    metrics["tpr"] = tpr
    metrics["precision_curve"] = precision
    metrics["recall_curve"] = recall

    return metrics
