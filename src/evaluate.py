import torch
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, data, x, y, test_mask):
    print("\nEvaluating model...")

    model.eval()

    with torch.no_grad():
        out = model(x, data.edge_index)
        pred = out.argmax(dim=1)

    y_true = y[test_mask].cpu()
    y_pred = pred[test_mask].cpu()

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))