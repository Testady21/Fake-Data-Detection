import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.metrics import classification_report
from main import early_fusion_model, build_loaders, valid_epoch

# === Define absolute base paths ===
PROJECT_ROOT = r"C:\Users\tvijo\Desktop\coding\LLM\SOP\fakenews"
RESULTS_DIR = os.path.join(PROJECT_ROOT, r"MLP-CLIP\MediaEval_VN_data\results\Im+Ev\early_fusion\10-23-2025_23;34;53")
DATA_DIR = os.path.join(PROJECT_ROOT, r"MLP-CLIP\data\2015_data\EvRep")

# === Load config ===
cfg_path = os.path.join(RESULTS_DIR, "CFG.json")
print(f"Loading config from: {cfg_path}")
if not os.path.exists(cfg_path):
    raise FileNotFoundError(f"❌ CFG.json not found at: {cfg_path}")
CFG = pd.read_json(cfg_path)

# === Setup device & loss ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lossfunction = nn.CrossEntropyLoss()

# === Helper for saving reports ===
def report_to_json(reports, outdir):
    import json
    reports['avg precision'] = reports['weighted avg']['precision']
    reports['avg f1'] = reports['weighted avg']['f1-score']
    reports['avg recall'] = reports['weighted avg']['recall']
    reports['macro-precision'] = reports['macro avg']['precision']
    reports['macro-f1'] = reports['macro avg']['f1-score']
    reports['macro-recall'] = reports['macro avg']['recall']
    del reports['macro avg']
    del reports['weighted avg']
    out_path = os.path.join(outdir, "Creports_100samples_test.json")
    print(f"Saving classification report to: {out_path}")
    with open(out_path, 'w') as fp:
        json.dump(reports, fp, indent=4)

# === Load model ===
dropout = CFG.get("dropout", [CFG.loc[0, "dropout"]])[0]
model = early_fusion_model(dropout)
model_path = os.path.join(RESULTS_DIR, "best.pt")
print(f"Loading model weights from: {model_path}")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model weights not found at: {model_path}")

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# === Load test data ===
features_path = os.path.join(DATA_DIR, "100samples_test_features.npy")
df_path = os.path.join(DATA_DIR, "100samples_test.csv")
print(f"Loading test features: {features_path}")
print(f"Loading test CSV: {df_path}")

if not os.path.exists(features_path):
    raise FileNotFoundError(f"❌ Missing file: {features_path}")
if not os.path.exists(df_path):
    raise FileNotFoundError(f"❌ Missing file: {df_path}")

test_clip_features = np.load(features_path)
test_df = pd.read_csv(df_path)
 

# === Load labels robustly ===
if "label" not in test_df.columns:
    raise KeyError("❌ The test CSV must contain a 'label' column.")

first_val = test_df["label"].iloc[0]

if isinstance(first_val, str):
    # String-based labels ("real"/"fake")
    test_labels = test_df["label"].apply(lambda x: 1 if x.lower() == "real" else 0).values
else:
    # Already numeric (0/1)
    test_labels = test_df["label"].astype(int).values


# === Create args stub ===
class Args:
    device = device
    batch_size = 64

args = Args()

# === Build DataLoader ===
test_loader = build_loaders(test_clip_features, test_labels, args, mode="test")

# === Run inference ===
with torch.no_grad():
    test_loss, test_acc, test_predictions, test_labels_pred, test_targets = valid_epoch(
        model, test_loader, lossfunction, args
    )

# === Generate classification report ===
reports = classification_report(
    torch.tensor(test_targets, device="cpu"),
    torch.tensor(test_labels_pred, device="cpu"),
    target_names=["fake", "real"],
    output_dict=True
)
report_to_json(reports, RESULTS_DIR)

# === Save output labels ===
test_predictions = test_predictions.cpu().detach().numpy()
out_df = pd.DataFrame({
    "id": np.arange(len(test_targets)),
    "target": test_targets,
    "predicted": test_labels_pred,
    "prob_fake": test_predictions[:, 0],
    "prob_real": test_predictions[:, 1]
})

out_csv = os.path.join(RESULTS_DIR, "labels_100samples_test.csv")
out_df.to_csv(out_csv, index=False)
print(f"✅ Inference complete. Results saved at: {out_csv}")
