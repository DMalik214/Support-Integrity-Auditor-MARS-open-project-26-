import re
import os
import json
import math
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DATA_FILE = "customer_support_tickets.csv"
ALT_FILE = "enhanced_customer_support_data.csv"
OUTPUT_FILE = "sia_pseudo_labeled_v3.csv"
MODEL_PATH = "sia_classifier_v1.joblib"
VECTORIZER_PATH = "sia_tfidf_v1.joblib"


# --- Constants and Mappings ---

SEVERITY_MAP = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
SEVERITY_REV = {v: k for k, v in SEVERITY_MAP.items()}

subject_map = {
    "critical": [
        "stolen card", "unrecognized login", "account hacked", "phishing attempt",
        "suspicious activity", "alert notification", "suspicious charge", "account compromised",
    ],
    "high": [
        "api error 500", "screen freezes", "installation issue", "data not syncing",
        "app crashing", "login failed", "2fa issues", "application crashes",
    ],
    "medium": [
        "refund status", "update credit card", "charged twice", "invoice discrepancy",
        "payment failed", "change email", "password reset", "subscription upgrade",
        "profile update", "delete account", "refund",
    ],
    "low": [
        "product question", "hours of operation", "demo request", "feature request",
        "office location", "pricing tiers", "pricing question",
    ],
}

SECURITY_KWS = ["compromised", "hacked", "stolen", "suspicious", "fraud"]
OUTAGE_KWS = ["crash", "error", "spinning wheel", "not loading", "failing"]

MISMATCH_HIDDEN_CRISIS = "Hidden Crisis (Under-prioritized)"
MISMATCH_FALSE_ALARM = "False Alarm (Over-prioritized)"
MISMATCH_CONSISTENT = "Consistent"


# --- Preprocessing ---

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_subject(raw):
    if not isinstance(raw, str):
        return ""
    return raw.split(" - ")[0].strip()

def clean_description(raw):
    if not isinstance(raw, str):
        return ""
    text = re.sub(r"^Hi Support,\s*", "", raw).strip()
    first_sentence = re.split(r"(?<=[.!?])\s+", text)[0].strip()
    return first_sentence.rstrip(".").strip()


# --- Template Severity Lookup ---

def build_lookup(mapping):
    lookup = {}
    for sev, items in mapping.items():
        for item in items:
            lookup[normalize_text(item)] = SEVERITY_MAP[sev.capitalize()]
    return lookup

SUBJECT_LOOKUP = build_lookup(subject_map)

def get_template_severity(text):
    norm = normalize_text(text)
    return SUBJECT_LOOKUP.get(norm, 1)


# --- Feature Engineering ---

def keyword_rule_signal(text):
    norm = normalize_text(text)
    if any(k in norm for k in SECURITY_KWS):
        return 3
    if any(k in norm for k in OUTAGE_KWS):
        return 2
    return 1


# --- Data Loading ---

def load_data():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    print(f"Dataset loaded: {df.shape}")
    return df


# --- Full Preprocessing Pipeline ---

def preprocess(df):
    df["ticket_subject_clean"] = df["Ticket_Subject"].apply(clean_subject)
    df["ticket_desc_clean"] = df["Ticket_Description"].apply(clean_description)
    df["text_combined"] = (df["ticket_subject_clean"] + " " + df["ticket_desc_clean"]).str.strip()
    return df


# --- Template Severity ---

def apply_template_severity(df):
    df["template_severity_num"] = df["ticket_subject_clean"].apply(get_template_severity)
    df["template_severity"] = df["template_severity_num"].map(SEVERITY_REV)
    return df


# --- Multi-Signal Fusion (Pseudo-Label Generation) ---

def apply_fusion(df):
    df["time_rank"] = df["Resolution_Time_Hours"].rank(pct=True)
    df["signal_time_num"] = pd.cut(
        df["time_rank"],
        bins=[0, 0.2, 0.5, 0.8, 1.0],
        labels=[3, 2, 1, 0],
        include_lowest=True
    ).astype(int)

    df["signal_rule_num"] = df["text_combined"].apply(keyword_rule_signal)

    df["fusion_score"] = (
        0.60 * df["template_severity_num"] +
        0.25 * df["signal_time_num"] +
        0.15 * df["signal_rule_num"]
    )

    df["inferred_severity_num"] = df["fusion_score"].round().astype(int).clip(0, 3)
    df["inferred_severity"] = df["inferred_severity_num"].map(SEVERITY_REV)
    return df


# --- Integrity Audit (Validation) ---

def apply_integrity_audit(df):
    df["priority_num"] = df["Priority_Level"].map(SEVERITY_MAP)
    df["is_mismatch"] = (df["inferred_severity_num"] != df["priority_num"]).astype(int)

    mismatch_rate = df["is_mismatch"].mean()
    kappa = cohen_kappa_score(df["inferred_severity_num"], df["priority_num"])

    print(f"Integrity Audit Results:")
    print(f"- Mismatch Rate: {mismatch_rate:.2%}")
    print(f"- Cohen's Kappa (Agreement): {kappa:.4f}")

    df["delta"] = df["inferred_severity_num"] - df["priority_num"]
    df["mismatch_type"] = MISMATCH_CONSISTENT
    df.loc[df["delta"] > 0, "mismatch_type"] = MISMATCH_HIDDEN_CRISIS
    df.loc[df["delta"] < 0, "mismatch_type"] = MISMATCH_FALSE_ALARM
    return df


# --- Export Pseudo-Labeled Dataset ---

def export_pseudo_labeled(df):
    export_cols = [
        "Ticket_ID", "ticket_subject_clean", "ticket_desc_clean",
        "Priority_Level", "inferred_severity", "is_mismatch", "mismatch_type"
    ]
    df[export_cols].to_csv(OUTPUT_FILE, index=False)
    print(f"Final project-ready dataset exported to: {OUTPUT_FILE}")
    print(f"Processed {len(df)} tickets.")
    print(f"Identified {df['is_mismatch'].sum()} potential integrity issues.")


# --- Classifier Training ---

def train_classifier(df):
    X_text = df["text_combined"]
    X_num = df[["Resolution_Time_Hours"]]
    y = df["inferred_severity_num"]

    X_train_text, X_temp_text, X_train_num, X_temp_num, y_train, y_temp = train_test_split(
        X_text, X_num, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val_text, X_test_text, X_val_num, X_test_num, y_val, y_test = train_test_split(
        X_temp_text, X_temp_num, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
    X_train_tfidf = tfidf.fit_transform(X_train_text).toarray()
    X_val_tfidf = tfidf.transform(X_val_text).toarray()
    X_test_tfidf = tfidf.transform(X_test_text).toarray()

    X_train_final = np.hstack([X_train_tfidf, X_train_num.values])
    X_val_final = np.hstack([X_val_tfidf, X_val_num.values])
    X_test_final = np.hstack([X_test_tfidf, X_test_num.values])

    if HAS_LGBM:
        print("Training LightGBM classifier...")
        clf = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
            class_weight='balanced',
            importance_type='gain',
            verbose=-1
        )
        clf.fit(X_train_final, y_train, eval_set=[(X_val_final, y_val)])
    else:
        print("LightGBM not available, falling back to GradientBoostingClassifier...")
        clf = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42,
        )
        clf.fit(X_train_final, y_train)

    y_pred = clf.predict(X_test_final)
    accuracy = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)

    print(f"\n--- Classifier Performance (Test Set) ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High', 'Critical']))

    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Low', 'Medium', 'High', 'Critical'],
                yticklabels=['Low', 'Medium', 'High', 'Critical'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix: Inferred Severity vs Predicted')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()

    feature_names = tfidf.get_feature_names_out().tolist() + ["Resolution_Time_Hours"]
    importances = clf.feature_importances_
    indices = np.argsort(importances)[-10:]

    plt.figure(figsize=(10, 6))
    plt.title('Top 10 Feature Importances (Gain)')
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.close()

    return clf, tfidf


# --- Save Artifacts ---

def save_artifacts(clf, tfidf):
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(tfidf, VECTORIZER_PATH)
    print(f"\nModel exported to: {MODEL_PATH}")
    print(f"Vectorizer exported to: {VECTORIZER_PATH}")


# --- Entry Point ---

if __name__ == "__main__":
    df = load_data()
    df = preprocess(df)
    df = apply_template_severity(df)
    df = apply_fusion(df)
    df = apply_integrity_audit(df)
    export_pseudo_labeled(df)
    clf, tfidf = train_classifier(df)
    save_artifacts(clf, tfidf)
    print("\nTraining pipeline complete.")
