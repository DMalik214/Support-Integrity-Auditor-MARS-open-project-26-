import re
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")

MODEL_PATH = "sia_classifier_v1.joblib"
VECTORIZER_PATH = "sia_tfidf_v1.joblib"


# --- Constants (mirrors notebook exactly) ---

P3_SEVERITY_MAP = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
P3_SEVERITY_REV = {v: k for k, v in P3_SEVERITY_MAP.items()}

P3_SUBJECT_MAP_RAW = {
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

P3_SECURITY_KWS = [
    "compromised", "hacked", "stolen", "suspicious", "fraud",
    "phishing", "unauthorized", "breach", "account compromised",
    "unrecognized login", "suspicious activity", "suspicious charge",
]
P3_OUTAGE_KWS = [
    "crash", "error", "spinning wheel", "not loading", "failing",
    "500", "api error", "screen freezes", "data not syncing",
    "application crashes", "app crashing", "installation issue",
]
P3_BILLING_KWS = [
    "refund", "charged twice", "invoice", "payment", "billing",
    "credit card", "subscription", "upgrade", "pricing",
]
P3_ACCOUNT_KWS = [
    "password reset", "change email", "profile update", "delete account",
    "login failed", "2fa issues",
]
P3_INQUIRY_KWS = [
    "product question", "hours of operation", "demo request",
    "feature request", "office location", "pricing question", "pricing tiers",
]

MISMATCH_HIDDEN_CRISIS = "Hidden Crisis (Under-prioritized)"
MISMATCH_FALSE_ALARM = "False Alarm (Over-prioritized)"
MISMATCH_CONSISTENT = "Consistent"

REQUIRED_FIELDS = [
    "ticket_id", "assigned_priority", "inferred_severity",
    "mismatch_type", "severity_delta", "feature_evidence",
    "constraint_analysis", "confidence",
]


# --- Text Normalization ---

def _p3_normalize(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# --- Build Subject Lookup ---

def _build_subject_lookup():
    lookup = {}
    for sev, items in P3_SUBJECT_MAP_RAW.items():
        for item in items:
            lookup[_p3_normalize(item)] = P3_SEVERITY_MAP[sev.capitalize()]
    return lookup

P3_SUBJECT_LOOKUP = _build_subject_lookup()


# --- Artifact Loading ---

def load_artifacts():
    if not Path(MODEL_PATH).exists() or not Path(VECTORIZER_PATH).exists():
        raise FileNotFoundError(
            f"Model artifacts not found. Run train_pipeline.py first.\n"
            f"  Expected: {MODEL_PATH}, {VECTORIZER_PATH}"
        )
    clf = joblib.load(MODEL_PATH)
    tfidf = joblib.load(VECTORIZER_PATH)
    return clf, tfidf


# --- Preprocessing ---

def extract_customer_tier(email):
    if not isinstance(email, str):
        return "Standard"
    email = email.lower()
    if any(domain in email for domain in ['@corp.', '@enterprise.', '@global.', '@premium.']):
        return "Enterprise"
    return "Standard"

def validate_ticket_input(ticket):
    required = ["Ticket_ID", "Ticket_Subject", "Ticket_Description", "Priority_Level"]
    for col in required:
        if col not in ticket or (isinstance(ticket[col], str) and not ticket[col].strip()):
            raise ValueError(f"Missing or empty required field: {col}")
    valid_priorities = ["Low", "Medium", "High", "Critical"]
    if ticket["Priority_Level"] not in valid_priorities:
        raise ValueError(
            f"Invalid Priority_Level: {ticket['Priority_Level']}. Must be one of {valid_priorities}"
        )


# --- Evidence Extraction ---

def _match_keywords(text, keyword_list):
    norm = _p3_normalize(text)
    return [kw for kw in keyword_list if kw in norm]

def extract_subject_evidence(row):
    evidence = []
    subject_norm = _p3_normalize(row.get("ticket_subject_clean", ""))
    matched_sev_num = P3_SUBJECT_LOOKUP.get(subject_norm)
    if matched_sev_num is not None:
        evidence.append({
            "source": "ticket_subject_clean",
            "value": row["ticket_subject_clean"],
            "impact": f"template_match_severity_{P3_SEVERITY_REV[matched_sev_num].lower()}",
        })
    return evidence

def extract_template_mapping_evidence(row):
    evidence = []
    if pd.notna(row.get("template_severity")):
        evidence.append({
            "source": "template_mapping",
            "value": row["template_severity"],
            "impact": "template_severity",
        })
    return evidence

def extract_security_keyword_evidence(row):
    evidence = []
    text = str(row.get("text_combined", "")) + " " + str(row.get("ticket_desc_clean", ""))
    for kw in _match_keywords(text, P3_SECURITY_KWS):
        evidence.append({
            "source": "ticket_desc_clean",
            "value": kw,
            "impact": "critical_security_indicator",
        })
    return evidence

def extract_outage_keyword_evidence(row):
    evidence = []
    text = str(row.get("text_combined", "")) + " " + str(row.get("ticket_desc_clean", ""))
    for kw in _match_keywords(text, P3_OUTAGE_KWS):
        evidence.append({
            "source": "ticket_desc_clean",
            "value": kw,
            "impact": "service_outage_indicator",
        })
    return evidence

def extract_billing_keyword_evidence(row):
    evidence = []
    text = str(row.get("text_combined", "")) + " " + str(row.get("ticket_desc_clean", ""))
    for kw in _match_keywords(text, P3_BILLING_KWS):
        evidence.append({
            "source": "ticket_desc_clean",
            "value": kw,
            "impact": "billing_inquiry_indicator",
        })
    return evidence

def extract_account_keyword_evidence(row):
    evidence = []
    text = str(row.get("text_combined", "")) + " " + str(row.get("ticket_desc_clean", ""))
    for kw in _match_keywords(text, P3_ACCOUNT_KWS):
        evidence.append({
            "source": "ticket_desc_clean",
            "value": kw,
            "impact": "account_management_indicator",
        })
    return evidence

def extract_time_signal_evidence(row):
    evidence = []
    if pd.notna(row.get("Resolution_Time_Hours")):
        hours = float(row["Resolution_Time_Hours"])
        sig = int(row.get("signal_time_num", 1))
        label = {
            3: "fastest_20pct_resolution",
            2: "fast_resolution",
            1: "average_resolution",
            0: "slowest_20pct_resolution",
        }.get(sig, "unknown_time_signal")
        evidence.append({
            "source": "Resolution_Time_Hours",
            "value": round(hours, 2),
            "impact": label,
        })
    return evidence

def extract_rule_signal_evidence(row):
    evidence = []
    sig = int(row.get("signal_rule_num", 1))
    if sig == 3:
        evidence.append({
            "source": "signal_rule_num",
            "value": sig,
            "impact": "rule_signal_security_critical",
        })
    elif sig == 2:
        evidence.append({
            "source": "signal_rule_num",
            "value": sig,
            "impact": "rule_signal_outage_high",
        })
    return evidence

def extract_fusion_score_evidence(row):
    evidence = []
    if pd.notna(row.get("fusion_score")):
        evidence.append({
            "source": "fusion_score",
            "value": round(float(row["fusion_score"]), 4),
            "impact": "weighted_fusion_0.60_template_0.25_time_0.15_rule",
        })
    return evidence

def extract_classifier_confidence_evidence(row):
    evidence = []
    if pd.notna(row.get("clf_confidence")):
        evidence.append({
            "source": "classifier_probability",
            "value": round(float(row["clf_confidence"]), 4),
            "impact": f"clf_predicted_severity_{P3_SEVERITY_REV.get(int(row['clf_pred_severity']), 'unknown').lower()}",
        })
    return evidence

def collect_all_evidence(row):
    evidence = []
    evidence += extract_security_keyword_evidence(row)
    evidence += extract_outage_keyword_evidence(row)
    evidence += extract_billing_keyword_evidence(row)
    evidence += extract_account_keyword_evidence(row)
    evidence += extract_subject_evidence(row)
    evidence += extract_template_mapping_evidence(row)
    evidence += extract_time_signal_evidence(row)
    evidence += extract_rule_signal_evidence(row)
    evidence += extract_fusion_score_evidence(row)
    evidence += extract_classifier_confidence_evidence(row)

    seen = set()
    unique = []
    for item in evidence:
        key = (item["source"], str(item["value"]))
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


# --- Confidence Scoring ---

def compute_signal_agreement(row):
    inferred = int(row.get("inferred_severity_num", 1))
    signals = []

    tmpl = row.get("template_severity_num")
    if pd.notna(tmpl):
        signals.append(
            1.0 if int(tmpl) == inferred else (0.5 if abs(int(tmpl) - inferred) == 1 else 0.0)
        )

    rule_raw = row.get("signal_rule_num")
    if pd.notna(rule_raw):
        rule_scaled = max(0, int(rule_raw) - 1)
        signals.append(
            1.0 if rule_scaled == inferred else (0.5 if abs(rule_scaled - inferred) == 1 else 0.0)
        )

    time_sig = row.get("signal_time_num")
    if pd.notna(time_sig):
        signals.append(
            1.0 if int(time_sig) == inferred else (0.5 if abs(int(time_sig) - inferred) == 1 else 0.0)
        )

    clf_pred = row.get("clf_pred_severity")
    if pd.notna(clf_pred):
        signals.append(
            1.0 if int(clf_pred) == inferred else (0.5 if abs(int(clf_pred) - inferred) == 1 else 0.0)
        )

    return float(np.mean(signals)) if signals else 0.5

def compute_dossier_confidence(row):
    clf_conf = float(row.get("clf_confidence", 0.5))
    sig_agree = compute_signal_agreement(row)
    delta = abs(int(row.get("delta", 0)))
    delta_w = min(delta / 3.0, 1.0)
    score = 0.50 * clf_conf + 0.30 * sig_agree + 0.20 * delta_w
    return round(float(np.clip(score, 0.0, 1.0)), 4)


# --- Constraint Analysis ---

def generate_constraint_analysis(row):
    assigned = str(row.get("Priority_Level", "Unknown"))
    inferred = str(row.get("inferred_severity", "Unknown"))
    m_type = str(row.get("mismatch_type", MISMATCH_CONSISTENT))
    delta = int(row.get("delta", 0))
    tmpl_sev = str(row.get("template_severity", "Unknown"))
    sig_rule = int(row.get("signal_rule_num", 1))
    time_hrs = row.get("Resolution_Time_Hours")
    clf_conf = float(row.get("clf_confidence", 0.0))

    text_combined = _p3_normalize(str(row.get("text_combined", "")))

    if sig_rule == 3:
        primary_evidence = "security-related keywords detected in ticket text"
    elif sig_rule == 2:
        primary_evidence = "service failure indicators detected in ticket text"
    elif any(kw in text_combined for kw in P3_BILLING_KWS):
        primary_evidence = "billing inquiry patterns present in ticket text"
    elif any(kw in text_combined for kw in P3_INQUIRY_KWS):
        primary_evidence = "general inquiry patterns present in ticket text"
    else:
        primary_evidence = f"template mapping resolved subject to {tmpl_sev}"

    if m_type == MISMATCH_HIDDEN_CRISIS:
        sentence1 = (
            f"Assigned priority is {assigned} while inferred severity is {inferred} "
            f"({primary_evidence})."
        )
    elif m_type == MISMATCH_FALSE_ALARM:
        sentence1 = (
            f"Assigned priority is {assigned} while inferred severity is {inferred} "
            f"({primary_evidence})."
        )
    else:
        return "Assigned priority matches inferred severity; no mismatch detected."

    parts = []
    if pd.notna(time_hrs):
        parts.append(f"resolution time {round(float(time_hrs), 1)} hours")
    parts.append(f"template severity {tmpl_sev}")
    parts.append(f"classifier confidence {clf_conf:.2f}")
    sentence2 = "Supporting signals: " + ", ".join(parts) + "."

    return f"{sentence1} {sentence2}"


# --- Dossier Assembly ---

def build_dossier(row):
    ticket_id = str(row.get("Ticket_ID", "UNKNOWN"))
    dossier = {
        "ticket_id": ticket_id,
        "assigned_priority": str(row.get("Priority_Level", "Unknown")),
        "inferred_severity": str(row.get("inferred_severity", "Unknown")),
        "mismatch_type": str(row.get("mismatch_type", MISMATCH_CONSISTENT)),
        "severity_delta": int(row.get("delta", 0)),
        "feature_evidence": collect_all_evidence(row),
        "constraint_analysis": str(row.get("constraint_analysis", "")),
        "confidence": float(row.get("dossier_confidence", 0.0)),
    }
    return dossier


# --- Dossier Validation ---

def validate_dossiers(all_dossiers):
    results = {
        "total_dossiers": len(all_dossiers),
        "missing_field_count": 0,
        "null_field_count": 0,
        "invalid_confidence_count": 0,
        "empty_evidence_count": 0,
        "delta_integrity_errors": 0,
        "mismatch_type_errors": 0,
        "invalid_dossier_count": 0,
    }
    invalid_ids = []

    for dos in all_dossiers:
        errors = []

        missing = [f for f in REQUIRED_FIELDS if f not in dos]
        if missing:
            errors.append(f"missing_fields:{missing}")
            results["missing_field_count"] += 1

        for fld in ["ticket_id", "assigned_priority", "inferred_severity", "mismatch_type"]:
            val = dos.get(fld, None)
            if not val or val in ("", "Unknown", "None", None):
                errors.append(f"null_field:{fld}")
                results["null_field_count"] += 1

        conf = dos.get("confidence", -1)
        if not (0.0 <= float(conf) <= 1.0):
            errors.append(f"confidence_out_of_range:{conf}")
            results["invalid_confidence_count"] += 1

        if len(dos.get("feature_evidence", [])) < 1:
            errors.append("empty_evidence_list")
            results["empty_evidence_count"] += 1

        assigned_num = P3_SEVERITY_MAP.get(dos.get("assigned_priority", ""), None)
        inferred_num = P3_SEVERITY_MAP.get(dos.get("inferred_severity", ""), None)
        reported_delta = dos.get("severity_delta", None)
        if assigned_num is not None and inferred_num is not None and reported_delta is not None:
            expected_delta = inferred_num - assigned_num
            if int(reported_delta) != expected_delta:
                errors.append(f"delta_mismatch:expected={expected_delta},got={reported_delta}")
                results["delta_integrity_errors"] += 1

        delta_val = dos.get("severity_delta", 0)
        m_type = dos.get("mismatch_type", "")
        if delta_val > 0 and m_type != MISMATCH_HIDDEN_CRISIS:
            errors.append(f"type_inconsistency:delta>0 but type={m_type}")
            results["mismatch_type_errors"] += 1
        elif delta_val < 0 and m_type != MISMATCH_FALSE_ALARM:
            errors.append(f"type_inconsistency:delta<0 but type={m_type}")
            results["mismatch_type_errors"] += 1

        if errors:
            results["invalid_dossier_count"] += 1
            invalid_ids.append({"ticket_id": dos.get("ticket_id"), "errors": errors})

    print("=" * 60)
    print("DOSSIER VALIDATION REPORT")
    print("=" * 60)
    for k, v in results.items():
        status = "✓" if (v == 0 or k == "total_dossiers") else "✗"
        print(f"  {status}  {k:<35}: {v}")
    print()
    if results["invalid_dossier_count"] == 0:
        print("✓ ALL DOSSIERS PASSED VALIDATION.")
    else:
        print(f"✗ {results['invalid_dossier_count']} dossier(s) failed. Details:")
        for rec in invalid_ids[:10]:
            print(f"   Ticket {rec['ticket_id']}: {rec['errors']}")

    return results


# --- Inference ---

def predict_single_ticket(ticket_dict, clf, tfidf):
    validate_ticket_input(ticket_dict)

    row = pd.Series(ticket_dict)
    row['ticket_subject_clean'] = _p3_normalize(row['Ticket_Subject'])
    row['ticket_desc_clean'] = _p3_normalize(row['Ticket_Description'])
    row['text_combined'] = row['ticket_subject_clean'] + " " + row['ticket_desc_clean']
    row['customer_tier'] = extract_customer_tier(row.get('Customer_Email', ''))

    tmpl_sev_num = P3_SUBJECT_LOOKUP.get(row['ticket_subject_clean'], 1)
    row['template_severity_num'] = tmpl_sev_num
    row['template_severity'] = P3_SEVERITY_REV[tmpl_sev_num]

    rule_sig_num = 1
    if any(kw in row['text_combined'] for kw in P3_SECURITY_KWS):
        rule_sig_num = 3
    elif any(kw in row['text_combined'] for kw in P3_OUTAGE_KWS):
        rule_sig_num = 2
    row['signal_rule_num'] = rule_sig_num

    row['signal_time_num'] = 1
    row['Resolution_Time_Hours'] = 24.0

    rule_scaled = max(0, rule_sig_num - 1)
    fusion_score = (0.60 * tmpl_sev_num) + (0.25 * row['signal_time_num']) + (0.15 * rule_scaled)
    row['fusion_score'] = fusion_score

    inf_sev_num = int(round(fusion_score))
    inf_sev_num = int(np.clip(inf_sev_num, 0, 3))
    row['inferred_severity_num'] = inf_sev_num
    row['inferred_severity'] = P3_SEVERITY_REV[inf_sev_num]

    assigned_num = P3_SEVERITY_MAP[row['Priority_Level']]
    delta = int(inf_sev_num - assigned_num)
    row['delta'] = delta
    row['is_mismatch'] = 1 if delta != 0 else 0

    if delta > 0:
        row['mismatch_type'] = MISMATCH_HIDDEN_CRISIS
    elif delta < 0:
        row['mismatch_type'] = MISMATCH_FALSE_ALARM
    else:
        row['mismatch_type'] = MISMATCH_CONSISTENT

    X_text = tfidf.transform([row['text_combined']]).toarray()
    X_num = np.array([[row['Resolution_Time_Hours']]])
    X_full = np.hstack([X_text, X_num])

    proba = clf.predict_proba(X_full)[0]
    row['clf_pred_severity'] = clf.predict(X_full)[0]
    row['clf_confidence'] = float(np.max(proba))

    row['signal_agreement'] = compute_signal_agreement(row)
    row['dossier_confidence'] = compute_dossier_confidence(row)
    row['constraint_analysis'] = generate_constraint_analysis(row)

    return build_dossier(row)

def predict_dataframe(df_new, clf, tfidf):
    return [predict_single_ticket(row.to_dict(), clf, tfidf) for _, row in df_new.iterrows()]

def format_human_result(dossier):
    output = []
    output.append(f"Ticket ID: {dossier['ticket_id']}")
    output.append(f"Assigned Priority: {dossier['assigned_priority']}")
    output.append(f"Inferred Severity: {dossier['inferred_severity']}")
    output.append(f"Prediction: {dossier['mismatch_type']}")
    output.append(f"Confidence: {dossier['confidence']:.2f}")
    output.append("Evidence:")
    for e in dossier['feature_evidence']:
        if e['impact'] in ['critical_security_indicator', 'service_outage_indicator', 'template_severity']:
            output.append(f"  - {e['value']} ({e['impact'].replace('_', ' ')})")
    return "\n".join(output)


# --- Entry Point ---

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py input.csv")
        sys.exit(1)

    input_path = sys.argv[1]
    if not Path(input_path).exists():
        print(f"Error: input file not found: {input_path}")
        sys.exit(1)

    clf, tfidf = load_artifacts()

    df_input = pd.read_csv(input_path)
    print(f"Loaded {len(df_input)} tickets from {input_path}")

    all_dossiers = predict_dataframe(df_input, clf, tfidf)
    print(f"Processed {len(all_dossiers)} tickets.")

    pd.DataFrame([
        {k: d[k] for k in ['ticket_id', 'assigned_priority', 'inferred_severity', 'mismatch_type', 'confidence']}
        for d in all_dossiers
    ]).to_csv("predictions.csv", index=False)

    with open("dossiers.json", "w") as f:
        json.dump(all_dossiers, f, indent=2)

    validate_dossiers(all_dossiers)

    print("\n✓ Exported predictions.csv and dossiers.json.")
