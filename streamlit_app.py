import json
import sys
import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# predict.py must be in the same directory as this script
sys.path.insert(0, str(Path(__file__).parent))
from predict import (
    load_artifacts,
    predict_single_ticket,
    predict_dataframe,
    validate_dossiers,
    MISMATCH_HIDDEN_CRISIS,
    MISMATCH_FALSE_ALARM,
    MISMATCH_CONSISTENT,
)

# ─── Page config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SIA — Support Integrity Auditor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Minimal custom CSS ─────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Tighten default padding */
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    /* Verdict badges */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    .badge-crisis  { background: #fee2e2; color: #991b1b; }
    .badge-alarm   { background: #fef9c3; color: #92400e; }
    .badge-ok      { background: #dcfce7; color: #166534; }

    /* Evidence table rows */
    .ev-row { font-size: 0.85rem; border-bottom: 1px solid #e5e7eb; padding: 4px 0; }
    .ev-source { color: #6b7280; font-family: monospace; }
    .ev-impact  { color: #374151; }

    /* Section headings inside expanders */
    .subhead { font-weight: 600; color: #374151; margin-bottom: 4px; }
</style>
""", unsafe_allow_html=True)

# ─── Cached artifact loader ──────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model artifacts…")
def get_artifacts():
    try:
        clf, tfidf = load_artifacts()
        return clf, tfidf, None
    except FileNotFoundError as exc:
        return None, None, str(exc)

clf, tfidf, artifact_error = get_artifacts()

# ─── Helper: render a dossier result ────────────────────────────────────────

def mismatch_badge(mtype):
    if mtype == MISMATCH_HIDDEN_CRISIS:
        return '<span class="badge badge-crisis">🔴 Hidden Crisis</span>'
    if mtype == MISMATCH_FALSE_ALARM:
        return '<span class="badge badge-alarm">🟡 False Alarm</span>'
    return '<span class="badge badge-ok">✅ Consistent</span>'


def render_dossier(dossier):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Assigned Priority", dossier["assigned_priority"])
    col2.metric("Inferred Severity", dossier["inferred_severity"])
    col3.metric("Severity Δ", dossier["severity_delta"])
    col4.metric("Confidence", f"{dossier['confidence']:.0%}")

    st.markdown(f"**Verdict:** {mismatch_badge(dossier['mismatch_type'])}", unsafe_allow_html=True)
    st.caption(dossier.get("constraint_analysis", ""))

    st.markdown("---")

    with st.expander("📋 Evidence", expanded=True):
        evidence = dossier.get("feature_evidence", [])
        if not evidence:
            st.write("No evidence items generated.")
        else:
            for item in evidence:
                cols = st.columns([2, 3, 5])
                cols[0].markdown(f'<span class="ev-source">{item["source"]}</span>', unsafe_allow_html=True)
                cols[1].write(str(item["value"]))
                cols[2].markdown(f'<span class="ev-impact">{item["impact"].replace("_", " ")}</span>', unsafe_allow_html=True)

    with st.expander("🗂 Full Dossier JSON"):
        st.json(dossier)


# ─── Tab 1 — Single Ticket Audit ────────────────────────────────────────────

def tab_single():
    st.header("Single Ticket Audit")
    st.write("Enter a ticket below to check whether its assigned priority matches the inferred severity.")

    if artifact_error:
        st.error(f"Could not load model artifacts: {artifact_error}")
        st.stop()

    with st.form("single_ticket_form"):
        c1, c2 = st.columns(2)
        ticket_id       = c1.text_input("Ticket ID",       value="TKT-001", placeholder="TKT-001")
        priority_level  = c2.selectbox("Assigned Priority", ["Low", "Medium", "High", "Critical"])
        ticket_subject  = st.text_input("Ticket Subject",  placeholder="e.g. account hacked")
        ticket_desc     = st.text_area("Ticket Description", height=120,
                                       placeholder="Describe the issue…")
        c3, c4 = st.columns(2)
        ticket_channel  = c3.selectbox("Ticket Channel", ["Email", "Phone", "Chat", "Social Media", "Other"])
        customer_email  = c4.text_input("Customer Email (optional)",
                                        placeholder="user@example.com")

        submitted = st.form_submit_button("Run Audit", type="primary", use_container_width=True)

    if submitted:
        if not ticket_subject.strip() or not ticket_desc.strip():
            st.warning("Ticket Subject and Description are required.")
            return

        ticket = {
            "Ticket_ID":          ticket_id.strip() or "UNKNOWN",
            "Ticket_Subject":     ticket_subject.strip(),
            "Ticket_Description": ticket_desc.strip(),
            "Priority_Level":     priority_level,
            "Ticket_Channel":     ticket_channel,
            "Customer_Email":     customer_email.strip(),
        }

        with st.spinner("Running inference…"):
            try:
                dossier = predict_single_ticket(ticket, clf, tfidf)
            except ValueError as exc:
                st.error(f"Validation error: {exc}")
                return
            except Exception as exc:
                st.error(f"Inference failed: {exc}")
                return

        st.success("Audit complete.")
        render_dossier(dossier)



# ─── Dashboard: severity delta heatmap ───────────────────────────────────────

def render_heatmap(df_heatmap):
    st.markdown("#### Severity Delta Heatmap")
    st.caption("Mean severity delta (inferred − assigned) by channel and assigned priority.")

    # Guard: column missing entirely or all values are blank
    has_channel = (
        "Ticket_Channel" in df_heatmap.columns
        and df_heatmap["Ticket_Channel"].replace("", float("nan")).notna().any()
    )
    if not has_channel:
        st.warning(
            "Ticket_Channel is not available in the uploaded CSV — "
            "add a Ticket_Channel column to enable this heatmap."
        )
        return

    # Drop rows where channel is blank (treat as unknown)
    df = df_heatmap[df_heatmap["Ticket_Channel"].str.strip() != ""].copy()
    if df.empty:
        st.warning("All Ticket_Channel values are blank — cannot build heatmap.")
        return

    priority_order = ["Low", "Medium", "High", "Critical"]

    pivot = (
        df.groupby(["Ticket_Channel", "assigned_priority"])["severity_delta"]
        .mean()
        .round(2)
        .unstack(level="assigned_priority")
        .reindex(columns=priority_order)  # consistent column order
    )
    pivot.index.name = "Ticket Channel"
    pivot.columns.name = "Assigned Priority"

    # px is imported at module level
    fig_heat = px.imshow(
        pivot,
        text_auto=True,
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        zmin=-3,
        zmax=3,
        labels={"color": "Mean Δ", "x": "Assigned Priority", "y": "Ticket Channel"},
        title="Mean Severity Delta by Channel × Priority",
        aspect="auto",
    )
    fig_heat.update_layout(
        margin=dict(t=48, b=16, l=16, r=16),
        title_font_size=14,
        coloraxis_colorbar=dict(title="Mean Δ", tickvals=[-3, -2, -1, 0, 1, 2, 3]),
        xaxis=dict(side="bottom"),
    )
    fig_heat.update_traces(
        hovertemplate=(
            "Channel: %{y}<br>"
            "Priority: %{x}<br>"
            "Mean Δ: %{z:.2f}<extra></extra>"
        )
    )
    st.plotly_chart(fig_heat, use_container_width=True)


# ─── Dashboard: mismatch distribution + top signals ─────────────────────────

def render_dashboard(all_dossiers, df_heatmap):
    st.markdown("### Priority Mismatch Dashboard")

    ch1, ch2 = st.columns(2)

    # ── Chart 1: Mismatch type distribution (donut) ───────────────────────────
    type_counts = {}
    for d in all_dossiers:
        label = d["mismatch_type"]
        type_counts[label] = type_counts.get(label, 0) + 1

    df_dist = pd.DataFrame(
        [{"Mismatch Type": k, "Count": v} for k, v in type_counts.items()]
    ).sort_values("Count", ascending=False)

    color_map = {
        MISMATCH_HIDDEN_CRISIS: "#ef4444",
        MISMATCH_FALSE_ALARM:   "#f59e0b",
        MISMATCH_CONSISTENT:    "#22c55e",
    }

    fig_dist = px.pie(
        df_dist,
        names="Mismatch Type",
        values="Count",
        hole=0.52,
        color="Mismatch Type",
        color_discrete_map=color_map,
        title="Mismatch Type Distribution",
    )
    fig_dist.update_traces(
        textposition="outside",
        textinfo="percent+label",
        hovertemplate="%{label}: %{value} tickets (%{percent})<extra></extra>",
    )
    fig_dist.update_layout(
        showlegend=False,
        margin=dict(t=48, b=16, l=16, r=16),
        title_font_size=14,
    )
    ch1.plotly_chart(fig_dist, use_container_width=True)

    # ── Chart 2: Top contributing signals (horizontal bar) ───────────────────
    # Count how many times each `impact` value appears across all evidence lists.
    impact_counts = {}
    for d in all_dossiers:
        for ev in d.get("feature_evidence", []):
            raw = ev.get("impact", "")
            if not raw:
                continue
            # Skip bookkeeping impacts that don't reflect a signal decision
            if raw.startswith("weighted_fusion") or raw.startswith("clf_predicted"):
                continue
            label = raw.replace("_", " ").title()
            impact_counts[label] = impact_counts.get(label, 0) + 1

    if impact_counts:
        df_sig = (
            pd.DataFrame(
                [{"Signal": k, "Occurrences": v} for k, v in impact_counts.items()]
            )
            .sort_values("Occurrences", ascending=True)
            .tail(12)
        )

        fig_sig = px.bar(
            df_sig,
            x="Occurrences",
            y="Signal",
            orientation="h",
            title="Top Contributing Signals",
            color="Occurrences",
            color_continuous_scale=[
                [0.0,  "#bfdbfe"],
                [0.5,  "#3b82f6"],
                [1.0,  "#1d4ed8"],
            ],
        )
        fig_sig.update_layout(
            coloraxis_showscale=False,
            margin=dict(t=48, b=16, l=16, r=16),
            title_font_size=14,
            yaxis_title=None,
            xaxis_title="Occurrences across all dossiers",
        )
        fig_sig.update_traces(
            hovertemplate="%{y}: %{x} occurrences<extra></extra>",
        )
        ch2.plotly_chart(fig_sig, use_container_width=True)
    else:
        ch2.info("No evidence signals to display.")

    render_heatmap(df_heatmap)

    st.markdown("---")

# ─── Tab 2 — Batch Upload ────────────────────────────────────────────────────

def tab_batch():
    st.header("Batch CSV Upload")
    st.write(
        "Upload a CSV file containing ticket rows. "
        "Required columns: `Ticket_ID`, `Ticket_Subject`, `Ticket_Description`, `Priority_Level`. "
        "Optional: `Customer_Email`, `Ticket_Channel`."
    )

    if artifact_error:
        st.error(f"Could not load model artifacts: {artifact_error}")
        st.stop()

    uploaded = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded is None:
        st.info("No file uploaded yet.")
        return

    try:
        df_input = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not parse CSV: {exc}")
        return

    required_cols = {"Ticket_ID", "Ticket_Subject", "Ticket_Description", "Priority_Level"}
    missing_cols = required_cols - set(df_input.columns)
    if missing_cols:
        st.error(f"CSV is missing required columns: {', '.join(sorted(missing_cols))}")
        return

    st.write(f"Loaded **{len(df_input)}** rows. Preview:")
    st.dataframe(df_input.head(5), use_container_width=True)

    if st.button("Run Batch Inference", type="primary"):
        progress = st.progress(0, text="Processing tickets…")

        all_dossiers = []
        heatmap_rows = []   # (ticket_channel, assigned_priority, severity_delta)
        errors = []
        for i, (_, row) in enumerate(df_input.iterrows()):
            try:
                d = predict_single_ticket(row.to_dict(), clf, tfidf)
                all_dossiers.append(d)
                heatmap_rows.append({
                    "Ticket_Channel":    str(row.get("Ticket_Channel", "")).strip(),
                    "assigned_priority": d["assigned_priority"],
                    "severity_delta":    d["severity_delta"],
                })
            except Exception as exc:
                errors.append({"ticket": row.get("Ticket_ID", i), "error": str(exc)})
            progress.progress((i + 1) / len(df_input),
                              text=f"Processing {i+1} / {len(df_input)}")

        df_heatmap = pd.DataFrame(heatmap_rows)

        progress.empty()

        if errors:
            st.warning(f"{len(errors)} ticket(s) failed inference and were skipped.")
            with st.expander("Show errors"):
                st.json(errors)

        if not all_dossiers:
            st.error("No tickets were successfully processed.")
            return

        # ── Summary metrics ──────────────────────────────────────────────
        total      = len(all_dossiers)
        mismatches = [d for d in all_dossiers if d["mismatch_type"] != MISMATCH_CONSISTENT]
        hidden     = [d for d in all_dossiers if d["mismatch_type"] == MISMATCH_HIDDEN_CRISIS]
        false_alrm = [d for d in all_dossiers if d["mismatch_type"] == MISMATCH_FALSE_ALARM]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Tickets",   total)
        m2.metric("Mismatches",      len(mismatches))
        m3.metric("Hidden Crises",   len(hidden))
        m4.metric("False Alarms",    len(false_alrm))

        # ── Dashboard ─────────────────────────────────────────────────────
        render_dashboard(all_dossiers, df_heatmap)

        # ── Results table ─────────────────────────────────────────────────
        summary_rows = [
            {
                "Ticket ID":         d["ticket_id"],
                "Assigned Priority": d["assigned_priority"],
                "Inferred Severity": d["inferred_severity"],
                "Mismatch Type":     d["mismatch_type"],
                "Confidence":        round(d["confidence"], 3),
                "Severity Δ":        d["severity_delta"],
            }
            for d in all_dossiers
        ]
        df_results = pd.DataFrame(summary_rows)

        st.markdown("### Results")
        st.dataframe(df_results, use_container_width=True)

        # ── Downloads ─────────────────────────────────────────────────────
        st.markdown("### Download")
        dc1, dc2 = st.columns(2)

        csv_bytes = df_results.to_csv(index=False).encode("utf-8")
        dc1.download_button(
            "⬇ predictions.csv",
            data=csv_bytes,
            file_name="predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )

        json_bytes = json.dumps(all_dossiers, indent=2).encode("utf-8")
        dc2.download_button(
            "⬇ dossiers.json",
            data=json_bytes,
            file_name="dossiers.json",
            mime="application/json",
            use_container_width=True,
        )

        # ── Dossier validation report ─────────────────────────────────────
        with st.expander("Dossier Validation Report"):
            val_results = validate_dossiers(all_dossiers)
            col_a, col_b = st.columns(2)
            col_a.metric("Total Dossiers",          val_results["total_dossiers"])
            col_a.metric("Invalid Dossiers",         val_results["invalid_dossier_count"])
            col_b.metric("Missing Field Count",      val_results["missing_field_count"])
            col_b.metric("Empty Evidence Count",     val_results["empty_evidence_count"])


# ─── Tab 3 — Project Overview ────────────────────────────────────────────────

def tab_overview():
    st.header("Project Overview")

    # Check for architecture image
    arch_candidates = [
        Path("images/architecture.png"),
        Path("architecture.png"),
    ]
    for p in arch_candidates:
        if p.exists():
            st.image(str(p), caption="SIA End-to-End Architecture", use_container_width=True)
            break

    st.markdown("""
**Support Integrity Auditor (SIA)** detects inconsistencies between customer support ticket content
and the priority level assigned during triage. The system surfaces two classes of error:

- **Hidden Crisis** — a critical issue assigned a low priority (under-prioritized).
- **False Alarm** — a minor issue escalated beyond its actual severity (over-prioritized).

SIA does not rely on a manually labelled ground-truth dataset. Instead it builds a multi-signal
fusion score that infers what severity a ticket *should* be, then compares that against the assigned
`Priority_Level`. Flagged tickets are explained through structured evidence dossiers.
""")

    st.markdown("---")
    st.subheader("Pipeline Phases")

    phases = [
        (
            "Phase 1 — Data Preprocessing & Pseudo-Label Generation",
            """Loads and cleans the ticket dataset. Generates inferred severity labels via a
three-signal fusion score:
- **Template signal (60%)** — subject matched against 26 known templates.
- **Time signal (25%)** — resolution time ranked percentile-wise.
- **Rule signal (15%)** — keyword scan for security and outage terms.

Tickets where inferred severity differs from assigned priority are flagged as mismatches.
Output: `sia_pseudo_labeled_v3.csv`.""",
        ),
        (
            "Phase 2 — Classifier Training",
            """Trains a LightGBM (or fallback GradientBoosting) severity classifier on the
pseudo-labeled data. Features: 1,000-dimensional TF-IDF from the combined subject and
description, plus `Resolution_Time_Hours`. Split: 70 / 15 / 15 (train / val / test).
Outputs: `sia_classifier_v1.joblib`, `sia_tfidf_v1.joblib`.""",
        ),
        (
            "Phase 3 — Evidence Dossier Generation",
            """For every mismatch, produces a structured dossier with: ticket metadata,
severity delta, feature evidence items (ten extractors covering security keywords, outage
terms, billing patterns, account management, subject template, resolution time, rule signal,
fusion score, and classifier confidence), constraint analysis, and a three-component
confidence score.
Outputs: `dossier_output.csv`, `dossier_output.json`.""",
        ),
        (
            "Phase 4 — Inference Pipeline",
            """`predict_single_ticket()` and `predict_dataframe()` reuse all Phase 3 logic
without retraining. `Resolution_Time_Hours` defaults to 24.0 at inference time since it
is unavailable when a ticket is first opened.
Outputs: `predictions.csv`, `dossiers.json`.""",
        ),
    ]

    for title, body in phases:
        with st.expander(title):
            st.markdown(body)

    st.markdown("---")
    st.subheader("Key Design Decisions")

    decisions = {
        "Why pseudo-labels?": (
            "The source dataset is synthetically generated with fixed subject templates, "
            "making template-matching a reliable labeling strategy without human annotators."
        ),
        "Why is Issue_Category excluded?": (
            "Cross-tabulation showed near-perfect correlation with Priority_Level in the "
            "synthetic data — including it would constitute label leakage."
        ),
        "Why TF-IDF over embeddings?": (
            "The dataset uses short, fixed subject phrases. TF-IDF with 1,000 features is "
            "sufficient and produces an interpretable feature importance chart."
        ),
        "Resolution time in both label and features?": (
            "Deliberate: it contributes to the pseudo-label fusion (training signal) and "
            "is also passed as a raw feature. At inference time it defaults to 24.0, "
            "which has minimal impact given the feature weight structure."
        ),
    }

    for question, answer in decisions.items():
        st.markdown(f"**{question}**")
        st.write(answer)

    st.markdown("---")
    st.subheader("Artifact Reference")

    artifact_table = pd.DataFrame([
        ("sia_pseudo_labeled_v3.csv",  "Cleaned tickets with inferred severity and mismatch flag"),
        ("sia_classifier_v1.joblib",   "Trained LightGBM (or GB) severity classifier"),
        ("sia_tfidf_v1.joblib",        "Fitted TF-IDF vectorizer (max_features=1000)"),
        ("dossier_output.csv",         "Flat CSV of all training-phase mismatch dossiers"),
        ("dossier_output.json",        "Nested JSON dossiers with evidence lists (training phase)"),
        ("predictions.csv",            "Batch inference summary — 5-column flat file"),
        ("dossiers.json",              "Batch inference full dossiers"),
        ("confusion_matrix.png",       "Test-set confusion matrix (4-class)"),
        ("feature_importance.png",     "Top 10 features by gain"),
    ], columns=["File", "Description"])

    st.dataframe(artifact_table, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Known Limitations")

    limitations = [
        "Dataset is synthetic and template-driven; real support corpora are noisier.",
        "Resolution time is unavailable at ticket-open time — inference uses a fixed default (24.0 h).",
        "Severity inference relies on handcrafted templates and rules; quality depends on those mappings.",
        "Issue_Category was excluded due to label leakage; this is documented in the leakage audit.",
    ]
    for item in limitations:
        st.markdown(f"- {item}")


# ─── Main layout ─────────────────────────────────────────────────────────────

st.title("🔍 Support Integrity Auditor")
st.caption("MARS Open Project 2026 — Mismatch detection for customer support ticket priority levels")

tab1, tab2, tab3 = st.tabs(["Single Ticket Audit", "Batch Upload", "Project Overview"])

with tab1:
    tab_single()

with tab2:
    tab_batch()

with tab3:
    tab_overview()
