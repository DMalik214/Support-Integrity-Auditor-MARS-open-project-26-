# SIA Adversarial Evaluation Report

**Project:** MARS Open Project 2026 — Support Integrity Auditor  
**Evaluation Date:** _______________  
**Evaluator:** _______________  
**Model Artifacts Used:** `sia_classifier_v1.joblib`, `sia_tfidf_v1.joblib`  
**Notebook Version:** SIA_Cleaned_Notebook.ipynb (finalized)

---

## Purpose

This evaluation measures how well SIA handles tickets that are specifically designed to challenge keyword-based and template-based systems. Each test case in `adversarial_test_cases.csv` represents a real-world failure mode: a ticket where surface-level signals (subject line, obvious keywords) are deceptive, and the correct severity inference requires reading the full description, detecting buried keywords, or resisting framing language.

The goal is not to show that SIA performs perfectly on adversarial cases — its architecture has known limitations. The goal is to document exactly where those limits are, and why.

---

## Evaluation Procedure

1. Run `adversarial_test_cases.csv` through the inference pipeline:
   ```bash
   python predict.py adversarial_test_cases.csv
   ```

2. Collect `predictions.csv` and `dossiers.json` from the output.

3. For each ticket, compare `inferred_severity` against the `Expected_Mismatch` column in the adversarial CSV and against `Priority_Level`.

4. Fill in the results table below.

5. Write observations for each adversarial category.

---

## Results Table

Fill in `Inferred_Severity`, `Mismatch_Type_Predicted`, and `Confidence` from `predictions.csv`. The `Expected_Mismatch` column comes from the adversarial CSV.

| Ticket_ID | Subject | Priority_Level | Expected_Mismatch | Inferred_Severity | Mismatch_Type_Predicted | Confidence | Correct? |
|-----------|---------|----------------|-------------------|-------------------|------------------------|------------|----------|
| ADV-001   | product question | Low | Hidden Crisis | ___ | ___ | ___ | ___ |
| ADV-002   | password reset | Medium | Hidden Crisis | ___ | ___ | ___ | ___ |
| ADV-003   | refund status | Medium | Hidden Crisis | ___ | ___ | ___ | ___ |
| ADV-004   | pricing question | Low | Hidden Crisis | ___ | ___ | ___ | ___ |
| ADV-005   | feature request | Low | Hidden Crisis | ___ | ___ | ___ | ___ |
| ADV-006   | suspicious activity | Critical | Consistent | ___ | ___ | ___ | ___ |
| ADV-007   | app crashing | Critical | False Alarm | ___ | ___ | ___ | ___ |
| ADV-008   | hours of operation | Critical | Hidden Crisis | ___ | ___ | ___ | ___ |
| ADV-009   | invoice discrepancy | Critical | False Alarm | ___ | ___ | ___ | ___ |
| ADV-010   | data not syncing | Critical | False Alarm | ___ | ___ | ___ | ___ |
| ADV-011   | demo request | Low | Consistent | ___ | ___ | ___ | ___ |
| ADV-012   | account hacked | Low | Hidden Crisis | ___ | ___ | ___ | ___ |
| ADV-013   | subscription upgrade | Medium | Hidden Crisis | ___ | ___ | ___ | ___ |
| ADV-014   | login failed | Critical | False Alarm | ___ | ___ | ___ | ___ |
| ADV-015   | profile update | Medium | Hidden Crisis | ___ | ___ | ___ | ___ |

**Summary:**
- Total cases: 15
- Correct predictions: ___  
- Incorrect predictions: ___  
- Overall adversarial accuracy: ___%

---

## Category Analysis

### Category A: Security Buried in Low-Signal Subjects (ADV-001, ADV-002, ADV-003, ADV-004, ADV-005)

These cases place security or outage-related content in the description while the subject line maps to a low-severity template. Because the template signal carries 60% of the fusion score weight, SIA will tend to anchor to the subject-derived severity unless the rule signal (15% weight) is strong enough to shift the rounded score.

**Prediction:** For ADV-001 through ADV-005, expect SIA to underestimate severity on most cases. The 60% template weight is heavy enough that even a rule signal of 3 (security keywords) at 15% weight is unlikely to move the fused score past a rounding threshold unless the template severity is already Medium (1).

**Results observed:**

> *[Fill in after running evaluation. Note which tickets were correctly escalated, which were missed, and what the fusion scores were. Reference dossiers.json for per-ticket evidence details.]*

**Analysis:**

> *[Explain why SIA caught or missed each case. For misses, identify whether the failure was in the template dominance, keyword coverage, or rounding. Use the actual `fusion_score` values from dossier output.]*

---

### Category B: Subject-Template Correct, Agent Assignment Wrong (ADV-006, ADV-007, ADV-012)

These cases test whether SIA correctly holds to its inferred severity even when a human agent has assigned something inconsistent. ADV-006 and ADV-012 test under-assignment on high-signal subjects. ADV-007 tests over-assignment.

**Prediction:** SIA should perform well here. The subject template is the dominant signal, and these subjects are in the lookup table with correct mappings (`suspicious activity` → Critical, `account hacked` → Critical, `app crashing` → High). If the agent assigned differently, SIA should correctly flag the mismatch.

**Results observed:**

> *[Fill in.]*

**Analysis:**

> *[If SIA was correct on these, confirm that the template signal was the driving force by citing the fusion_score and template_severity_num from the dossier evidence. If incorrect, investigate whether the description's downplaying language influenced the classifier prediction and shifted the composite confidence.]*

---

### Category C: Trivial Issues Over-Assigned (ADV-009, ADV-010, ADV-014)

These are False Alarm cases where a ticket describing a minor or already-resolved issue was assigned a high priority. SIA should infer a lower severity and flag the mismatch as `False Alarm (Over-prioritized)`.

**Prediction:** These should be detectable, particularly ADV-009 (invoice discrepancy, clearly low-stakes language) and ADV-014 (historical, self-resolved login failure). ADV-010 is trickier because `data not syncing` maps to High in the template, and it was assigned Critical — only a one-level gap, which may result in a correct flag or a near-miss depending on the fusion score rounding.

**Results observed:**

> *[Fill in.]*

**Analysis:**

> *[For each case, note the severity_delta value from the dossier. A delta of -1 vs -2 matters here — ADV-009 should have a large negative delta, ADV-014 a smaller one.]*

---

### Category D: Urgency Language on Low-Severity Content (ADV-008, ADV-011)

ADV-008 uses executive-account language (`CEO's account`, `taken over`) without triggering the exact keyword matches in `P3_SECURITY_KWS`. ADV-011 uses business pressure language on a genuine Low-severity inquiry. These test whether the system is fooled by urgency framing in either direction.

**Prediction:** ADV-011 should be handled correctly — `demo request` maps to Low, and no security or outage keywords are present. ADV-008 is the harder case: the subject (`hours of operation`) maps to Low, and while `account` is in the text, `taken over` is not in the security keyword list. SIA may miss this one.

**Results observed:**

> *[Fill in.]*

**Known limitation:** The security keyword list (`P3_SECURITY_KWS`) covers specific phrases. Paraphrases of account takeover language that use different wording (`taken over`, `seized`, `someone else is in my account`) will not trigger the keyword rule signal. This is an acknowledged limitation of the rule-based component.

---

### Category E: Compound Discovery — Secondary Observation Contains the Real Issue (ADV-013, ADV-015)

These cases bury the actual problem as a secondary comment while the primary framing is innocuous. ADV-013 frames an access lockout as a billing upgrade problem. ADV-015 frames potential fraud as an incidental observation while doing a profile update.

**Prediction:** ADV-015 should be detected — the word `fraud` appears directly in the description and is in `P3_SECURITY_KWS`. ADV-013 is the more interesting case: `locked out` is not in the outage keyword list, so the keyword signal will likely miss the severity escalation.

**Results observed:**

> *[Fill in.]*

**Analysis:**

> *[For ADV-013, examine whether the TF-IDF classifier picks up on 'locked out' as a high-severity signal even without an explicit keyword hit. This is one case where the learned classifier may outperform the rule signal.]*

---

## Leakage Audit Summary

During Phase 1 of the project, `Issue_Category` was identified as a leakage feature and excluded from training. This section documents that finding for submission completeness.

**Finding:** A cross-tabulation of `Issue_Category` against `Priority_Level` showed that certain categories (e.g., "Fraud") mapped almost exclusively to specific priority levels. Including this feature in the training set would allow the classifier to predict `Priority_Level` through a direct proxy rather than from ticket content, defeating the purpose of the integrity audit.

**Action taken:** `Issue_Category` is excluded from all features in `train_pipeline.py` and `predict.py`. It does not appear in `text_combined`, is not passed to the TF-IDF vectorizer, and is not included in the numerical feature matrix.

**Fraud/Critical ratio observed:**

> *[Fill in from notebook output: "Ratio of Fraud tickets marked as Critical: XX.XX%"]*

---

## Integrity Audit Metrics (from `train_pipeline.py` run)

These values come from the `apply_integrity_audit()` function in the training pipeline. Fill in from the console output after running `train_pipeline.py`.

| Metric | Value |
|--------|-------|
| Total tickets processed | ___ |
| Mismatch rate | ___% |
| Cohen's Kappa (SIA vs Priority_Level) | ___ |
| Hidden Crisis tickets | ___ |
| False Alarm tickets | ___ |
| Consistent tickets | ___ |

**Interpretation of Cohen's Kappa:**

> *[Fill in. A Kappa between 0.60 and 0.80 would indicate substantial agreement between the fused severity inference and the assigned priority, suggesting most assigned priorities are defensible. A Kappa below 0.60 would indicate meaningful systematic disagreement — i.e., the dataset has real integrity issues worth auditing.]*

---

## Classifier Performance (from `train_pipeline.py` run)

Fill in from console output after running `train_pipeline.py`.

| Metric | Value |
|--------|-------|
| Test set accuracy | ___ |
| Test set Cohen's Kappa | ___ |

**Classification Report:**

```
              precision    recall  f1-score   support

         Low      ___       ___      ___       ___
      Medium      ___       ___      ___       ___
        High      ___       ___      ___       ___
    Critical      ___       ___      ___       ___

    accuracy                         ___       ___
```

---

## Dossier Validation Results (from `predict.py` run on adversarial set)

The six-rule validation suite runs automatically at the end of `predict.py`. Paste the output here.

```
============================================================
DOSSIER VALIDATION REPORT
============================================================

[paste output here]
```

---

## Known Limitations

These are structural limitations of the current SIA architecture. They are not bugs — they follow directly from the design choices documented in the notebook.

**Template dominance:** The template signal carries 60% of the fusion weight. For tickets where the subject subject line is a poor indicator of actual severity, this can anchor the inferred severity incorrectly regardless of description content. See ADV-001 through ADV-005.

**Keyword list coverage:** The security and outage keyword lists are fixed at the values used in Phase 1. Paraphrases, synonyms, and novel language for the same concepts will not trigger these rules. See ADV-008 and ADV-013 for specific examples.

**Resolution time as a label and feature:** `Resolution_Time_Hours` is used both in the fusion label (as `signal_time_num`) and as a feature in the classifier. During inference, this is defaulted to 24.0 hours. This means the time signal cannot contribute useful information for new tickets at inference time, which slightly degrades the classifier's ability to replicate the training-time fusion logic.

**No NLP beyond TF-IDF:** The model does not use sentence embeddings, transformers, or any semantic similarity measure. It relies on exact token overlap with the training vocabulary. Tickets with rare vocabulary that overlaps poorly with the training corpus will receive lower-confidence predictions.

---

## Conclusion

> *[Fill in after completing the results table and category analysis. Summarize overall adversarial accuracy, which categories performed best and worst, and what those results imply for real-world deployment of SIA. Reference specific ticket IDs and fusion scores.]*

---

*This template is part of the SIA submission package for MARS Open Project 2026.*
