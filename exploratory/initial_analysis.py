# Exploratory analysis (T. Lazebnik): logistic GEE / MixedLM models.
# Complements the canonical analysis in ../analysis_pipeline.R.

# !pip -q install pandas numpy scipy statsmodels lifelines matplotlib openpyxl

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Exchangeable
from lifelines import CoxPHFitter, KaplanMeierFitter
from patsy import dmatrix

# -----------------------------
# Helper functions
# -----------------------------
def clip(value, low, high):
    return max(low, min(high, value))


# ============================================================
# 1. Settings
# ============================================================

DATA_PATH = "data/emails.csv"
OUTPUT_DIR = "analysis_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEFAULT_CENSORING_WINDOW_MINUTES = 7 * 24 * 60

LLM_DETECTOR_THRESHOLD = 0.5

# Preferred category order
PREFERRED_CONDITION_ORDER = [
    "business_as_usual",
    "no_llm",
    "professional_llm",
    "fun_llm"
]

PREFERRED_PERIOD_ORDER = ["pre", "experiment"]
PREFERRED_RECIPIENT_ORDER = ["internal", "external"]


# ============================================================
# 2. Load data
# ============================================================

df = pd.read_csv(DATA_PATH, na_values=["NA", "NaN", "nan", "", "null", "None"])

# Standardize column names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("Loaded data:")
print("Rows:", len(df))
print("Columns:", list(df.columns))


# ============================================================
# 3. Check required and optional columns
# ============================================================

minimal_required_cols = [
    "sender_id",
    "gender",
    "age",
    "company_id",
    "condition_assigned",
    "recipient_type",
    "opened",
    "time_to_open_minutes",
    "replied",
    "time_to_reply_minutes",
    "positivity_score",
    "llm_detector_score"
]

optional_cols = [
    "email_id",
    "period",
    "position",
    "compliance_flag"
]

missing_required = [c for c in minimal_required_cols if c not in df.columns]
if missing_required:
    raise ValueError(f"Missing required columns: {missing_required}")

missing_optional = [c for c in optional_cols if c not in df.columns]

print("\nOptional columns missing:")
print(missing_optional if missing_optional else "None")


# ============================================================
# 4. Add safe fallback columns if missing
# ============================================================

# email_id fallback
if "email_id" not in df.columns:
    df["email_id"] = [f"E{i:06d}" for i in range(1, len(df) + 1)]
    print("\nCreated email_id from row number.")

# period fallback
# Important: If period is missing, pre-post and DID analyses cannot be done.
if "period" not in df.columns:
    df["period"] = "experiment"
    HAS_PERIOD = False
    print("\nWARNING: period column is missing.")
    print("Pre-post and difference-in-differences analyses will be skipped.")
else:
    HAS_PERIOD = df["period"].nunique(dropna=True) >= 2

# position fallback
if "position" not in df.columns:
    df["position"] = "unknown"
    HAS_POSITION = False
    print("\nWARNING: position column is missing.")
    print("Models will include position='unknown' only, so position adjustment is not meaningful.")
else:
    HAS_POSITION = True

# compliance fallback
# Creates an exploratory compliance flag if missing.
# Recommended official coding:
# - fun_llm and professional_llm should usually have high detector scores.
# - no_llm should usually have low detector scores.
# - business_as_usual can be treated as not_applicable or analyzed separately.
if "compliance_flag" not in df.columns:
    def derive_compliance(row):
        cond = str(row["condition_assigned"]).lower()
        score = row["llm_detector_score"]

        if pd.isna(score):
            return "unknown"

        if cond in ["fun_llm", "professional_llm"]:
            return "compliant" if score >= LLM_DETECTOR_THRESHOLD else "non_compliant"

        if cond in ["no_llm"]:
            return "compliant" if score < LLM_DETECTOR_THRESHOLD else "non_compliant"

        if cond in ["business_as_usual"]:
            return "not_applicable"

        return "unknown"

    df["compliance_flag"] = df.apply(derive_compliance, axis=1)
    print("\nWARNING: compliance_flag column is missing.")
    print("Created derived compliance_flag using llm_detector_score threshold =", LLM_DETECTOR_THRESHOLD)


# ============================================================
# 5. Clean and type variables
# ============================================================

categorical_cols = [
    "sender_id",
    "gender",
    "company_id",
    "condition_assigned",
    "period",
    "recipient_type",
    "position",
    "compliance_flag"
]

for col in categorical_cols:
    df[col] = df[col].astype(str).str.strip().str.lower()

numeric_cols = [
    "age",
    "opened",
    "time_to_open_minutes",
    "replied",
    "time_to_reply_minutes",
    "positivity_score",
    "llm_detector_score"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Binary cleanup
df["opened"] = df["opened"].fillna(0).astype(int)
df["replied"] = df["replied"].fillna(0).astype(int)

# Clip impossible binary values
df["opened"] = df["opened"].clip(0, 1)
df["replied"] = df["replied"].clip(0, 1)

# Create censoring flags
df["open_censored"] = 1 - df["opened"]
df["reply_censored"] = 1 - df["replied"]

# Category ordering based on available values
available_conditions = [c for c in PREFERRED_CONDITION_ORDER if c in df["condition_assigned"].unique()]
other_conditions = [c for c in sorted(df["condition_assigned"].dropna().unique()) if c not in available_conditions]
CONDITION_ORDER = available_conditions + other_conditions

available_periods = [p for p in PREFERRED_PERIOD_ORDER if p in df["period"].unique()]
other_periods = [p for p in sorted(df["period"].dropna().unique()) if p not in available_periods]
PERIOD_ORDER = available_periods + other_periods

available_recipients = [r for r in PREFERRED_RECIPIENT_ORDER if r in df["recipient_type"].unique()]
other_recipients = [r for r in sorted(df["recipient_type"].dropna().unique()) if r not in available_recipients]
RECIPIENT_ORDER = available_recipients + other_recipients

df["condition_assigned"] = pd.Categorical(df["condition_assigned"], categories=CONDITION_ORDER, ordered=True)
df["period"] = pd.Categorical(df["period"], categories=PERIOD_ORDER, ordered=True)
df["recipient_type"] = pd.Categorical(df["recipient_type"], categories=RECIPIENT_ORDER, ordered=True)

# Pick reference condition
if "business_as_usual" in CONDITION_ORDER:
    REF_CONDITION = "business_as_usual"
elif "no_llm" in CONDITION_ORDER:
    REF_CONDITION = "no_llm"
else:
    REF_CONDITION = CONDITION_ORDER[0]

REF_RECIPIENT = "internal" if "internal" in RECIPIENT_ORDER else RECIPIENT_ORDER[0]
REF_PERIOD = "pre" if "pre" in PERIOD_ORDER else PERIOD_ORDER[0]

print("\nReference categories:")
print("Condition reference:", REF_CONDITION)
print("Recipient reference:", REF_RECIPIENT)
print("Period reference:", REF_PERIOD)

print("\nCondition counts:")
display(df["condition_assigned"].value_counts(dropna=False))

print("\nRecipient type counts:")
display(df["recipient_type"].value_counts(dropna=False))

print("\nPeriod counts:")
display(df["period"].value_counts(dropna=False))

print("\nCompliance counts:")
display(df["compliance_flag"].value_counts(dropna=False))


# ============================================================
# 6. Data audit
# ============================================================

audit = pd.DataFrame({
    "column": df.columns,
    "missing_n": df.isna().sum().values,
    "missing_pct": (100 * df.isna().mean().values).round(2),
    "unique_n": [df[c].nunique(dropna=True) for c in df.columns],
    "dtype": [str(df[c].dtype) for c in df.columns]
})

audit.to_csv(f"{OUTPUT_DIR}/data_audit.csv", index=False)
display(audit)


# ============================================================
# 7. Helper functions
# ============================================================

def stars(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.10:
        return "+"
    return ""


def save_table(table, name, index=True):
    csv_path = f"{OUTPUT_DIR}/{name}.csv"
    tex_path = f"{OUTPUT_DIR}/{name}.tex"

    table.to_csv(csv_path, index=index)

    try:
        table.to_latex(tex_path, index=index, escape=False)
    except Exception:
        pass

    print(f"Saved {csv_path}")
    return table


def summarize_group(g):
    return pd.Series({
        "n_rows": len(g),
        "n_emails": g["email_id"].nunique(),
        "n_senders": g["sender_id"].nunique(),
        "n_companies": g["company_id"].nunique(),

        "open_rate": g["opened"].mean(),
        "response_rate": g["replied"].mean(),

        "positivity_mean": g["positivity_score"].mean(),
        "positivity_median": g["positivity_score"].median(),
        "positivity_sd": g["positivity_score"].std(),

        "time_to_open_mean_observed": g.loc[g["opened"] == 1, "time_to_open_minutes"].mean(),
        "time_to_open_median_observed": g.loc[g["opened"] == 1, "time_to_open_minutes"].median(),
        "time_to_open_sd_observed": g.loc[g["opened"] == 1, "time_to_open_minutes"].std(),

        "time_to_reply_mean_observed": g.loc[g["replied"] == 1, "time_to_reply_minutes"].mean(),
        "time_to_reply_median_observed": g.loc[g["replied"] == 1, "time_to_reply_minutes"].median(),
        "time_to_reply_sd_observed": g.loc[g["replied"] == 1, "time_to_reply_minutes"].std(),

        "llm_detector_mean": g["llm_detector_score"].mean(),
        "llm_detector_median": g["llm_detector_score"].median(),

        "compliance_rate": (g["compliance_flag"].astype(str) == "compliant").mean()
    })


def extract_statsmodels_table(result, model_name):
    out = pd.DataFrame({
        "model": model_name,
        "term": result.params.index,
        "estimate": result.params.values,
        "std_error": result.bse.values,
        "p_value": result.pvalues.values
    })

    out["stars"] = out["p_value"].apply(stars)
    out["estimate_with_stars"] = out["estimate"].round(4).astype(str) + out["stars"]
    return out


def extract_cox_table(cph, model_name):
    out = cph.summary.reset_index().rename(columns={"covariate": "term"})
    out.insert(0, "model", model_name)
    out["stars"] = out["p"].apply(stars)
    return out


def primary_terms(table):
    term_col = "term"
    keep = (
        table[term_col].astype(str).str.contains("condition_assigned", case=False, regex=False)
        | table[term_col].astype(str).str.contains("period", case=False, regex=False)
        | table[term_col].astype(str).str.contains("recipient_type", case=False, regex=False)
    )
    return table.loc[keep].copy()


def make_model_df(data):
    d = data.copy()

    needed = [
        "condition_assigned",
        "recipient_type",
        "age",
        "gender",
        "position",
        "company_id",
        "sender_id"
    ]

    if HAS_PERIOD:
        needed.append("period")

    d = d.dropna(subset=needed)

    # Age fallback, just in case
    if d["age"].isna().any():
        d["age"] = d["age"].fillna(d["age"].median())

    return d


def get_censoring_window(data):
    observed_max = np.nanmax([
        data["time_to_open_minutes"].max(skipna=True),
        data["time_to_reply_minutes"].max(skipna=True),
        DEFAULT_CENSORING_WINDOW_MINUTES
    ])

    if pd.isna(observed_max) or observed_max <= 0:
        return DEFAULT_CENSORING_WINDOW_MINUTES

    # Ensure censored observations have at least as much follow-up as observed events.
    return float(max(DEFAULT_CENSORING_WINDOW_MINUTES, observed_max))


def make_survival_df(data, time_col, event_col, duration_col, event_name):
    d = make_model_df(data).copy()

    censoring_window = get_censoring_window(d)

    d[event_name] = d[event_col].astype(int)
    d[duration_col] = d[time_col]

    d.loc[d[event_name] == 0, duration_col] = censoring_window

    d[duration_col] = pd.to_numeric(d[duration_col], errors="coerce")
    d = d.dropna(subset=[duration_col])
    d = d[d[duration_col] > 0].copy()

    return d


def build_formula_rhs(include_period=True):
    base = (
        f'C(condition_assigned, Treatment(reference="{REF_CONDITION}"))'
        f' * C(recipient_type, Treatment(reference="{REF_RECIPIENT}"))'
    )

    if include_period and HAS_PERIOD:
        base = (
            f'C(condition_assigned, Treatment(reference="{REF_CONDITION}"))'
            f' * C(period, Treatment(reference="{REF_PERIOD}"))'
            f' * C(recipient_type, Treatment(reference="{REF_RECIPIENT}"))'
        )

    controls = " + age + C(gender) + C(position) + C(company_id)"
    return base + controls


def build_design_matrix_for_cox(data, include_period=True):
    rhs = build_formula_rhs(include_period=include_period)
    X = dmatrix(rhs, data, return_type="dataframe")

    if "Intercept" in X.columns:
        X = X.drop(columns=["Intercept"])

    # Drop zero-variance columns
    keep = X.nunique(dropna=False) > 1
    X = X.loc[:, keep]

    return X


def compliant_subset(data):
    flag = data["compliance_flag"].astype(str).str.lower().str.strip()

    keep = (
        flag.isin(["compliant", "not_applicable", "na", "nan", "unknown"])
    )

    # If there is a real pre-period, keep it by default because no treatment was applied yet.
    if HAS_PERIOD:
        keep = keep | data["period"].astype(str).eq("pre")

    return data.loc[keep].copy()


def remove_extreme_times(data, q=0.99):
    d = data.copy()

    open_cut = d.loc[d["opened"] == 1, "time_to_open_minutes"].quantile(q)
    reply_cut = d.loc[d["replied"] == 1, "time_to_reply_minutes"].quantile(q)

    keep_open = (d["opened"] == 0) | (d["time_to_open_minutes"] <= open_cut)
    keep_reply = (d["replied"] == 0) | (d["time_to_reply_minutes"] <= reply_cut)

    return d.loc[keep_open & keep_reply].copy()


# ============================================================
# 8. Sample and balance tables
# ============================================================

sample_balance = df.groupby("condition_assigned", observed=False).agg(
    n_rows=("email_id", "size"),
    n_emails=("email_id", "nunique"),
    n_senders=("sender_id", "nunique"),
    n_companies=("company_id", "nunique"),
    age_mean=("age", "mean"),
    age_sd=("age", "std"),
    detector_mean=("llm_detector_score", "mean"),
    detector_sd=("llm_detector_score", "std")
)

save_table(sample_balance, "table_1_sample_balance")

gender_balance = pd.crosstab(df["condition_assigned"], df["gender"], normalize="index")
save_table(gender_balance, "table_1_gender_balance")

position_balance = pd.crosstab(df["condition_assigned"], df["position"], normalize="index")
save_table(position_balance, "table_1_position_balance")

display(sample_balance)
display(gender_balance)
display(position_balance)


# ============================================================
# 9. Descriptive statistics
# ============================================================

desc_by_condition = (
    df
    .groupby(["condition_assigned"], observed=False)
    .apply(summarize_group)
    .reset_index()
)

desc_by_condition_recipient = (
    df
    .groupby(["condition_assigned", "recipient_type"], observed=False)
    .apply(summarize_group)
    .reset_index()
)

save_table(desc_by_condition, "table_2_descriptive_by_condition", index=False)
save_table(desc_by_condition_recipient, "table_2_descriptive_by_condition_recipient", index=False)

display(desc_by_condition)
display(desc_by_condition_recipient)

if HAS_PERIOD:
    desc_by_condition_period = (
        df
        .groupby(["condition_assigned", "period"], observed=False)
        .apply(summarize_group)
        .reset_index()
    )

    desc_by_condition_period_recipient = (
        df
        .groupby(["condition_assigned", "period", "recipient_type"], observed=False)
        .apply(summarize_group)
        .reset_index()
    )

    save_table(desc_by_condition_period, "table_2_descriptive_by_condition_period", index=False)
    save_table(desc_by_condition_period_recipient, "table_2_descriptive_by_condition_period_recipient", index=False)

    display(desc_by_condition_period)
    display(desc_by_condition_period_recipient)


# ============================================================
# 10. Pre-post and DID tables
# Only possible if period has both pre and experiment
# ============================================================

if HAS_PERIOD:
    def prepost_did_table(data, outcome, label):
        agg = (
            data
            .groupby(["condition_assigned", "period"], observed=False)[outcome]
            .mean()
            .reset_index()
            .pivot(index="condition_assigned", columns="period", values=outcome)
        )

        if "pre" not in agg.columns or "experiment" not in agg.columns:
            return None

        agg["change_experiment_minus_pre"] = agg["experiment"] - agg["pre"]

        if REF_CONDITION in agg.index:
            ref_change = agg.loc[REF_CONDITION, "change_experiment_minus_pre"]
            agg["did_vs_reference"] = agg["change_experiment_minus_pre"] - ref_change
        else:
            agg["did_vs_reference"] = np.nan

        agg.insert(0, "outcome", label)
        return agg.reset_index()

    prepost_tables = []

    for outcome, label in [
        ("opened", "open_rate"),
        ("replied", "response_rate"),
        ("positivity_score", "positivity_score")
    ]:
        t = prepost_did_table(df, outcome, label)
        if t is not None:
            prepost_tables.append(t)

    opened_only = df[df["opened"] == 1].copy()
    replied_only = df[df["replied"] == 1].copy()

    if len(opened_only) > 0:
        t = prepost_did_table(opened_only, "time_to_open_minutes", "time_to_open_minutes_observed")
        if t is not None:
            prepost_tables.append(t)

    if len(replied_only) > 0:
        t = prepost_did_table(replied_only, "time_to_reply_minutes", "time_to_reply_minutes_observed")
        if t is not None:
            prepost_tables.append(t)

    if prepost_tables:
        prepost_all = pd.concat(prepost_tables, ignore_index=True)
        save_table(prepost_all, "table_3_prepost_did", index=False)
        display(prepost_all)

else:
    print("\nSKIPPED: Pre-post and DID analyses require a period column with both pre and experiment.")


# ============================================================
# 11. Main regression models
# ============================================================

model_df = make_model_df(df)
print("\nModel dataset rows:", len(model_df))

rhs = build_formula_rhs(include_period=HAS_PERIOD)

response_formula = "replied ~ " + rhs
positivity_formula = "positivity_score ~ " + rhs

print("\nResponse model formula:")
print(response_formula)

print("\nPositivity model formula:")
print(positivity_formula)


# ----------------------------
# 11A. Response rate model
# Logistic GEE clustered by sender
# ----------------------------

gee_response = smf.gee(
    formula=response_formula,
    groups="sender_id",
    data=model_df,
    family=Binomial(),
    cov_struct=Exchangeable()
).fit()

response_table = extract_statsmodels_table(
    gee_response,
    "Response rate: logistic GEE clustered by sender"
)

save_table(response_table, "model_response_rate_gee", index=False)
save_table(primary_terms(response_table), "model_response_rate_gee_primary_terms", index=False)

print("\nResponse rate model:")
print(gee_response.summary())
display(primary_terms(response_table))


# ----------------------------
# 11B. Positivity model
# Linear mixed model if possible.
# Falls back to OLS clustered by sender.
# ----------------------------

try:
    positivity_lmm = smf.mixedlm(
        formula=positivity_formula,
        data=model_df,
        groups=model_df["sender_id"],
        vc_formula={"company": "0 + C(company_id)"}
    ).fit(method="lbfgs", maxiter=2000)

    positivity_table = extract_statsmodels_table(
        positivity_lmm,
        "Positivity: linear mixed model"
    )

    save_table(positivity_table, "model_positivity_lmm", index=False)
    save_table(primary_terms(positivity_table), "model_positivity_lmm_primary_terms", index=False)

    print("\nPositivity model:")
    print(positivity_lmm.summary())
    display(primary_terms(positivity_table))

except Exception as e:
    print("\nLinear mixed model failed; using OLS with sender-clustered standard errors.")
    print("Reason:", e)

    positivity_ols = smf.ols(
        formula=positivity_formula,
        data=model_df
    ).fit(cov_type="cluster", cov_kwds={"groups": model_df["sender_id"]})

    positivity_table = extract_statsmodels_table(
        positivity_ols,
        "Positivity: OLS clustered by sender"
    )

    save_table(positivity_table, "model_positivity_ols_clustered", index=False)
    save_table(primary_terms(positivity_table), "model_positivity_ols_clustered_primary_terms", index=False)

    print(positivity_ols.summary())
    display(primary_terms(positivity_table))


# ============================================================
# 12. Survival models
# Cox models for time to open and time to reply
# ============================================================

# ----------------------------
# 12A. Time to open
# ----------------------------

surv_open = make_survival_df(
    df,
    time_col="time_to_open_minutes",
    event_col="opened",
    duration_col="duration_to_open",
    event_name="event_opened"
)

X_open = build_design_matrix_for_cox(surv_open, include_period=HAS_PERIOD)

cox_open_df = pd.concat(
    [
        surv_open[["duration_to_open", "event_opened", "sender_id"]].reset_index(drop=True),
        X_open.reset_index(drop=True)
    ],
    axis=1
)

cph_open = CoxPHFitter()
cph_open.fit(
    cox_open_df,
    duration_col="duration_to_open",
    event_col="event_opened",
    cluster_col="sender_id",
    robust=True
)

cox_open_table = extract_cox_table(cph_open, "Time to open: Cox model")
save_table(cox_open_table, "model_time_to_open_cox", index=False)
save_table(primary_terms(cox_open_table), "model_time_to_open_cox_primary_terms", index=False)

print("\nCox model: time to open")
display(cph_open.summary)
display(primary_terms(cox_open_table))


# ----------------------------
# 12B. Time to reply
# ----------------------------

surv_reply = make_survival_df(
    df,
    time_col="time_to_reply_minutes",
    event_col="replied",
    duration_col="duration_to_reply",
    event_name="event_replied"
)

X_reply = build_design_matrix_for_cox(surv_reply, include_period=HAS_PERIOD)

cox_reply_df = pd.concat(
    [
        surv_reply[["duration_to_reply", "event_replied", "sender_id"]].reset_index(drop=True),
        X_reply.reset_index(drop=True)
    ],
    axis=1
)

cph_reply = CoxPHFitter()
cph_reply.fit(
    cox_reply_df,
    duration_col="duration_to_reply",
    event_col="event_replied",
    cluster_col="sender_id",
    robust=True
)

cox_reply_table = extract_cox_table(cph_reply, "Time to reply: Cox model")
save_table(cox_reply_table, "model_time_to_reply_cox", index=False)
save_table(primary_terms(cox_reply_table), "model_time_to_reply_cox_primary_terms", index=False)

print("\nCox model: time to reply")
display(cph_reply.summary)
display(primary_terms(cox_reply_table))


# ============================================================
# 13. Internal vs external stratified analyses
# ============================================================

for recipient_value in RECIPIENT_ORDER:
    sub = df[df["recipient_type"].astype(str) == recipient_value].copy()

    if len(sub) < 30:
        print(f"\nSkipping recipient_type={recipient_value}: too few observations.")
        continue

    sub_model = make_model_df(sub)

    # In stratified models, recipient_type is fixed, so remove it from the formula.
    if HAS_PERIOD:
        strat_rhs = (
            f'C(condition_assigned, Treatment(reference="{REF_CONDITION}"))'
            f' * C(period, Treatment(reference="{REF_PERIOD}"))'
            f' + age + C(gender) + C(position) + C(company_id)'
        )
    else:
        strat_rhs = (
            f'C(condition_assigned, Treatment(reference="{REF_CONDITION}"))'
            f' + age + C(gender) + C(position) + C(company_id)'
        )

    strat_response_formula = "replied ~ " + strat_rhs
    strat_positivity_formula = "positivity_score ~ " + strat_rhs

    print(f"\nStratified models for recipient_type={recipient_value}")

    try:
        strat_gee = smf.gee(
            formula=strat_response_formula,
            groups="sender_id",
            data=sub_model,
            family=Binomial(),
            cov_struct=Exchangeable()
        ).fit()

        strat_response_table = extract_statsmodels_table(
            strat_gee,
            f"Response rate: {recipient_value}"
        )

        save_table(
            strat_response_table,
            f"stratified_{recipient_value}_response_gee",
            index=False
        )

        display(primary_terms(strat_response_table))

    except Exception as e:
        print("Stratified response model failed:", e)

    try:
        strat_ols = smf.ols(
            formula=strat_positivity_formula,
            data=sub_model
        ).fit(cov_type="cluster", cov_kwds={"groups": sub_model["sender_id"]})

        strat_pos_table = extract_statsmodels_table(
            strat_ols,
            f"Positivity: {recipient_value}"
        )

        save_table(
            strat_pos_table,
            f"stratified_{recipient_value}_positivity_ols_clustered",
            index=False
        )

        display(primary_terms(strat_pos_table))

    except Exception as e:
        print("Stratified positivity model failed:", e)


# ============================================================
# 14. Robustness check: compliance-adjusted analysis
# ============================================================

df_compliant = compliant_subset(df)
print("\nCompliance-adjusted rows:", len(df_compliant), "out of", len(df))

desc_compliant = (
    df_compliant
    .groupby(["condition_assigned", "recipient_type"], observed=False)
    .apply(summarize_group)
    .reset_index()
)

save_table(desc_compliant, "robustness_compliance_descriptive", index=False)
display(desc_compliant)

if len(df_compliant) > 30:
    comp_model_df = make_model_df(df_compliant)

    try:
        comp_response = smf.gee(
            formula=response_formula,
            groups="sender_id",
            data=comp_model_df,
            family=Binomial(),
            cov_struct=Exchangeable()
        ).fit()

        comp_response_table = extract_statsmodels_table(
            comp_response,
            "Compliance-adjusted response rate: GEE"
        )

        save_table(comp_response_table, "robustness_compliance_response_gee", index=False)
        save_table(primary_terms(comp_response_table), "robustness_compliance_response_gee_primary_terms", index=False)

        display(primary_terms(comp_response_table))

    except Exception as e:
        print("Compliance-adjusted response model failed:", e)

    try:
        comp_positivity = smf.ols(
            formula=positivity_formula,
            data=comp_model_df
        ).fit(cov_type="cluster", cov_kwds={"groups": comp_model_df["sender_id"]})

        comp_positivity_table = extract_statsmodels_table(
            comp_positivity,
            "Compliance-adjusted positivity: OLS clustered by sender"
        )

        save_table(comp_positivity_table, "robustness_compliance_positivity_ols", index=False)
        save_table(primary_terms(comp_positivity_table), "robustness_compliance_positivity_ols_primary_terms", index=False)

        display(primary_terms(comp_positivity_table))

    except Exception as e:
        print("Compliance-adjusted positivity model failed:", e)


# ============================================================
# 15. Robustness check: remove extreme observed times
# ============================================================

df_no_extreme = remove_extreme_times(df, q=0.99)
print("\nRows after removing top 1% extreme observed times:", len(df_no_extreme), "out of", len(df))

desc_no_extreme = (
    df_no_extreme
    .groupby(["condition_assigned", "recipient_type"], observed=False)
    .apply(summarize_group)
    .reset_index()
)

save_table(desc_no_extreme, "robustness_no_extreme_times_descriptive", index=False)
display(desc_no_extreme)

if len(df_no_extreme) > 30:
    no_extreme_model_df = make_model_df(df_no_extreme)

    try:
        no_extreme_response = smf.gee(
            formula=response_formula,
            groups="sender_id",
            data=no_extreme_model_df,
            family=Binomial(),
            cov_struct=Exchangeable()
        ).fit()

        no_extreme_response_table = extract_statsmodels_table(
            no_extreme_response,
            "No-extreme-times response rate: GEE"
        )

        save_table(no_extreme_response_table, "robustness_no_extreme_times_response_gee", index=False)
        save_table(primary_terms(no_extreme_response_table), "robustness_no_extreme_times_response_gee_primary_terms", index=False)

        display(primary_terms(no_extreme_response_table))

    except Exception as e:
        print("No-extreme response model failed:", e)


# ============================================================
# 16. Figures
# ============================================================

def save_fig(name):
    path = f"{OUTPUT_DIR}/{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print("Saved", path)
    plt.show()


# Response rate by condition
plot_response = (
    df
    .groupby("condition_assigned", observed=False)["replied"]
    .mean()
    .reindex(CONDITION_ORDER)
)

plt.figure(figsize=(8, 5))
plot_response.plot(kind="bar")
plt.ylabel("Response rate")
plt.xlabel("Condition")
plt.title("Response rate by condition")
save_fig("figure_response_rate_by_condition")


# Open rate by condition
plot_open = (
    df
    .groupby("condition_assigned", observed=False)["opened"]
    .mean()
    .reindex(CONDITION_ORDER)
)

plt.figure(figsize=(8, 5))
plot_open.plot(kind="bar")
plt.ylabel("Open rate")
plt.xlabel("Condition")
plt.title("Open rate by condition")
save_fig("figure_open_rate_by_condition")


# Positivity by condition
plot_pos = (
    df
    .groupby("condition_assigned", observed=False)["positivity_score"]
    .mean()
    .reindex(CONDITION_ORDER)
)

plt.figure(figsize=(8, 5))
plot_pos.plot(kind="bar")
plt.ylabel("Mean positivity score")
plt.xlabel("Condition")
plt.title("Mean positivity score by condition")
save_fig("figure_positivity_by_condition")


# LLM detector score by condition
plot_detector = (
    df
    .groupby("condition_assigned", observed=False)["llm_detector_score"]
    .mean()
    .reindex(CONDITION_ORDER)
)

plt.figure(figsize=(8, 5))
plot_detector.plot(kind="bar")
plt.ylabel("Mean LLM detector score")
plt.xlabel("Condition")
plt.title("Mean LLM detector score by condition")
save_fig("figure_llm_detector_by_condition")


# Time to open observed boxplot
opened_only = df[df["opened"] == 1].copy()
if len(opened_only) > 0:
    plt.figure(figsize=(9, 5))
    opened_only.boxplot(column="time_to_open_minutes", by="condition_assigned", rot=45)
    plt.title("Observed time to open by condition")
    plt.suptitle("")
    plt.ylabel("Minutes")
    plt.xlabel("Condition")
    save_fig("figure_time_to_open_boxplot")


# Time to reply observed boxplot
replied_only = df[df["replied"] == 1].copy()
if len(replied_only) > 0:
    plt.figure(figsize=(9, 5))
    replied_only.boxplot(column="time_to_reply_minutes", by="condition_assigned", rot=45)
    plt.title("Observed time to reply by condition")
    plt.suptitle("")
    plt.ylabel("Minutes")
    plt.xlabel("Condition")
    save_fig("figure_time_to_reply_boxplot")


# Kaplan-Meier: time to open
plt.figure(figsize=(9, 6))
kmf = KaplanMeierFitter()

for condition in CONDITION_ORDER:
    sub = surv_open[surv_open["condition_assigned"].astype(str) == condition]
    if len(sub) == 0:
        continue

    kmf.fit(
        durations=sub["duration_to_open"],
        event_observed=sub["event_opened"],
        label=condition
    )
    kmf.plot_survival_function()

plt.title("Kaplan-Meier curves: time to open")
plt.xlabel("Minutes since sent")
plt.ylabel("Probability not yet opened")
save_fig("figure_km_time_to_open")


# Kaplan-Meier: time to reply
plt.figure(figsize=(9, 6))
kmf = KaplanMeierFitter()

for condition in CONDITION_ORDER:
    sub = surv_reply[surv_reply["condition_assigned"].astype(str) == condition]
    if len(sub) == 0:
        continue

    kmf.fit(
        durations=sub["duration_to_reply"],
        event_observed=sub["event_replied"],
        label=condition
    )
    kmf.plot_survival_function()

plt.title("Kaplan-Meier curves: time to reply")
plt.xlabel("Minutes since sent")
plt.ylabel("Probability not yet replied")
save_fig("figure_km_time_to_reply")


# If period exists, create period-based figures
if HAS_PERIOD:
    response_period = (
        df
        .groupby(["condition_assigned", "period"], observed=False)["replied"]
        .mean()
        .reset_index()
        .pivot(index="condition_assigned", columns="period", values="replied")
        .reindex(CONDITION_ORDER)
    )

    plt.figure(figsize=(9, 5))
    response_period.plot(kind="bar")
    plt.ylabel("Response rate")
    plt.xlabel("Condition")
    plt.title("Response rate by condition and period")
    save_fig("figure_response_rate_by_condition_period")

    positivity_period = (
        df
        .groupby(["condition_assigned", "period"], observed=False)["positivity_score"]
        .mean()
        .reset_index()
        .pivot(index="condition_assigned", columns="period", values="positivity_score")
        .reindex(CONDITION_ORDER)
    )

    plt.figure(figsize=(9, 5))
    positivity_period.plot(kind="bar")
    plt.ylabel("Mean positivity")
    plt.xlabel("Condition")
    plt.title("Positivity by condition and period")
    save_fig("figure_positivity_by_condition_period")


# ============================================================
# 17. Statistical tests: simple pairwise comparisons
# ============================================================

pairwise_rows = []

conditions = [c for c in CONDITION_ORDER if c in df["condition_assigned"].astype(str).unique()]

for i in range(len(conditions)):
    for j in range(i + 1, len(conditions)):
        c1, c2 = conditions[i], conditions[j]

        d1 = df[df["condition_assigned"].astype(str) == c1]
        d2 = df[df["condition_assigned"].astype(str) == c2]

        # Response rate: chi-square
        table_response = pd.crosstab(
            df[df["condition_assigned"].astype(str).isin([c1, c2])]["condition_assigned"],
            df[df["condition_assigned"].astype(str).isin([c1, c2])]["replied"]
        )

        try:
            chi2, p, _, _ = stats.chi2_contingency(table_response)
            pairwise_rows.append({
                "comparison": f"{c1} vs {c2}",
                "outcome": "response_rate",
                "test": "chi_square",
                "statistic": chi2,
                "p_value": p,
                "stars": stars(p)
            })
        except Exception:
            pass

        # Open rate: chi-square
        table_open = pd.crosstab(
            df[df["condition_assigned"].astype(str).isin([c1, c2])]["condition_assigned"],
            df[df["condition_assigned"].astype(str).isin([c1, c2])]["opened"]
        )

        try:
            chi2, p, _, _ = stats.chi2_contingency(table_open)
            pairwise_rows.append({
                "comparison": f"{c1} vs {c2}",
                "outcome": "open_rate",
                "test": "chi_square",
                "statistic": chi2,
                "p_value": p,
                "stars": stars(p)
            })
        except Exception:
            pass

        # Positivity: Welch t-test
        try:
            t, p = stats.ttest_ind(
                d1["positivity_score"].dropna(),
                d2["positivity_score"].dropna(),
                equal_var=False
            )

            pairwise_rows.append({
                "comparison": f"{c1} vs {c2}",
                "outcome": "positivity_score",
                "test": "welch_t",
                "statistic": t,
                "p_value": p,
                "stars": stars(p)
            })
        except Exception:
            pass

        # Observed time to open: Mann-Whitney
        try:
            x1 = d1.loc[d1["opened"] == 1, "time_to_open_minutes"].dropna()
            x2 = d2.loc[d2["opened"] == 1, "time_to_open_minutes"].dropna()

            if len(x1) > 0 and len(x2) > 0:
                u, p = stats.mannwhitneyu(x1, x2, alternative="two-sided")

                pairwise_rows.append({
                    "comparison": f"{c1} vs {c2}",
                    "outcome": "time_to_open_minutes_observed",
                    "test": "mann_whitney_u",
                    "statistic": u,
                    "p_value": p,
                    "stars": stars(p)
                })
        except Exception:
            pass

        # Observed time to reply: Mann-Whitney
        try:
            x1 = d1.loc[d1["replied"] == 1, "time_to_reply_minutes"].dropna()
            x2 = d2.loc[d2["replied"] == 1, "time_to_reply_minutes"].dropna()

            if len(x1) > 0 and len(x2) > 0:
                u, p = stats.mannwhitneyu(x1, x2, alternative="two-sided")

                pairwise_rows.append({
                    "comparison": f"{c1} vs {c2}",
                    "outcome": "time_to_reply_minutes_observed",
                    "test": "mann_whitney_u",
                    "statistic": u,
                    "p_value": p,
                    "stars": stars(p)
                })
        except Exception:
            pass

pairwise_tests = pd.DataFrame(pairwise_rows)
save_table(pairwise_tests, "pairwise_condition_tests", index=False)
display(pairwise_tests)


# ============================================================
# 18. Manuscript result snippets
# ============================================================

snippet_path = f"{OUTPUT_DIR}/manuscript_result_snippets.txt"

def fmt_pct(x):
    if pd.isna(x):
        return "NA"
    return f"{100*x:.1f}%"

def fmt_num(x, digits=2):
    if pd.isna(x):
        return "NA"
    return f"{x:.{digits}f}"

with open(snippet_path, "w") as f:
    f.write("AUTOMATIC MANUSCRIPT RESULT SNIPPETS\n")
    f.write("=" * 80 + "\n\n")

    f.write("DATA AVAILABILITY NOTE\n")
    f.write("-" * 80 + "\n")
    if HAS_PERIOD:
        f.write("The dataset includes a period variable, allowing pre-post and DID analyses.\n")
    else:
        f.write("The dataset does not include a period variable. Therefore, pre-post and difference-in-differences analyses cannot be estimated from this file.\n")

    if HAS_POSITION:
        f.write("The dataset includes sender position, allowing position-adjusted models.\n")
    else:
        f.write("The dataset does not include sender position. Models include a placeholder position='unknown'.\n")

    if "business_as_usual" in CONDITION_ORDER:
        f.write("The business-as-usual condition is present and used as the reference group.\n")
    else:
        f.write(f"The business-as-usual condition is not present. The reference condition is {REF_CONDITION}.\n")

    f.write("\nDESCRIPTIVE RESULTS\n")
    f.write("-" * 80 + "\n")

    for _, row in desc_by_condition.iterrows():
        f.write(
            f"In the {row['condition_assigned']} condition, "
            f"the open rate was {fmt_pct(row['open_rate'])}, "
            f"the response rate was {fmt_pct(row['response_rate'])}, "
            f"the mean positivity score was {fmt_num(row['positivity_mean'])}, "
            f"the observed median time to open was {fmt_num(row['time_to_open_median_observed'])} minutes, "
            f"and the observed median time to reply was {fmt_num(row['time_to_reply_median_observed'])} minutes.\n"
        )

    f.write("\nMODEL INTERPRETATION NOTES\n")
    f.write("-" * 80 + "\n")
    f.write("For the logistic response model, positive coefficients indicate higher log-odds of receiving a reply.\n")
    f.write("For the positivity model, positive coefficients indicate more positive outgoing language.\n")
    f.write("For Cox models, hazard ratios above 1 indicate faster opening or replying, while hazard ratios below 1 indicate slower opening or replying.\n")

print("Saved", snippet_path)


# ============================================================
# 19. Save complete model summaries
# ============================================================

summary_path = f"{OUTPUT_DIR}/model_summaries.txt"

with open(summary_path, "w") as f:
    f.write("MODEL SUMMARIES\n")
    f.write("=" * 80 + "\n\n")

    f.write("Response rate model: Logistic GEE clustered by sender\n")
    f.write("-" * 80 + "\n")
    f.write(str(gee_response.summary()))
    f.write("\n\n")

    f.write("Positivity model\n")
    f.write("-" * 80 + "\n")
    try:
        f.write(str(positivity_lmm.summary()))
    except Exception:
        try:
            f.write(str(positivity_ols.summary()))
        except Exception:
            f.write("No positivity model summary available.")
    f.write("\n\n")

    f.write("Time to open Cox model\n")
    f.write("-" * 80 + "\n")
    f.write(str(cph_open.summary))
    f.write("\n\n")

    f.write("Time to reply Cox model\n")
    f.write("-" * 80 + "\n")
    f.write(str(cph_reply.summary))
    f.write("\n\n")

print("Saved", summary_path)


# ============================================================
# 20. Export Excel workbook with all main tables
# ============================================================

excel_path = f"{OUTPUT_DIR}/analysis_tables.xlsx"

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    audit.to_excel(writer, sheet_name="data_audit", index=False)
    sample_balance.to_excel(writer, sheet_name="sample_balance")
    gender_balance.to_excel(writer, sheet_name="gender_balance")
    position_balance.to_excel(writer, sheet_name="position_balance")
    desc_by_condition.to_excel(writer, sheet_name="desc_condition", index=False)
    desc_by_condition_recipient.to_excel(writer, sheet_name="desc_cond_recipient", index=False)
    response_table.to_excel(writer, sheet_name="model_response", index=False)
    positivity_table.to_excel(writer, sheet_name="model_positivity", index=False)
    cox_open_table.to_excel(writer, sheet_name="cox_open", index=False)
    cox_reply_table.to_excel(writer, sheet_name="cox_reply", index=False)
    pairwise_tests.to_excel(writer, sheet_name="pairwise_tests", index=False)

    if HAS_PERIOD and "prepost_all" in globals():
        prepost_all.to_excel(writer, sheet_name="prepost_did", index=False)

print("Saved", excel_path)


# ============================================================
# 21. Zip outputs for download
# ============================================================

import shutil

zip_path = shutil.make_archive("analysis_outputs", "zip", OUTPUT_DIR)

print("\nDONE.")
print("All outputs saved in:", OUTPUT_DIR)
print("Zip file created:", zip_path)

try:
    from google.colab import files
    files.download(zip_path)
except Exception:
    print("If not running in Colab, manually download:", zip_path)