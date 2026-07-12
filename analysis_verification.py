"""
================================================================================
PLAYFUL AI IN PROFESSIONAL EMAIL — VERIFICATION PIPELINE
Ben-Zion & Lazebnik | Nature Human Behaviour
================================================================================
Single reproducible pipeline for Code Availability.

INPUT:
    data/emails.csv   (16,880 emails, 121 senders)

OUTPUT:
    analysis_pipeline_outputs/canonical_results.txt   (all numbers used in the paper)
    analysis_pipeline_outputs/*.csv                   (tables)

Models (within-subject; every sender is their own control across 3 conditions):
    a-path  positivity ~ condition            (linear mixed model, random sender intercept)
    c-path  opened/replied ~ condition         (logistic GEE, sender clusters)   + Cox (strata sender)
    b-path  opened/replied ~ pos_within + cond (logistic GEE)  -> OR for within-sender positivity
    mediation  Sobel indirect = a * b (delta-method SE)

NOTE ON ENGINE: This Python pipeline is the VERIFICATION engine. The binary
mixed models in the manuscript were fit in R (lme4::glmer, subject-specific ORs);
statsmodels GEE here returns population-averaged ORs, so binary ORs may differ
slightly from the R values. The linear (positivity) model matches lmer closely.
The official R script reproduces the exact manuscript numbers.
================================================================================
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Exchangeable
from lifelines import CoxPHFitter

OUT = "analysis_pipeline_outputs"
os.makedirs(OUT, exist_ok=True)

CENSOR_OPEN = 14 * 1440       # 14 days, in minutes (matches merge_and_analyze.R)
CENSOR_REPLY = 40 * 1440      # 40 days, in minutes
CONDITIONS = ["no_llm", "professional_llm", "fun_llm"]

log_lines = []
def log(s=""):
    print(s)
    log_lines.append(str(s))

def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "+" if p < .10 else ""

# ============================================================
# 1. LOAD DATA
# ============================================================
df = pd.read_csv("data/emails.csv")

# types
for c in ["sender_id", "company_id", "condition_assigned", "recipient_type", "gender"]:
    df[c] = df[c].astype(str).str.strip().str.lower()
for c in ["age", "opened", "time_to_open_minutes", "replied", "time_to_reply_minutes",
          "positivity_score", "llm_detector_score"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["opened"] = df["opened"].fillna(0).clip(0, 1).astype(int)
df["replied"] = df["replied"].fillna(0).clip(0, 1).astype(int)
df["condition_assigned"] = pd.Categorical(df["condition_assigned"], categories=CONDITIONS, ordered=True)

# censored survival times
df["t_open"] = np.where(df["opened"] == 1, df["time_to_open_minutes"], CENSOR_OPEN)
df["t_reply"] = np.where(df["replied"] == 1, df["time_to_reply_minutes"], CENSOR_REPLY)

log("=" * 78)
log("CANONICAL DATASET")
log("=" * 78)
log(f"N emails  = {len(df)}")
log(f"N senders = {df['sender_id'].nunique()}")
log(f"N companies = {df['company_id'].nunique()}")
log("Condition counts:")
for c in CONDITIONS:
    log(f"   {c:18s} n = {(df['condition_assigned'] == c).sum()}")
log(f"Internal emails: {(df['recipient_type'] == 'internal').mean()*100:.1f}%  |  "
    f"External: {(df['recipient_type'] == 'external').mean()*100:.1f}%")

# demographics (per sender)
spp = df.drop_duplicates("sender_id")
log(f"Age: mean={spp['age'].mean():.1f}, SD={spp['age'].std():.1f}, "
    f"range {spp['age'].min():.0f}-{spp['age'].max():.0f}")
log(f"Gender: {dict(spp['gender'].value_counts())}")

# ============================================================
# 2. DESCRIPTIVES BY CONDITION
# ============================================================
log("\n" + "=" * 78)
log("DESCRIPTIVES BY CONDITION")
log("=" * 78)
rows = []
for c in CONDITIONS:
    g = df[df["condition_assigned"] == c]
    rows.append(dict(
        condition=c, n=len(g),
        open_rate=g["opened"].mean(), reply_rate=g["replied"].mean(),
        positivity_mean=g["positivity_score"].mean(), positivity_sd=g["positivity_score"].std(),
        median_t_open_h=g.loc[g["opened"] == 1, "time_to_open_minutes"].median() / 60,
        median_t_reply_h=g.loc[g["replied"] == 1, "time_to_reply_minutes"].median() / 60,
    ))
desc = pd.DataFrame(rows)
desc.to_csv(f"{OUT}/descriptives_by_condition.csv", index=False)
for _, r in desc.iterrows():
    log(f"   {r['condition']:18s} open={r['open_rate']*100:.1f}%  reply={r['reply_rate']*100:.1f}%  "
        f"pos={r['positivity_mean']:.3f} (SD {r['positivity_sd']:.2f})  "
        f"med_open={r['median_t_open_h']:.1f}h  med_reply={r['median_t_reply_h']:.1f}h")

# ============================================================
# 3. a-PATH: positivity ~ condition  (linear mixed model)
# ============================================================
log("\n" + "=" * 78)
log("a-PATH: positivity ~ condition + (1|sender)   [linear mixed model]")
log("=" * 78)
md = df.dropna(subset=["positivity_score"]).copy()
lmm = smf.mixedlm("positivity_score ~ C(condition_assigned)", md,
                  groups=md["sender_id"]).fit(method="lbfgs", reml=False)
lmm_null = smf.mixedlm("positivity_score ~ 1", md,
                       groups=md["sender_id"]).fit(method="lbfgs", reml=False)
lrt = 2 * (lmm.llf - lmm_null.llf)
p_omni = stats.chi2.sf(lrt, 2)
a_prof = lmm.params.get("C(condition_assigned)[T.professional_llm]")
a_fun = lmm.params.get("C(condition_assigned)[T.fun_llm]")
se_prof = lmm.bse.get("C(condition_assigned)[T.professional_llm]")
se_fun = lmm.bse.get("C(condition_assigned)[T.fun_llm]")
log(f"Omnibus LRT: chi2(2) = {lrt:.1f}, p = {p_omni:.3g} {stars(p_omni)}")
log(f"   fun_llm          B = {a_fun:+.4f} (SE {se_fun:.4f}), p = {lmm.pvalues['C(condition_assigned)[T.fun_llm]']:.3g}")
log(f"   professional_llm B = {a_prof:+.4f} (SE {se_prof:.4f}), p = {lmm.pvalues['C(condition_assigned)[T.professional_llm]']:.3g}")

# ============================================================
# 4. c-PATH: opened/replied ~ condition (logistic GEE) + Cox time-to-event
# ============================================================
log("\n" + "=" * 78)
log("c-PATH: direct condition effects on behavior")
log("=" * 78)
for outcome in ["opened", "replied"]:
    gee = smf.gee(f"{outcome} ~ C(condition_assigned)", groups="sender_id",
                  data=md, family=Binomial(), cov_struct=Exchangeable()).fit()
    wald = gee.wald_test("C(condition_assigned)[T.professional_llm] = 0, "
                         "C(condition_assigned)[T.fun_llm] = 0", scalar=True)
    log(f"\n{outcome}: omnibus Wald chi2(2) = {float(wald.statistic):.2f}, p = {float(wald.pvalue):.3g}")
    for c in ["professional_llm", "fun_llm"]:
        k = f"C(condition_assigned)[T.{c}]"
        log(f"   {c:18s} OR = {np.exp(gee.params[k]):.3f}, p = {gee.pvalues[k]:.3g}")

for outcome, tcol, censor in [("opened", "t_open", CENSOR_OPEN), ("replied", "t_reply", CENSOR_REPLY)]:
    d = md.copy()
    d["dur"] = d[tcol]
    dummies = pd.get_dummies(d["condition_assigned"], prefix="cond", drop_first=True).astype(float)
    cox_df = pd.concat([d[["dur", outcome, "sender_id"]].reset_index(drop=True),
                        dummies.reset_index(drop=True)], axis=1)
    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col="dur", event_col=outcome,
            strata="sender_id", robust=True)
    ll = cph.log_likelihood_ratio_test()
    log(f"\nCox {outcome} (strata sender): LRT chi2 = {ll.test_statistic:.2f}, p = {ll.p_value:.3g}")
    for term in dummies.columns:
        hr = np.exp(cph.params_[term])
        log(f"   {term:22s} HR = {hr:.3f}, p = {cph.summary.loc[term, 'p']:.3g}")

# ============================================================
# 5. b-PATH: within-sender centered positivity -> behavior
# ============================================================
log("\n" + "=" * 78)
log("b-PATH: opened/replied ~ pos_within + condition   [logistic GEE]")
log("(pos_within = positivity minus each sender's mean positivity)")
log("=" * 78)
md["pos_within"] = md["positivity_score"] - md.groupby("sender_id")["positivity_score"].transform("mean")
b_coef = {}
for outcome in ["opened", "replied"]:
    gee = smf.gee(f"{outcome} ~ pos_within + C(condition_assigned)", groups="sender_id",
                  data=md, family=Binomial(), cov_struct=Exchangeable()).fit()
    b = gee.params["pos_within"]; se_b = gee.bse["pos_within"]
    b_coef[outcome] = (b, se_b)
    log(f"   {outcome:8s} pos_within: OR = {np.exp(b):.3f}, "
        f"z = {gee.tvalues['pos_within']:.2f}, p = {gee.pvalues['pos_within']:.3g}")

# stratified by recipient type
log("\nb-path stratified by recipient type:")
for rt in ["internal", "external"]:
    sub = md[md["recipient_type"] == rt]
    for outcome in ["opened", "replied"]:
        try:
            gee = smf.gee(f"{outcome} ~ pos_within + C(condition_assigned)", groups="sender_id",
                          data=sub, family=Binomial(), cov_struct=Exchangeable()).fit()
            log(f"   {rt:9s} {outcome:8s} OR = {np.exp(gee.params['pos_within']):.3f}, "
                f"p = {gee.pvalues['pos_within']:.3g}")
        except Exception as e:
            log(f"   {rt} {outcome}: failed ({e})")

# ============================================================
# 6. MEDIATION: Sobel indirect effect a * b
# ============================================================
log("\n" + "=" * 78)
log("MEDIATION (Sobel): condition -> positivity -> behavior")
log("=" * 78)
med_rows = []
for cond, a, se_a in [("fun_llm", a_fun, se_fun), ("professional_llm", a_prof, se_prof)]:
    for outcome in ["opened", "replied"]:
        b, se_b = b_coef[outcome]
        ind = a * b
        se_ind = np.sqrt(a**2 * se_b**2 + b**2 * se_a**2)
        z = ind / se_ind
        p = 2 * stats.norm.sf(abs(z))
        med_rows.append(dict(condition=cond, outcome=outcome, a=a, b=b,
                             indirect=ind, sobel_z=z, p=p))
        log(f"   {cond:18s} -> {outcome:8s}: a={a:+.4f}, b={b:+.4f}, "
            f"indirect={ind:+.4f}, Sobel z={z:.2f}, p={p:.3g} {stars(p)}")
pd.DataFrame(med_rows).to_csv(f"{OUT}/mediation_sobel.csv", index=False)

# ============================================================
# 7. POSITIVITY QUARTILES -> open/reply rate
# ============================================================
log("\n" + "=" * 78)
log("POSITIVITY QUARTILES -> engagement")
log("=" * 78)
md["pos_q"] = pd.qcut(md["positivity_score"], 4, labels=["Q1(neg)", "Q2", "Q3", "Q4(pos)"])
q = md.groupby("pos_q", observed=True).agg(n=("opened", "size"),
                                           open_rate=("opened", "mean"),
                                           reply_rate=("replied", "mean"))
q.to_csv(f"{OUT}/positivity_quartiles.csv")
for idx, r in q.iterrows():
    log(f"   {idx:9s} n={int(r['n']):5d}  open={r['open_rate']*100:.1f}%  reply={r['reply_rate']*100:.1f}%")

# ============================================================
# 8. SAVE
# ============================================================
with open(f"{OUT}/canonical_results.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
log("\n" + "=" * 78)
log(f"DONE. Results written to {OUT}/canonical_results.txt")
