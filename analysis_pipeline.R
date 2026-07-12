# ==============================================================================
# PLAYFUL AI IN PROFESSIONAL EMAIL — ANALYSIS PIPELINE
# Ben-Zion & Lazebnik | Nature Human Behaviour
# ==============================================================================
# OFFICIAL analysis script for Code Availability. Reproduces every number in the
# manuscript. Companion Python verification: analysis_verification.py
#
# INPUT:
#   data/emails.csv   (16,880 emails, 121 senders)
#
# DESIGN: within-subject randomized crossover. Every sender serves as their own
# control across three conditions (no_llm / professional_llm / fun_llm).
#
# MODELS (all within-subject, random sender intercept):
#   a-path  positivity ~ condition                 (LMM)
#   c-path  opened/replied ~ condition             (GLMM binomial)  + Cox (strata sender)
#   b-path  opened/replied ~ pos_within + condition (GLMM binomial) -> within-sender positivity OR
#   mediation  Sobel indirect = a * b (delta-method SE)
#
# Run:  Rscript analysis_pipeline.R
# ==============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(lme4)
  library(lmerTest)
  library(survival)
  library(emmeans)
  library(multcomp)
})
emm_options(pbkrtest.limit = 20000, lmerTest.limit = 20000)
options(scipen = 3, digits = 4)

OUT <- "analysis_pipeline_outputs"
dir.create(OUT, showWarnings = FALSE)
CENSOR_OPEN  <- 14 * 1440    # 14 days (minutes)
CENSOR_REPLY <- 40 * 1440    # 40 days (minutes)

# ── 1. LOAD DATA ─────────────────────────────────────────────────────────────
df <- read_csv("data/emails.csv", show_col_types = FALSE) %>%
  mutate(
    condition  = factor(condition_assigned, levels = c("no_llm", "professional_llm", "fun_llm")),
    recip_type = factor(recipient_type, levels = c("internal", "external")),
    gender     = factor(gender),
    company    = factor(company_id),
    sender     = factor(sender_id),
    t_open     = if_else(opened  == 1, time_to_open_minutes,  CENSOR_OPEN),
    t_reply    = if_else(replied == 1, time_to_reply_minutes, CENSOR_REPLY)
  )

sink(file.path(OUT, "canonical_results.txt"), split = TRUE)
cat("================================================================\n")
cat("CANONICAL DATASET\n")
cat(sprintf("N emails = %d | N senders = %d | N companies = %d\n",
            nrow(df), n_distinct(df$sender), n_distinct(df$company)))
print(table(df$condition))
cat(sprintf("Internal %.1f%% | External %.1f%%\n",
            100*mean(df$recip_type=="internal"), 100*mean(df$recip_type=="external")))
spp <- df %>% distinct(sender, .keep_all = TRUE)
cat(sprintf("Age mean=%.1f SD=%.1f range %d-%d | Gender: %s\n",
            mean(spp$age), sd(spp$age), min(spp$age), max(spp$age),
            paste(names(table(spp$gender)), table(spp$gender), collapse=" ")))

# ── helpers ──────────────────────────────────────────────────────────────────
# robust to lme4 column naming ("Df" in >=1.1-35, "Chi Df" in older versions)
lrt_lab <- function(a) {
  k   <- nrow(a)
  chi <- a[["Chisq"]][k]
  dfx <- if ("Df" %in% names(a)) a[["Df"]][k] else a[["Chi Df"]][k]
  p   <- a[["Pr(>Chisq)"]][k]
  sprintf("chi2(%.0f) = %.2f, p = %.4g", dfx, chi, p)
}

# ── 2. DESCRIPTIVES BY CONDITION ─────────────────────────────────────────────
cat("\n=== DESCRIPTIVES BY CONDITION ===\n")
desc <- df %>% group_by(condition) %>%
  summarise(n = n(),
            open_rate  = mean(opened), reply_rate = mean(replied),
            pos_mean   = mean(positivity_score), pos_sd = sd(positivity_score),
            med_open_h  = median(time_to_open_minutes[opened==1])/60,
            med_reply_h = median(time_to_reply_minutes[replied==1])/60, .groups="drop")
print(desc)
write_csv(desc, file.path(OUT, "descriptives_by_condition.csv"))

# ── 3. a-PATH: positivity ~ condition (LMM) ──────────────────────────────────
cat("\n=== a-PATH: positivity ~ condition + (1|sender) ===\n")
m_pos  <- lmer(positivity_score ~ condition + (1 | sender), data = df, REML = FALSE)
m_pos0 <- lmer(positivity_score ~ (1 | sender), data = df, REML = FALSE)
cat("Omnibus LRT:", lrt_lab(anova(m_pos0, m_pos)), "\n")
print(summary(m_pos)$coefficients)
print(pairs(emmeans(m_pos, ~condition), adjust = "holm"))

# a-path x recipient interaction
m_int  <- lmer(positivity_score ~ condition * recip_type + gender + age + company + (1|sender),
               data = df, REML = FALSE)
m_ni   <- lmer(positivity_score ~ condition + recip_type + gender + age + company + (1|sender),
               data = df, REML = FALSE)
cat("\nInteraction LRT (condition x recipient):", lrt_lab(anova(m_ni, m_int)), "\n")
print(pairs(emmeans(m_int, ~condition | recip_type), adjust = "holm"))

# ── 4. c-PATH: direct condition effects ──────────────────────────────────────
cat("\n=== c-PATH: opened/replied ~ condition (GLMM) ===\n")
for (yv in c("opened", "replied")) {
  m  <- glmer(reformulate("condition + (1 | sender)", yv), data = df, family = binomial,
              control = glmerControl(optimizer = "bobyqa"))
  m0 <- glmer(reformulate("(1 | sender)", yv), data = df, family = binomial,
              control = glmerControl(optimizer = "bobyqa"))
  cat(sprintf("\n%s omnibus LRT: %s\n", yv, lrt_lab(anova(m0, m))))
  print(pairs(emmeans(m, ~condition, type = "response"), adjust = "holm"))
}
cat("\n=== c-PATH: time-to-event (Cox, strata sender) ===\n")
for (spec in list(c("t_open","opened"), c("t_reply","replied"))) {
  m <- coxph(as.formula(sprintf("Surv(%s, %s) ~ condition + strata(sender)", spec[1], spec[2])), data = df)
  cat(sprintf("\nCox %s: LRT chi2 = %.2f, p = %.3g\n", spec[2],
              summary(m)$logtest["test"], summary(m)$logtest["pvalue"]))
  print(summary(m)$coefficients)
}

# ── 5. b-PATH: within-sender centered positivity -> behavior ─────────────────
cat("\n=== b-PATH: opened/replied ~ pos_within + condition (GLMM) ===\n")
df <- df %>% group_by(sender) %>%
  mutate(pos_within = positivity_score - mean(positivity_score)) %>% ungroup()
bpath <- list()
for (yv in c("opened", "replied")) {
  m <- glmer(reformulate("pos_within + condition + (1 | sender)", yv), data = df,
             family = binomial, control = glmerControl(optimizer = "bobyqa"))
  cf <- summary(m)$coefficients["pos_within", ]
  bpath[[yv]] <- c(b = cf["Estimate"], se = cf["Std. Error"])
  cat(sprintf("%s: pos_within OR = %.3f, z = %.2f, p = %.3g\n",
              yv, exp(cf["Estimate"]), cf["z value"], cf["Pr(>|z|)"]))
}
cat("\nb-path stratified by recipient type:\n")
for (rt in c("internal", "external")) for (yv in c("opened", "replied")) {
  sub <- df %>% filter(recip_type == rt)
  m <- glmer(reformulate("pos_within + condition + (1 | sender)", yv), data = sub,
             family = binomial, control = glmerControl(optimizer = "bobyqa"))
  cf <- summary(m)$coefficients["pos_within", ]
  cat(sprintf("  %-8s %-8s OR = %.3f, p = %.3g\n", rt, yv, exp(cf["Estimate"]), cf["Pr(>|z|)"]))
}

# ── 6. MEDIATION: Sobel indirect a*b ─────────────────────────────────────────
cat("\n=== MEDIATION (Sobel): condition -> positivity -> behavior ===\n")
apath <- summary(m_pos)$coefficients
med <- list()
for (cond in c("professional_llm", "fun_llm")) {
  a  <- apath[paste0("condition", cond), "Estimate"]
  sa <- apath[paste0("condition", cond), "Std. Error"]
  for (yv in c("opened", "replied")) {
    b <- bpath[[yv]]["b.Estimate"]; sb <- bpath[[yv]]["se.Std. Error"]
    ind <- a * b
    se  <- sqrt(a^2*sb^2 + b^2*sa^2)
    z   <- ind/se; p <- 2*pnorm(-abs(z))
    cat(sprintf("  %-16s -> %-8s: a=%+.4f, b=%+.4f, indirect=%+.4f, Sobel z=%.2f, p=%.3g\n",
                cond, yv, a, b, ind, z, p))
    med[[length(med)+1]] <- tibble(condition=cond, outcome=yv, a=a, b=b,
                                   indirect=ind, sobel_z=z, p=p)
  }
}
write_csv(bind_rows(med), file.path(OUT, "mediation_sobel.csv"))

# ── 7. POSITIVITY QUARTILES -> engagement ────────────────────────────────────
cat("\n=== POSITIVITY QUARTILES -> engagement ===\n")
q <- df %>% mutate(pos_q = ntile(positivity_score, 4)) %>%
  group_by(pos_q) %>%
  summarise(n=n(), open_rate=mean(opened), reply_rate=mean(replied), .groups="drop")
print(q)
write_csv(q, file.path(OUT, "positivity_quartiles.csv"))

# ── 8. COMPLIANCE SENSITIVITY (per-protocol, LLM-detector) ───────────────────
# Intent-to-treat analyses above use every email under its assigned condition.
# Here we repeat the key models on a compliance-restricted sample to check that
# the null direct effects are not an artifact of imperfect LLM use.
cat("\n=== COMPLIANCE SENSITIVITY (LLM-detector, threshold 0.50) ===\n")
THRESH <- 0.50
df <- df %>% mutate(
  compliant = if_else(condition == "no_llm",
                      llm_detector_score <  THRESH,   # unaided: should NOT look LLM-edited
                      llm_detector_score >= THRESH)    # LLM arms: should look LLM-edited
)
crate <- df %>% group_by(condition) %>%
  summarise(n = n(), n_compliant = sum(compliant), compliance_rate = mean(compliant), .groups = "drop")
print(crate)
cat(sprintf("Overall compliance: %.1f%% (%d of %d emails)\n",
            100*mean(df$compliant), sum(df$compliant), nrow(df)))
write_csv(crate, file.path(OUT, "compliance_rate_by_condition.csv"))

dc <- df %>% filter(compliant)
cat(sprintf("Compliance-restricted N = %d | senders = %d\n", nrow(dc), n_distinct(dc$sender)))

# c-path (direct effects) on restricted sample
for (yv in c("opened", "replied")) {
  m  <- glmer(reformulate("condition + (1 | sender)", yv), data = dc, family = binomial,
              control = glmerControl(optimizer = "bobyqa"))
  m0 <- glmer(reformulate("(1 | sender)", yv), data = dc, family = binomial,
              control = glmerControl(optimizer = "bobyqa"))
  cat(sprintf("[compliant] %s omnibus LRT: %s\n", yv, lrt_lab(anova(m0, m))))
}

# b-path + Sobel mediation on restricted sample
dc <- dc %>% group_by(sender) %>%
  mutate(pos_within = positivity_score - mean(positivity_score)) %>% ungroup()
apath_c <- summary(lmer(positivity_score ~ condition + (1 | sender), data = dc, REML = FALSE))$coefficients
bp_c <- list()
for (yv in c("opened", "replied")) {
  m  <- glmer(reformulate("pos_within + condition + (1 | sender)", yv), data = dc,
              family = binomial, control = glmerControl(optimizer = "bobyqa"))
  cf <- summary(m)$coefficients["pos_within", ]
  bp_c[[yv]] <- c(b = cf["Estimate"], se = cf["Std. Error"])
  cat(sprintf("[compliant] %s: pos_within OR = %.3f, p = %.3g\n", yv, exp(cf["Estimate"]), cf["Pr(>|z|)"]))
}
for (cond in c("professional_llm", "fun_llm")) {
  a  <- apath_c[paste0("condition", cond), "Estimate"]
  sa <- apath_c[paste0("condition", cond), "Std. Error"]
  for (yv in c("opened", "replied")) {
    b <- bp_c[[yv]]["b.Estimate"]; sb <- bp_c[[yv]]["se.Std. Error"]
    ind <- a * b; se <- sqrt(a^2*sb^2 + b^2*sa^2); z <- ind/se; p <- 2*pnorm(-abs(z))
    cat(sprintf("[compliant] %-16s -> %-8s: indirect=%+.4f, Sobel z=%.2f, p=%.3g\n",
                cond, yv, ind, z, p))
  }
}

cat("\n================================================================\n")
cat("DONE. Outputs in", OUT, "\n")
sink()
