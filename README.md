# Playful AI in Professional Email — Analysis Code

Reproducibility materials for **"Playful AI in Professional Email: A Field Experiment
on Tone and Recipient Engagement"** (Ben-Zion & Lazebnik).

A randomized crossover field experiment in which 121 employees across six companies
sent work emails under three conditions over three weeks — **Unaided** (own writing),
**Playful** (GPT-5 rewrite in a playful tone), and **Professional** (GPT-5 rewrite in a
professional tone). The canonical dataset is 16,880 emails.

## Data
The de-identified, email-level data are **not distributed in this repository**. Because
the study involves real workplace correspondence, the data are available **only on
reasonable request to the authors (Z. Ben-Zion & T. Lazebnik)**, in fully anonymized form
and subject to ethical considerations. No raw email text, names, or addresses exist in
the dataset.

The analysis scripts expect two files under `data/`:

| File | Emails | Description |
|---|---|---|
| `data/data.csv` | 12,854 | Original extract |
| `data/data_rest.csv` | 4,026 | Additional emails; merged → **16,880** (canonical dataset) |

Columns: `sender_id, gender, age, company_id, condition_assigned, recipient_type, opened,
time_to_open_minutes, replied, time_to_reply_minutes, positivity_score, llm_detector_score`.
`positivity_score` is an automated sentiment-polarity score (−1 to +1); `llm_detector_score`
is the LLM-writing-detector probability.

## Code

### `analysis_pipeline.R` — main / canonical analysis
Reproduces **every number reported in the manuscript** from the raw CSVs. Merges the two
files into the 16,880-email dataset and runs the within-subject models: the a-path
(positivity ~ condition, linear mixed model), the c-path (opened/replied ~ condition,
GLMM + Cox), the b-path (within-sender centered positivity → behavior, GLMM), and the
Sobel mediation test. Written for R ≥ 4.5.

```r
Rscript analysis_pipeline.R
# deps: tidyverse, lme4, lmerTest, survival, emmeans, multcomp
```

### `analysis_verification.py` — independent verification
An independent re-implementation in Python (statsmodels / lifelines) used to cross-check
the R results. Linear and survival models match R exactly; binary odds ratios differ
slightly (GEE population-averaged vs. glmer subject-specific).

```bash
python analysis_verification.py
# deps: pandas, numpy, scipy, statsmodels, lifelines
```

### `exploratory/initial_analysis.py`
The initial exploratory analysis (by T. Lazebnik), on the original 12,854-email extract
only. **Superseded** by `analysis_pipeline.R`; retained for transparency of the analysis
history. Does not reproduce the manuscript numbers.

## Outputs
Both pipelines write results to an `analysis_pipeline_outputs/` (R) or `analysis_outputs/`
(Python) directory: descriptive tables, model coefficients, the mediation results, and
positivity-quartile summaries.
