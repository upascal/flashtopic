# Jev as a feature extractor for prevalence: design + pilot

## Context

flashtopic's prevalence pillar (used by topic-explorer's PREVALENCE phase) estimates "what
fraction of documents discuss topic t" with a design-based PPI estimator:

    pi = mean_corpus(f) - HT_mean_gold(f - Y)

`f` is a cheap whole-corpus surrogate, `Y` a gold label on a stratified sample (~160
docs/topic, a generative LLM reading the full document against a codebook). The estimator is
unbiased for **any** `f`; `f` only controls the **CI width**, via the within-stratum variance
of `g = f - Y` (`estimate.py:129-172`). Today `f = f_memb` ("doc contains ≥1 member sentence",
`prototypes.py:50-73`): a crude binary that over-counts passing mentions and misses
paraphrases. Consequences: wide CIs on rare topics (`reliable=false`), and a gold budget of
~160 full-document LLM calls per topic, which is why topic-explorer scopes the phase to the
top 10 topics (`prevalence_service.py:63-66`).

The proposal: use Jev (non-autoregressive typed-question model; calibrated `p(yes)` / choice /
score distributions in one input-token-priced pass) as a **feature extractor**, fit a small
logistic regression on top using the gold labels, and use that calibrated probability as `f`.
Better `f` ⇒ smaller `Var(f − Y)` ⇒ tighter CI at the same gold budget, or the same CI with
fewer gold labels (so the budget can cover every topic). The gold judge stays a generative
LLM / human.

## 1. Where Jev sits (and where it must not)

| Role | Verdict | Why |
|---|---|---|
| Whole-corpus surrogate `f` (Jev features → LR) | **Yes, the design** | PPI stays unbiased; only variance improves; cheap enough for every doc. |
| Stratification score for the gold design | Yes | Raw zero-shot Jev presence prob replaces cosine for the negative bins, so the borderline bin actually holds the false negatives. Needs no gold, so no circularity. |
| Gold judge (replace the generative LLM) | **No** | Gold must be trusted. Jev is calibrated to TypeSafe's data, can't abstain or explain. Keep LLM-as-gold + human spot check. |
| Question drafting / error analysis | Generative LLM, once per topic | Jev can't say why it was wrong. |

## 2. Breaking a topic into Jev questions

Every topic already carries structured material that maps onto narrow typed questions. The
core set is generated **deterministically from topic structure, with no gold in the loop**
(pre-registered; see §3 pitfall on selection):

| Source (existing) | Question | Jev type | Feature(s) |
|---|---|---|---|
| `Topic.label` + codebook `definition` (put in `criteria`) | "Does the document discuss {label}?" | Noul | `p_presence` (as clipped logit) |
| The codebook's substantive-vs-passing rule (the documented main source of naive bias) | "How much of the document is devoted to {label}?" levels: none / passing mention / a sentence or two / a major part / the main subject | Score (5 levels) | expected level; `P(≥ "a sentence or two")` = direct analogue of `discusses` |
| `meta_theme` / `meta_cluster_id` parent label | "Does the document discuss {parent theme}?" | Noul, shared by siblings | `p_parent` (hierarchical gate) |
| Sibling set under one meta-topic + contrastive `differs` | "Which of these facets of {parent theme} does the document address most?" options = sibling labels + "none of these" | Choice, one per group | `p_choice[self]`, `p_choice[none]` |
| `discerned_subtopics` (only `coherence == "mixed"`) | one Noul per facet | Noul | facet probs |
| **LLM-drafted facets** (optional, v2) from codebook + exemplars + sibling `differs` — never from gold residuals | 2–4 narrow Nouls operationalizing the definition | Noul | facet probs |

Deterministic features alongside (zero cost, computable from `RunContext` unique sentences,
no full-text fetch): cosine `score`, `f_memb`, member sentences in doc, log doc chars,
count of `distinctive_terms` present.

Wire format (verified live against `typesafe/jev-1.13` on 2026-09-28): POST
`/api/alpha/decisions` with `{model: "typesafe/jev-1.13", state: <doc text>, questions: {name:
{type, instructions, criteria}}}`. `criteria` is required and typed: noul →
`{"true": ..., "false": ...}`; score → ordered level labels; choice → `{option: description}`.
Answers (under `answers`): `{noul: p}`, `{choice, probabilities:{opt:p}, confidence}`,
`{score, legend:{i:level}, probabilities:{i:p}, confidence}`. `typesafe/jev-latest` and
`typesafe/jev-router` are rejected by the endpoint.
Many questions per request are evaluated in parallel; Choice ≤ 255 options, Score 2–10
levels, 32K context. The `criteria` object carries the codebook (definition + positive
examples): integrators report structured criteria moving confidence from ~0.3 to ~0.97.
The per-request question cap is undocumented: the adapter chunks by meta-topic group
(~10–30 questions) and the pilot verifies. Question sets are hashed (`QUESTION_VERSION` +
content) so a changed codebook invalidates cached answers, as `codebook_hash` does for
verdicts.

Recommendation: start with presence + depth + parent + sibling-choice (≈2 questions per topic
+ 2 per group). Add drafted facets in v2, validated on **fresh** gold, not the gold used to
notice the gap.

## 3. The regression layer (statistically sound version)

- Per topic: L2 logistic regression (`sklearn`, already a dependency), `Y` = gold
  (`judge_label`, or `human_label` when present), `X` = Jev features + deterministic ones.
- **Cross-prediction, not "OOF for gold + full refit for corpus".** PPI needs the *same* `f`
  on both sides. K stratified folds by `hash(doc_idx) % K` (so a doc labeled for several
  topics never influences its own prediction for any topic); for fold k fit on the rest,
  predict fold k's gold docs **and** the whole corpus; `f_pop = mean_k f_pop^(k)`,
  `f_gold[i] = f^(k(i))[i]`. Jev features are computed once, so K corpus scorings are free.
- Weights: `sample_weight = (1/pi)^α`, α=1 (matches the design variance being minimized);
  fall back to α=0.5 when Kish effective sample size < ~40; report ESS.
- Rare topics: `MIN_POS_FOR_LR = 10`; below it use raw Jev `p_presence` with a 2-parameter
  Platt recalibration (also cross-fitted); below that the `memb` baseline. Candidate set is
  fixed in code (`memb`, `presence_platt`, `lr`), chosen by smallest design SE.
- Pooled model with per-topic intercepts is v2 (features are all topic-relative, so
  stacking is natural), after the pilot shows per-topic gains.
- Existing designs keep their stored `pi`/strata (only `f` is swapped); a phase-2 top-up
  draw uses **new** stratum ids so `pi` stays constant within stratum
  (`estimate.py:160`).
- Switching gold source (`re_estimate(human_first)`) must refit the surrogate from stored
  features, since the LR target changed.
- Keep `ppi_boot` as a cross-check: small residuals with n_h=15 can make the normal CI
  anti-conservative.

## 4. What changes for a continuous `f` (Phase B, not yet implemented)

`ppi_point` and `ppi_analytic_ci` already accept float `f`. Binary assumptions to fix:

- `estimate.py:318-325` `min_cell` uses `f_s == 1` — with `f ∈ (0,1)` every topic gets
  `min_cell=0`, `reliable=False`. Use `f_bin = f_s >= threshold`. Same for `tpr/fpr`
  (`:75-93`) and `acc_point` (`:96-107`, which must get `p_hat_hard = mean(f_pop >= t)`,
  not the soft mean). Add `threshold`, `naive_hard`, `surrogate_kind` to `PrevalenceResult`.
- `estimate.py:282` takes a full `f_pop`; topic-explorer rebuilds it from a count
  (`prevalence_service.py:173-174`), impossible for continuous `f`. Add
  `estimate_prevalence_from_summary(topic_id, n_docs, p_hat, p_hat_hard, f_s, y_s, pi_s,
  strata, ...)` and have `estimate_prevalence` delegate.
- `sampling.py:72-98`: add `pos_mask=` and `rank=` params (defaults `f_memb`/`score`, so
  backward compatible); `f_s` becomes float; `stratum_info` records the rank kind.
- `prototypes.py:33-43`: trailing optional `f_prob`, `f_kind`; property `f`; `naive =
  mean(f)`; `candidate_false_negatives` ranks by `f`.
- `report.py:30-65`: add `surrogate_kind`, `threshold`, `se`, `naive_hard` columns.
- `prevalence_service.py:137,173-174,309,559,625-626`; `models/analysis.py:202-204`
  (`f_prob`, `features` JSONB, `qset_hash` + a `_keep_if_same_qset` CASE); `routes/views.py:769`;
  `cost_service` line item for N docs × Jev input price; `_load_run_context` needs full
  corpus text for Jev (a whole-slice `fetch_texts`, which the phase currently avoids).
- `estimate.py:367` `_simulate`: add a continuous variant for the coverage self-check.

## 5. Phase A — the cheap pilot (implemented on this branch)

**One test, not two.** "Does the regression work" and "is prevalence estimated correctly"
are the same measurement. PPI is unbiased for any `f`, so correctness rests on the
probability sample, the gold labels and honest cross-fitting, none of which involve the
regression. The regression only moves the CI width, so the pilot's single headline number is
the design SE from out-of-fold gold residuals, per topic, for each candidate surrogate.
The two checks that stand apart are (i) gold quality (judge↔human agreement, the existing
spot check) and (ii) cross-fitting correctness (the leakage test + coverage simulation in
§6), which are code tests, not data tests.

Key fact: the analytic SE depends **only on gold-row residuals**
(`ppi_analytic_ci`), so Jev runs on the ~800 gold docs only, never the corpus.

Build in flashtopic (all new files under `src/flashtopic/prevalence/`, no changes to
existing modules in this phase):

1. `classifier.py` — `Question`/`Answer` dataclasses, `DocClassifier` protocol,
   `JevClassifier(model, transport=None)` for `POST /api/alpha/decisions` reusing
   `llm._config` (key/base URL), `llm._get_semaphore()` and `llm._backoff`;
   `MockClassifier(rule)`; per-`(doc, qid, classifier.name)` disk cache mirroring
   `judge.py:61-64`; `classify_docs(...)`. Never cache errors.
2. `questions.py` — `presence_question`, `depth_question`, `parent_question`,
   `sibling_question` (asserts ≤255), `facet_questions`, `build_question_set(topics,
   ctx_by_topic) -> QuestionSet` (dedups shared group questions; stable `hash`).
3. `features.py` — `deterministic_features(ctx, topic, sur, doc_ids)`,
   `classifier_features(answers, ...)` (noul → clipped logit; score → E[level] and
   P(≥ level 2); choice → p(self), p(none); missing → 0 + indicator), `assemble(...)`.
4. `surrogate.py` — `doc_folds`, `design_se` (= `ppi_analytic_ci(...)[2]`),
   `fit_cross_fitted(...)` (accepts `X_pop=None` in the pilot), `platt_presence`,
   `baseline_memb`, `choose_surrogate`; returns `FittedSurrogate` with `f_gold`, `f_pop`,
   `coef_mean`, `ess`, `se_gold`.
5. `pilot.py` — `compare_on_gold(...) -> PilotRow` and `run_pilot(...)`; `__main__` runs
   end-to-end with `MockClassifier` on the synthetic context from
   `tests/test_prevalence_pipeline.py::_ctx`.
6. A run script (kept outside this repo) — the real run, on a local machine (needs a cluster backend,
   spaCy model, `OPENROUTER_API_KEY`; not runnable in this container — no numpy, no GPU,
   no corpus). Takes a JSONL corpus path + `--text-field`, then:
   - `run_pipeline` + `label_model=` haiku (existing pipeline; hierarchy via
     `cluster_levels`/`label_hierarchy` if available, else parent/sibling questions are
     skipped and the report says so);
   - picks ~5 topics spanning common → rare; `build_surrogate`, `design_sample` (cosine
     stratified, as today), `build_codebook`, `run_judge` → gold `Y`
     (~5 × 160 haiku full-doc calls; cached);
   - builds the question set, runs `JevClassifier` on the same gold docs (cached);
   - fits three candidates per topic — (a) `memb` (today), (b) LR on deterministic
     features only, (c) LR on deterministic + Jev — cross-fitted;
   - writes `flashtopic_output/jev_pilot.md` + `.csv`: per topic `Var(f−Y)` per stratum,
     analytic CI half-width for (a)/(b)/(c) at the current gold size, the gold size (c) needs
     to match (a)'s CI, ESS, Jev raw `p_presence` AUC/Brier vs the judge, LR coefficients
     (which questions carry weight), how many gold positives land in the top negative bin
     under cosine vs Jev ranking, and a dump of the largest-residual gold docs with their
     Jev probs for the LLM-assisted error-analysis loop (v2 input only).

## 6. Verification

- Unit tests (no network; run in this container after `pip install -e .[dev]`, no cluster
  backend needed): `test_prevalence_questions.py` (deterministic, order-independent qids
  and hash; "none of these" present; >255 rejected; shared questions deduped);
  `test_prevalence_classifier.py` (mock satisfies protocol; cache hit skips call; fake
  `transport` round-trips noul/choice/score; malformed payload → retry → `Answer(error)`,
  not cached); `test_prevalence_features.py` (stable names; term hits from unique sentences
  with `doc_texts={}`; missing-answer indicator); `test_prevalence_surrogate.py` — the
  **leakage test**: pure-noise features give OOF `se_gold ≈ memb` baseline (an in-sample fit
  would be smaller); informative synthetic features give `se_lr < se_memb`; `< MIN_POS`
  routes to Platt; `doc_folds` identical across topics; `test_prevalence_pilot.py`
  (SEs equal `ppi_analytic_ci(...)[2]`; end-to-end with `MockClassifier`); a slow-marked
  coverage simulation (latent-feature world, cross-fitted LR, PPI coverage ≈ 95% over 100
  reps — guards the cross-prediction implementation).
- `pytest` on the existing suite stays green (no existing module is modified in Phase A).
- The pilot's decision rule, written into the report: build Phase B only if (c) beats (a)
  on CI half-width for most topics with cross-fitted residuals **and** (c) beats (b), which
  isolates Jev's contribution from the deterministic features.

## 7. Risks / unknowns the pilot must answer

- Jev's per-request question cap and behaviour on long documents ("five-second judgement"
  framing; depth questions on long docs may be weak).
- Whether raw Jev presence prob beats cosine as the stratification score.
- OpenRouter Decisions endpoint is alpha; `transport` injection isolates wire-format drift.
- Weight collapse (ESS) in the low-probability negative bin; the α fallback is the guard.
