# Fake News Detection — LIAR Dataset

A binary classifier that labels short political claims as likely true or likely false,
ending in a small Streamlit demo app.

Class project. The goal is a working end-to-end pipeline and an **honestly measured**
result — not the highest number possible.

---

## Results

| model | accuracy | vs. floor | precision (fake) | recall (fake) | macro-F1 |
|---|---|---|---|---|---|
| Majority baseline | 56.8% | — | 0.00 | 0.00 | 0.36 |
| **Logistic Regression** (cutoff 0.45) | **63.7%** | **+7.0 pts** | 0.58 | 0.59 | 0.63 |
| Logistic Regression (cutoff 0.50) | 62.1% | +5.3 pts | 0.60 | 0.38 | 0.59 |
| Linear SVM | 61.8% | +5.1 pts | 0.59 | 0.40 | 0.59 |
| Multinomial Naive Bayes | 61.0% | +4.2 pts | 0.60 | 0.28 | 0.55 |

Published reference: **63.5%** — Galli et al. (2022), optimized Logistic Regression on LIAR.

### The number that matters: 56.8%

That is the **majority-class baseline** — what you score on the test set by ignoring the text
entirely and calling every claim "real". Any accuracy figure has to be read against it.
"63.7% accurate" on its own is misleading; "7 points better than guessing" is the real result.

### The three models are not actually different

McNemar's test on paired per-claim outcomes: Logistic Regression vs Linear SVM p = 0.74,
vs Naive Bayes p = 0.30, SVM vs Naive Bayes p = 0.43. All well above 0.05.

The 0.2-point gap between Logistic Regression and Linear SVM is **3 claims out of 1,281**.
They performed the same. Logistic Regression is preferred here for a different reason: you
can read its weights.

---

## What the model actually learned

The most useful finding in the project, and it is not in the accuracy table.

Because Logistic Regression keeps one weight per feature, the weights can be sorted:

| pushes toward FAKE | pushes toward REAL |
|---|---|
| `obama`, `obamacare`, `barack` | `percent`, `million`, `000` |
| `scott walker`, `wisconsin` | `more than`, `three`, `average` |
| `government`, `president`, `says` | `since`, `times`, `debt` |

The "fake" signals are **people and topics**. The "real" signals are **numbers**.

So the classifier has not learned to detect deception. It learned two simpler things:

1. **Which subjects attracted harsh PolitiFact ratings between 2007 and 2016** — a fact about
   what journalists chose to fact-check, not about truth.
2. **That claims citing specific figures tend to be rated true** — a real pattern, but a
   stylistic one. A fabricated statistic would sail straight past this model.

This matches published work: the 2021 study *"To trust a LIAR"* found models on this dataset
were largely learning speaker reputation rather than veracity. It explains the low ceiling
better than any amount of hyperparameter tuning.

---

## Method

Follows Galli et al., *"A comprehensive Benchmark for fake news detection"*
(J Intell Inf Syst, 2022): classical ML classifiers over TF-IDF text features, with BERT as
a heavier comparison.

**Evaluation protocol**, applied throughout:

- Hyperparameters and the decision cutoff tuned on the **validation** split only
- The test split used **once**, at the end
- Every model scored on identical rows so comparisons are paired
- Confusion matrix reported for every model, not just the best
- Accuracy never reported without the 56.8% floor next to it

---

## Data

[LIAR](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip) — William Yang Wang,
*"Liar, Liar Pants on Fire: A New Benchmark Dataset for Fake News Detection"*, ACL 2017.

12,836 short claims from PolitiFact, split 10,269 / 1,284 / 1,283. After cleaning:
10,261 / 1,284 / 1,281.

**The dataset is not committed.** Notebook 1 downloads it into `data/` on first run.

### Labels

Six PolitiFact ratings collapsed to binary:

| binary | ratings |
|---|---|
| 1 = fake | pants-fire, false, barely-true |
| 0 = real | half-true, mostly-true, true |

An alternative mapping counting `half-true` as fake is carried through as a sensitivity
check. The 3/3 split was chosen because it matches published work and keeps the classes
near-balanced — the 4/2 split pushes the floor to 64.1%, which would make a 62% model score
worse than a constant predictor.

### Excluded columns

The five speaker credit-history columns are **dropped**. Per the dataset README they include
the rating of the claim being predicted, so using them leaks the answer. This is a known trap
with LIAR.

---

## Repo contents

| file | contents | status |
|---|---|---|
| `step1_data_prep.ipynb` | load, relabel, clean, diagnostics | done |
| `step2_baseline.ipynb` | TF-IDF, Logistic Regression, SVM and Naive Bayes comparison | done |
| `step3_bert.ipynb` | fine-tuned BERT comparison | stretch goal |
| `app.py` | Streamlit demo | planned |

---

## Running it

Open either notebook in [Google Colab](https://colab.research.google.com/) and run top to
bottom. Both are self-contained — notebook 2 rebuilds the cleaned data automatically if it
is not already there.

Dependencies: `pandas`, `scikit-learn`, `matplotlib`, `scipy` (all preinstalled in Colab)
and `streamlit` for the app. The BERT stretch goal additionally needs `transformers` and a
GPU runtime (**Runtime → Change runtime type → T4 GPU**).

Notebook 2 saves `fake_news_model.joblib`, which the app loads.

---

## Known limitations

State these plainly rather than hoping nobody asks.

1. **The gain is modest.** About 7 points over a trivial baseline. Real, but small.
2. **Claims average ~18 words.** TF-IDF over a single sentence is a very thin feature vector.
   This caps performance regardless of which classifier is used.
3. **US politics, 2007–2016 only.** It will not transfer to other countries, other topics,
   or later events. This is not a general fake-news detector.
4. **PolitiFact ratings are editorial judgments**, not ground truth. The sample is not random
   either — PolitiFact checks claims it finds interesting, which skews toward the contentious.
5. **The model keys on topic and style, not truth** — see above. This is the central caveat.
6. **The binary cut point is a design decision**, not a property of the data.
7. **Wrongly flagging a true claim is worse than missing a fake one.** In any real deployment
   a false positive accuses someone who did nothing wrong, so precision matters more than the
   headline score. The model is advisory: it suggests, a person decides.

---

## References

- Wang, W. Y. (2017). *"Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News
  Detection.* ACL 2017.
- Galli, A., Masciari, E., Moscato, V., Sperlí, G. (2022). *A comprehensive Benchmark for
  fake news detection.* Journal of Intelligent Information Systems.
  https://doi.org/10.1007/s10844-021-00646-9
- *"To trust a LIAR": Does Machine Learning Really Classify Fine-grained Fake News?*
  OHARS 2021. https://ceur-ws.org/Vol-3012/OHARS2021-paper1.pdf
