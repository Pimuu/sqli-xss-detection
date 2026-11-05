# Beyond the Random Split: Robustness of Character n-gram and Hand-crafted-Feature Classifiers for SQL-Injection and XSS Payload Detection

*Research mini-project — reproducible code, data and results in this repository.*

---

## Abstract

SQL injection (SQLi) and cross-site scripting (XSS) are still among the most
common web-application attacks. Machine-learning (ML) detectors are often
proposed as a complement to signature-based web application firewalls (WAFs).
This paper builds a reproducible benchmark. It contains 20,000 synthetic
web-request strings: 10,000 benign, 5,000 SQLi and 5,000 XSS. The strings
cover 14 attack families and 11 obfuscation techniques, and 3,356 benign
strings are "hard" because they contain SQL or HTML vocabulary. We compare
five three-class classifiers. Three use character 1–4-gram TF-IDF (Naive
Bayes, logistic regression, linear SVM). One is a random forest on 28
hand-crafted lexical features. One is a hybrid of both representations.

On a stratified random split every TF-IDF model exceeds 0.98 macro-F1, and
the linear SVM classifies all 4,000 test strings correctly. This ceiling hides
large differences in robustness, which three harder protocols reveal:

* **Obfuscations held out from training.** Macro-F1 falls to 0.79–0.87 on raw
  input. WAF-style canonicalisation recovers 0.06–0.10 macro-F1 for every
  model. The hybrid model is best (0.9645).
* **Benign padding.** Wrapping a payload in innocuous prose cuts whole-string
  TF-IDF detection to at most 0.2 %. Max-pooling the classifier over 48-character
  sliding windows restores it to 96.2 % and adds no false positives. The
  linear SVM then reaches 0.9895 macro-F1 on the shift test.
* **Unseen attack families.** When a whole family is left out of training,
  hand-crafted features detect 95.4 % of it on average. The best TF-IDF model
  detects 81.8 %.

We also report a shortcut: SQL keyword capitalisation is the single most
important hand-crafted feature. All payloads are inert strings, and the full
pipeline regenerates every number and figure in this paper with one command.

---

## 1 Introduction

Injection flaws have appeared in the OWASP Top 10 in every edition. In 2021
they ranked third, and XSS is now counted in the same category [1]. Two attack
types dominate:

* **SQL injection** places attacker-controlled text inside a database query.
  It can bypass authentication, leak or modify data and, in some
  configurations, execute operating-system commands [2, 3].
* **Cross-site scripting** places attacker-controlled markup or script in a
  page that a victim renders. It can steal sessions and perform actions in the
  victim's name [4].

The primary defences are well known: parameterised queries and context-aware
output encoding. However, legacy code, third-party components and
configuration mistakes leave gaps. Organisations therefore deploy web
application firewalls (WAFs), which inspect requests before they reach the
application. ModSecurity with the OWASP Core Rule Set (CRS) is the canonical
open-source example [5, 6]. Signature WAFs are effective against known
patterns, but they have two weaknesses. They are brittle against
obfuscation [7, 8], and they produce false positives on legitimate input that
merely *looks* like code.

ML classifiers trained on payload strings promise to generalise beyond
hand-written signatures, and they have been studied extensively, from token
graphs with SVMs [9] to deep networks [10]. Many published evaluations have
three things in common:

* a random train/test split of a single dataset;
* accuracy above 99 %;
* little analysis of what happens when the attacker does something the
  training data did not contain.

Arp et al. [11] list exactly these practices (sampling bias, data snooping,
lab-only evaluation) among the most common pitfalls of ML in security.

This project asks a narrower question that a practitioner can act on: **which
design choices make a payload classifier robust when the attacker changes
encoding, attack family or surrounding text?** We contribute:

1. **A documented synthetic dataset.** It has 20,000 strings, with
   per-sample labels for attack family, obfuscation technique, parameter
   context and a "hard benign" flag. All payloads are inert and the generator
   is deterministic.
2. **A controlled comparison of five classifiers.** The classifiers use
   character n-gram TF-IDF, hand-crafted features or both. They are trained
   for 3-class classification (benign/SQLi/XSS) and also evaluated as binary
   detectors.
3. **Three robustness protocols beyond the random split:**
   * *obfuscation shift*: six encodings never seen in training;
   * *benign-padding mimicry*;
   * *leave-one-attack-family-out*.
4. **Two inexpensive defences, measured.** WAF-style canonicalisation, and
   sliding-window max-pooling against padding. We also report an honest
   account of failure modes and of a dataset shortcut.

## 2 Background

### 2.1 Injection attacks

**SQLi.** An SQLi payload breaks out of the data context of a query, usually
by closing a string literal with a quote or by ending a numeric expression.
It then appends SQL syntax. Halfond et al. [2] classify the main techniques,
and we use the same families:

| Family | What it does | Example |
|---|---|---|
| Tautology | makes a condition always true | `' OR 1=1--` |
| UNION-based | appends a second result set | `' UNION SELECT username,password FROM users--` |
| Error-based | leaks data through DBMS error messages | `extractvalue`, `updatexml`, type-conversion errors |
| Piggy-backed (stacked) | adds a second statement | `; DROP TABLE …` |
| Blind (boolean) | infers data one bit at a time from responses | `AND SUBSTRING(…)='a'` |
| Blind (time-based) | infers data from response delays | `SLEEP`, `WAITFOR DELAY`, `pg_sleep`, `BENCHMARK` |
| Column probing | finds the number of columns for a UNION | `ORDER BY n` |

Comment sequences (`--`, `#`, `/*`) remove the remainder of the original query.

**XSS.** An XSS payload breaks out of an HTML text, attribute, URL or
JavaScript context and introduces executable script. Common vectors:

* `<script>` elements;
* event-handler attributes such as `onerror`, `onload` or `onfocus` on
  `<img>`, `<svg>`, `<body>` or `<details>`;
* `javascript:` URIs in `href`/`src`;
* `data:` URIs;
* attribute or tag breakouts such as `"><…` and `</textarea><…`;
* CSS vectors;
* client-side template injection, e.g. `{{constructor.constructor(…)()}}`.

The OWASP XSS Filter Evasion Cheat Sheet catalogues hundreds of variants [7].
Mutation-based XSS [12] shows that the browser's own parsing can turn an
apparently harmless string into executable markup.

**Obfuscation.** Both attack classes are routinely obfuscated, because
detectors and applications decode input differently [7, 8]. Common
techniques:

* mixed case;
* percent-encoding, single or double (RFC 3986 [13]);
* HTML character references;
* JavaScript `\uXXXX` escapes;
* SQL inline comments used as whitespace (`/**/`);
* MySQL version-conditional comments (`/*!50000SELECT*/`);
* alternative whitespace (`%09`, `%0a`);
* building strings from character codes (`CHAR(83,69,…)`, `0x…`,
  `String.fromCharCode(…)`).

### 2.2 Web application firewalls

ModSecurity [5] matches requests against rules after applying a chain of
*transformation functions*, such as `urlDecodeUni`, `htmlEntityDecode`,
`jsDecode`, `replaceComments`, `compressWhitespace` and `lowercase`. These
reduce the space of encodings each signature must cover. The OWASP CRS [6]
adds anomaly scoring and paranoia levels, which let operators trade detection
against false positives. libinjection [14] tokenises input and compares token
"fingerprints" against a list of known SQLi shapes. Signature WAFs can be
evaded systematically. Appelt et al. used ML-guided mutation to generate
bypasses for SQLi filters [8, 15]. WAF-A-MoLE [16] applied adversarial
mutation to ML-based WAFs.

### 2.3 Machine learning for payload detection

Learning-based web attack detection goes back at least to Kruegel and
Vigna [17], who built anomaly models of HTTP parameters. Supervised approaches
include:

* SVMs over token graphs (SQLiGoT [9]);
* classifiers over JavaScript features for obfuscated script [18];
* convolutional and recurrent networks over characters or word
  embeddings [10].

Character n-grams are a strong, simple baseline. They need no tokenizer for
mixed SQL, HTML and JS, and they capture sub-word syntax such as `'--` or
`<scr`. Hand-crafted features encode expert knowledge: special-character
density, keyword counts, entropy and length. These features are cheap and
interpretable.

Two lines of work warn that high laboratory accuracy may not survive
deployment:

* **Distribution shift.** Sommer and Paxson [19] argued that intrusion
  detection is an adversarial, "closed-world-violating" setting, and Arp et
  al. [11] documented ten recurring pitfalls.
* **Adversarial evasion.** Biggio and Roli [20] review this attack surface.
  Lowd and Meek's "good word" attack on spam filters [21], which adds benign
  tokens to drown the malicious signal, is the direct ancestor of the
  benign-padding attack we evaluate.

Shortcut learning [22] explains how models reach high accuracy by exploiting
features that are correlated with the label in the training set but not
causally related to it.

## 3 Methodology

### 3.1 Threat model

We model a detector placed at the WAF position. It sees each HTTP parameter
value as a string and outputs *benign*, *SQLi* or *XSS*. The attacker is a
remote client with full control of parameter values. The attacker wants an
injection payload to pass the detector, and may:

* (a) use any standard payload family;
* (b) apply known encodings;
* (c) apply encodings absent from the defender's training data;
* (d) use an attack family absent from training;
* (e) surround the payload with benign text.

We do not model:

* adaptive attackers that query the model and optimise against it;
* multi-parameter or second-order attacks;
* whether a payload would actually succeed against a particular backend.

A sample is malicious by construction.

### 3.2 Dataset generation

All data is produced by `src/data_generation.py` (seed 42). Each sample is
created in three steps. Each step is applied to every class with the same
probabilities, so that no step reveals the label by itself.

**(1) Base string.**

* *Benign* strings come from 14 generators covering 20 fine-grained families:
  * search queries, personal names (including `O'Brien`), emails, numbers,
    dates, phones and product IDs;
  * reviews with emoticons such as `<3` and `:)`, and free text drawn from a
    vocabulary that includes *select*, *union*, *drop*, *script* and *alert*;
  * developer questions containing literal SQL/HTML, e.g. `select * from
    users where id = 5 mysql tutorial` and `what does the <script> tag do in
    html`;
  * URLs, JSON snippets, symbol-heavy passwords, UUID/base64 tokens and
    colour codes;
  * file paths, arithmetic (`a < b && b > c`), postal addresses and
    non-Latin Unicode text.
* Generators that routinely contain quotes, `=`, `<` or SQL/HTML words set
  `hard_benign=True`. This applies to 3,356 benign rows.
* *SQLi* strings come from 50 templates in the 7 families of §2.1. The
  templates are randomised over:
  * 12 breakout prefixes (`'`, `"`, `')`, `1'`, …);
  * 7 distinct comment terminators, including none;
  * table, column, integer and column-list fills.
* *XSS* strings come from 45 templates in 7 families. The script body is
  always a harmless marker (`alert(1)`, `prompt(1)`, `console.log(n)`, …),
  and external hosts use reserved `.example`/`.test` names [23].
* In 12 % of malicious samples the payload is prefixed by a plausible value
  (`laptop' OR 1=1--`).

**(2) Obfuscation.**

* Each malicious sample is plain 40 % of the time. Otherwise it receives one
  of 11 transforms.
* Five transforms are *seen*, meaning they may appear in training:
  * `case` (random case);
  * `comment_ws` (`/**/` as whitespace);
  * `url_encode`;
  * `html_entity`;
  * `tag_slash` (`<img/src=x/onerror=…>`).
* Six transforms are *held out* for experiment E3:
  * `double_url`;
  * `whitespace_alt` (`%09`/`%0a`/`%0d`/`%0c`);
  * `versioned_comment` (`/*!50000UNION*/` for SQLi; JS comment splitting for
    XSS);
  * `unicode_escape` (`'`, `<`);
  * `char_encoding` (`CHAR(…)`/`0x…` for SQLi; `eval(String.fromCharCode(…))`
    for XSS);
  * `benign_padding`: the payload is placed between two blocks of 2–5
    generated review sentences.
* Benign samples are URL-encoded 15 % of the time and entity-encoded 3 % of
  the time, as real form submissions are, so that `%` or `&#` alone is not
  evidence.

**(3) Context.** Every sample is wrapped with the same odds: 65 % raw value,
25 % `param=value`, 10 % `a=b&param=value`.

Exact duplicates are removed during generation, and generation continues
until each class reaches its target count. Table 1 summarises the dataset, and
Figure 1 shows the length distributions.

*Table 1. Dataset composition. Obfuscation shares are of each class; "param context" means the string is wrapped as `param=value` or in a multi-parameter query string.*

| Class | n | families | plain (no obfuscation) | seen-obf. | held-out-obf. | param context | median len | mean len | max len |
|---|---|---|---|---|---|---|---|---|---|
| benign | 10,000 | 20 | 80.5 % | 19.5 % | 0.0 % | 37.0 % | 29 | 32.0 | 167 |
| sqli | 5,000 | 7 | 39.2 % | 34.1 % | 26.6 % | 36.9 % | 44 | 59.0 | 422 |
| xss | 5,000 | 7 | 32.3 % | 37.2 % | 30.5 % | 43.6 % | 58 | 77.5 | 439 |

![Figure 1](../results/figures/length_distribution.png)

*Figure 1. Length distribution by class (share of each class per log-spaced
bin). The overlap is large. The benign median is 29 characters, SQLi 44 and
XSS 58, so length alone is a weak discriminator. The long right tails of SQLi
and XSS are the `benign_padding` samples.*

### 3.3 Canonicalisation

`canonicalize()` mirrors a ModSecurity transformation chain [5]. It applies
these steps in order:

1. Iterative percent-decoding, up to three rounds. `+` is deliberately not
   treated as a space.
2. HTML-entity decoding.
3. JavaScript `\uXXXX` and `\xXX` decoding.
4. A second percent-decoding pass.
5. Unwrapping of MySQL `/*!NNNNN … */`, so that the hidden keyword is kept.
6. Replacement of remaining `/* … */` with a space.
7. Whitespace compression.
8. Lower-casing.

The output is a Python string and is never executed. As an ablation, every
model in E1 and E3 is also trained and tested on *raw* input, which is only
lower-cased.

### 3.4 Features and models

**Character n-gram TF-IDF.** We use character 1–4-grams with
`analyzer="char"`, so n-grams can cross spaces (`' or`, `1=1-`). Other
settings: sublinear term frequency, `min_df = 2`, at most 50,000 features and
L2 normalisation.

**Hand-crafted features.** There are 28 features, grouped as follows:

* *size*: raw and canonical length, token count;
* *composition*: special-character, digit, raw upper-case and whitespace
  ratios; Shannon entropy [24];
* *encoding evidence* (computed on the raw string): percent-encoding ratio,
  HTML-entity count, JS-escape count;
* *metacharacters*: counts of `'`, `"`, `<>`, `()`, `=`, `;` and SQL comment
  markers;
* *lexicons*: counts of 35 SQL keywords and 27 XSS keywords;
* *structure*: regex flags for an HTML tag, an event-handler assignment, a
  `javascript:`/`vbscript:` URI, a tautology (`n=n`, `'x'='x'`, `or true`)
  and `UNION [ALL] SELECT`; the number of function-call patterns, the longest
  run of non-alphanumeric characters, and odd quote parity.

**Classifiers.** All classifiers are scikit-learn [25] pipelines from raw
string to label. Every learned transformation is therefore fitted on training
data only.

| Key | Representation | Classifier |
|---|---|---|
| `tfidf_nb` | TF-IDF | Multinomial Naive Bayes, α = 0.1 (baseline) |
| `tfidf_lr` | TF-IDF | multinomial logistic regression, C = 10 |
| `tfidf_svm` | TF-IDF | linear SVM (LIBLINEAR [26], one-vs-rest), C = 1 |
| `hand_rf` | hand-crafted | random forest [27], 300 trees |
| `hybrid_lr` | TF-IDF ⊕ hand-crafted (signed log1p, then standardised) | logistic regression, C = 10 |

Hyperparameters were fixed *a priori* to common defaults and were not tuned
on any test set.

**Binary view.** The models are trained as 3-class classifiers. For binary
detection we merge SQLi and XSS into *malicious*. The continuous score is
1 − P(benign) for probabilistic models. For the SVM it is
max(margin_SQLi, margin_XSS) − margin_benign.

**Sliding-window detector.** The detector is a defence against padding and is
applied at inference time only. The canonical string is cut into windows of
width *w* with stride ⌊*w*/3⌋. The fitted model scores each window, and the
input inherits the label of its most malicious window. The threshold is
P(benign) < 0.5, or a positive margin for the SVM. A string no longer than
*w* is scored as a whole.

### 3.5 Experimental protocols

* **E1 – in-distribution.** Stratified 80/20 split: 16,000 training and 4,000
  test strings. Every model is run with both canonical and raw input.
* **E2 – cross-validation.** Stratified 5-fold CV on all 20,000 rows with
  canonical input, to estimate the variance of E1.
* **E3 – obfuscation shift.** Benign rows are split 80/20 at random.
  * *Training set*: the benign 80 % plus every malicious row whose obfuscation
    is `none` or one of the five seen techniques (15,143 rows).
  * *Test set*: the benign 20 % plus every malicious row that uses one of the
    six held-out techniques (4,857 rows, of which 2,857 are malicious).
  * The model never sees a held-out technique during training. Both input
    modes are evaluated.
* **E4 – sliding window.** The canonical models from E1 and E3 are re-scored
  with *w* ∈ {32, 48, 64, 96}. E1 measures the false-positive cost of
  windowing, and E3 measures the benefit.
* **E5 – leave-one-family-out.** For each of the 14 attack families, every
  model is retrained on the E1 training split with that family removed. It
  is then tested on *all* rows of the held-out family and on the E1 benign
  test rows (to measure false positives).

### 3.6 Metrics

**3-class metrics.** We report accuracy, macro-averaged precision, recall and
F1, and per-class F1. Macro averaging weights the minority attack classes
equally with the benign class.

**Binary metrics.** We report precision, recall (detection rate), F1, the
false-positive rate (FPR) on benign inputs, the FPR restricted to the
hard-benign subset, and ROC-AUC. Per-obfuscation and per-family results use
the *detection rate*: the share of malicious strings predicted as either
attack class. We also report confusion matrices.

**Reproducibility.** Every random process is seeded (seed 42). Library
versions are pinned in `requirements.txt`. The pipeline
(`python -m src.run_experiments`) regenerates the dataset, verifies it
against the committed CSV, and rewrites every table and figure.

## 4 Experiments and Results

All numbers in this section come from `results/*.csv`. The tables are
rendered by `src/make_tables.py` into `results/tables.md`.

### 4.1 E1/E2 – In-distribution performance

| Input | Model | Accuracy | Macro-P | Macro-R | Macro-F1 | F1 benign | F1 SQLi | F1 XSS | Bin. recall | FPR | Hard-benign FPR | ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| canonical | TF-IDF + Naive Bayes | 0.9825 | 0.9876 | 0.9777 | 0.9824 | 0.9828 | 0.9787 | 0.9858 | 0.9680 | 0.30 % | 0.70 % | 0.9971 |
| canonical | TF-IDF + LogReg | 0.9995 | 0.9997 | 0.9993 | 0.9995 | 0.9995 | 0.9990 | 1.0000 | 0.9990 | 0.00 % | 0.00 % | 1.0000 |
| canonical | TF-IDF + Linear SVM | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.00 % | 0.00 % | 1.0000 |
| canonical | Hand-crafted + Random Forest | 0.9975 | 0.9978 | 0.9972 | 0.9975 | 0.9975 | 0.9950 | 1.0000 | 0.9965 | 0.15 % | 0.42 % | 1.0000 |
| canonical | Hybrid + LogReg | 0.9970 | 0.9972 | 0.9967 | 0.9969 | 0.9972 | 0.9940 | 0.9995 | 0.9965 | 0.20 % | 0.56 % | 1.0000 |
| raw | TF-IDF + Naive Bayes | 0.9725 | 0.9760 | 0.9688 | 0.9723 | 0.9730 | 0.9726 | 0.9713 | 0.9620 | 1.65 % | 2.39 % | 0.9953 |
| raw | TF-IDF + LogReg | 0.9988 | 0.9990 | 0.9985 | 0.9988 | 0.9988 | 0.9975 | 1.0000 | 0.9980 | 0.05 % | 0.14 % | 1.0000 |
| raw | TF-IDF + Linear SVM | 0.9995 | 0.9995 | 0.9995 | 0.9995 | 0.9995 | 0.9990 | 1.0000 | 0.9995 | 0.05 % | 0.14 % | 1.0000 |
| raw | Hand-crafted + Random Forest | 0.9902 | 0.9913 | 0.9888 | 0.9900 | 0.9908 | 0.9834 | 0.9960 | 0.9870 | 0.55 % | 1.55 % | 0.9997 |
| raw | Hybrid + LogReg | 0.9970 | 0.9972 | 0.9965 | 0.9968 | 0.9975 | 0.9955 | 0.9975 | 0.9965 | 0.15 % | 0.42 % | 0.9999 |

*Table 2. E1: in-distribution test split (n = 4,000: 2,000 benign, 1,000 SQLi,
1,000 XSS). FPR is the share of benign strings flagged as malicious. The
hard-benign FPR is computed on the 711 hard-benign test strings.*

| Model | Accuracy | Macro-F1 | Binary F1 | FPR | ROC-AUC |
|---|---|---|---|---|---|
| TF-IDF + Naive Bayes | 0.9875 ± 0.0024 | 0.9875 ± 0.0025 | 0.9874 ± 0.0025 | 0.38 ± 0.14 % | 0.99815 |
| TF-IDF + LogReg | 0.9992 ± 0.0005 | 0.9992 ± 0.0005 | 0.9992 ± 0.0005 | 0.01 ± 0.02 % | 1.00000 |
| TF-IDF + Linear SVM | 0.9998 ± 0.0002 | 0.9998 ± 0.0002 | 0.9998 ± 0.0002 | 0.01 ± 0.02 % | 1.00000 |
| Hand-crafted + Random Forest | 0.9977 ± 0.0010 | 0.9977 ± 0.0010 | 0.9978 ± 0.0010 | 0.12 ± 0.04 % | 0.99997 |
| Hybrid + LogReg | 0.9982 ± 0.0009 | 0.9982 ± 0.0009 | 0.9983 ± 0.0008 | 0.13 ± 0.14 % | 0.99997 |

*Table 3. E2: 5-fold stratified cross-validation with canonical input
(mean ± standard deviation over folds).*

![Figure 2](../results/figures/confusion_matrices_all.png)

*Figure 2. E1 confusion matrices with canonical input (counts and
row-normalised shares).*

**The random split is close to its ceiling.** With canonical input, the
TF-IDF + Linear SVM model classifies all 4,000 test strings correctly
(Table 2, Figure 2). TF-IDF + LogReg misclassifies 2 strings. Cross-validation
confirms that this is not luck of the split: the SVM's macro-F1 is
0.9998 ± 0.0002, and binary ROC-AUC is 1.0000 for every model except Naive
Bayes (Table 3).

**The weaker models fail in instructive ways.** Naive Bayes is the weakest
model (macro-F1 0.9824). Its errors are almost all attacks predicted as benign
(36 SQLi and 28 XSS), because its independence assumption over-counts the
many benign-looking n-grams. The hand-crafted Random Forest and the hybrid
model each make 10–12 errors, of three kinds:

* *Symbol-heavy passwords flagged as SQLi*, e.g. `UR{VRan9#zBsQ'?#J^`, which
  contains a quote, a `#` comment marker and high entropy.
* *A benign free-text string made of SQL words* (`Join from image good order
  issue join table price help load union select :)`), flagged by the hybrid
  model.
* *Minimal attacks predicted as benign*, e.g. `q=router&item=admin'` and
  `charger973 AND 3=46`. These are borderline cases: a lone trailing quote,
  or a false comparison, carries very little evidence.

**Canonicalisation also helps in-distribution, mainly by reducing false
positives.** Compared with raw input:

* Naive Bayes FPR falls from 1.65 % to 0.30 %.
* Random Forest hard-benign FPR falls from 1.55 % to 0.42 %.
* Random Forest macro-F1 rises from 0.9900 to 0.9975.

The TF-IDF LogReg and SVM models are nearly unaffected (difference
≤ 0.0007), because both raw and decoded forms appear in training.

### 4.2 E3 – Shift to unseen obfuscations

| Input | Model | Accuracy | Macro-P | Macro-R | Macro-F1 | F1 benign | F1 SQLi | F1 XSS | Bin. recall | FPR | Hard-benign FPR | ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| canonical | TF-IDF + Naive Bayes | 0.8948 | 0.9264 | 0.8817 | 0.8958 | 0.8904 | 0.8980 | 0.8990 | 0.8316 | 0.45 % | 1.45 % | 0.8370 |
| canonical | TF-IDF + LogReg | 0.8820 | 0.9258 | 0.8674 | 0.8845 | 0.8747 | 0.9051 | 0.8737 | 0.7994 | 0.00 % | 0.00 % | 0.9052 |
| canonical | TF-IDF + Linear SVM | 0.8888 | 0.9291 | 0.8748 | 0.8912 | 0.8811 | 0.9073 | 0.8852 | 0.8110 | 0.00 % | 0.00 % | 0.9673 |
| canonical | Hand-crafted + Random Forest | 0.9386 | 0.9566 | 0.9341 | 0.9410 | 0.9306 | 0.9898 | 0.9025 | 0.8960 | 0.05 % | 0.16 % | 0.9998 |
| canonical | Hybrid + LogReg | 0.9634 | 0.9723 | 0.9590 | 0.9645 | 0.9573 | 0.9719 | 0.9644 | 0.9391 | 0.20 % | 0.64 % | 0.9903 |
| raw | TF-IDF + Naive Bayes | 0.8355 | 0.8907 | 0.8164 | 0.8365 | 0.8312 | 0.8334 | 0.8448 | 0.7403 | 2.50 % | 4.02 % | 0.8288 |
| raw | TF-IDF + LogReg | 0.7877 | 0.8866 | 0.7608 | 0.7856 | 0.7951 | 0.8032 | 0.7587 | 0.6391 | 0.00 % | 0.00 % | 0.8805 |
| raw | TF-IDF + Linear SVM | 0.7976 | 0.8902 | 0.7722 | 0.7965 | 0.8027 | 0.8192 | 0.7677 | 0.6559 | 0.00 % | 0.00 % | 0.9365 |
| raw | Hand-crafted + Random Forest | 0.8629 | 0.9144 | 0.8484 | 0.8654 | 0.8569 | 0.9152 | 0.8240 | 0.7704 | 0.45 % | 1.45 % | 0.9958 |
| raw | Hybrid + LogReg | 0.8666 | 0.9175 | 0.8475 | 0.8670 | 0.8608 | 0.8501 | 0.8902 | 0.7746 | 0.10 % | 0.32 % | 0.9713 |

*Table 4. E3: test set of 2,000 benign strings and 2,857 malicious strings
that use only the six held-out obfuscations.*

![Figure 3](../results/figures/model_comparison.png)

*Figure 3. Macro-F1 for raw input (grey circles) and canonical input (dark
diamonds). Left: E1. Right: E3. The x-axes differ: E1 spans 0.97–1.00 and
E3 spans 0.78–0.97.*

| Held-out obfuscation | n | TF-IDF + Naive Bayes | TF-IDF + LogReg | TF-IDF + Linear SVM | Hand-crafted + Random Forest | Hybrid + LogReg |
|---|---|---|---|---|---|---|
| double_url | 502 | 84.5 % → 100.0 % | 81.9 % → 99.0 % | 83.5 % → 99.8 % | 86.3 % → 97.6 % | 77.3 % → 99.2 % |
| whitespace_alt | 431 | 96.8 % → 100.0 % | 94.7 % → 99.8 % | 95.4 % → 100.0 % | 91.0 % → 99.8 % | 98.1 % → 100.0 % |
| versioned_comment | 448 | 100.0 % → 100.0 % | 100.0 % → 99.6 % | 100.0 % → 100.0 % | 99.8 % → 96.2 % | 97.8 % → 98.7 % |
| unicode_escape | 521 | 67.6 % → 100.0 % | 24.2 % → 100.0 % | 29.9 % → 100.0 % | 48.0 % → 99.8 % | 22.1 % → 99.8 % |
| char_encoding | 475 | 99.8 % → 99.8 % | 90.9 % → 81.9 % | 92.4 % → 87.4 % | 99.8 % → 99.8 % | 98.9 % → 99.2 % |
| benign_padding | 480 | 0.0 % → 0.0 % | 0.2 % → 0.2 % | 0.2 % → 0.2 % | 42.7 % → 44.8 % | 79.0 % → 66.9 % |

*Table 5. E3 detection rate per held-out obfuscation, raw → canonical input.*

![Figure 4](../results/figures/obfuscation_detection.png)

*Figure 4. E3 detection rate per held-out obfuscation for TF-IDF + Linear SVM,
the best model in E1.*

**The model that is perfect in E1 is not the best under shift.** On unseen
obfuscations every model loses accuracy (Table 4, Figure 3). The ranking also
changes:

* The Linear SVM, which was perfect in E1, drops to 0.8912 macro-F1.
* The hybrid model is best, at 0.9645.
* The hand-crafted Random Forest is second, at 0.9410.

False-positive rates stay low (≤ 0.45 % with canonical input), because the
benign test data is still in-distribution. The loss is almost entirely in
recall.

**Canonicalisation is the largest single factor.** It improves E3 macro-F1 for
every model by 0.059–0.099. For example, TF-IDF + LogReg rises from 0.7856 to
0.8845 and the hybrid model from 0.8670 to 0.9645. Table 5 shows where the
gain comes from:

* *Unicode escapes.* `'`-style escapes defeat raw TF-IDF (LogReg 24.2 %,
  SVM 29.9 % detected), because the escaped forms share almost no n-grams
  with training data. Canonical input lifts every model to 99.8–100 %.
* *Double URL encoding and alternative whitespace.* These go from 77–98 % to
  97.6–100 %.
* *MySQL versioned comments.* These are caught even on raw input, because
  the keywords inside `/*!…*/` stay visible.

**Two techniques survive canonicalisation.**

1. *Character-code encoding* (`CHAR(…)`, `0x…`, `String.fromCharCode(…)`).
   No WAF transformation can decode it without interpreting code.
   Canonicalisation even lowers TF-IDF detection for this technique (SVM
   92.4 % → 87.4 %). The missed strings are mostly XSS payloads in which
   `fromCharCode` replaces the only marker call, e.g.
   `<script>eval(String.fromCharCode(97,108,…))</script>`. The canonical
   n-gram models have learned `alert(`/`aler` as strong XSS evidence
   (Section 4.5), so when those n-grams disappear the evidence goes with them.
   The hand-crafted RF still detects 99.8 %, from structural features such as
   tags, parentheses and function calls.
2. *Benign padding.* Detection is 0.0–0.2 % for all three TF-IDF models,
   whatever the preprocessing. The payload is intact in the string, but
   2–5 surrounding review sentences dominate the L2-normalised TF-IDF
   vector. This is the n-gram analogue of the "good word" attack on spam
   filters [21]. The hand-crafted RF (44.8 %) and the hybrid (66.9 %) resist
   partly, because count features such as `n_sql_comment`, `n_parentheses`
   and the keyword counts are not normalised by length.

Padding accounts for nearly all of the TF-IDF loss in E3. Excluding the 480
padded strings, canonical SVM recall on the other five held-out techniques is
97.4 %.

**Ranking survives better than thresholding for the Random Forest.** Its E3
ROC-AUC is 0.9998, while its binary recall is only 0.896. The model still
ranks padded and encoded attacks above benign inputs, but their scores fall
below the 0.5 threshold. A threshold recalibrated on shifted data would
recover much of the loss; the TF-IDF models do not have this property
(AUC 0.84–0.97).

![Figure 5](../results/figures/confusion_matrix_shift.png)

*Figure 5. E3 confusion matrices for the Linear SVM with raw and canonical
input. Every error is an attack predicted as benign. Confusion between
SQLi and XSS does not occur.*

### 4.3 E4 – Sliding-window inference against padding

| Model | Window | E3 macro-F1 | E3 bin. recall | E3 padding detected | E3 FPR | E1 macro-F1 | E1 FPR |
|---|---|---|---|---|---|---|---|
| TF-IDF + Naive Bayes | whole | 0.8958 | 0.8316 | 0.0 % | 0.45 % | 0.9824 | 0.30 % |
| TF-IDF + Naive Bayes | 32 | 0.9877 | 0.9909 | 94.6 % | 0.95 % | 0.9943 | 1.15 % |
| TF-IDF + Naive Bayes | 48 | 0.9759 | 0.9709 | 82.7 % | 0.45 % | 0.9985 | 0.30 % |
| TF-IDF + Naive Bayes | 64 | 0.9627 | 0.9527 | 71.9 % | 0.45 % | 0.9982 | 0.30 % |
| TF-IDF + Naive Bayes | 96 | 0.9334 | 0.9031 | 42.5 % | 0.45 % | 0.9975 | 0.30 % |
| TF-IDF + LogReg | whole | 0.8845 | 0.7994 | 0.2 % | 0.00 % | 0.9995 | 0.00 % |
| TF-IDF + LogReg | 32 | 0.9921 | 0.9888 | 97.9 % | 0.35 % | 0.9990 | 0.10 % |
| TF-IDF + LogReg | 48 | 0.9828 | 0.9706 | 91.7 % | 0.10 % | 0.9995 | 0.00 % |
| TF-IDF + LogReg | 64 | 0.9678 | 0.9436 | 82.3 % | 0.00 % | 0.9995 | 0.00 % |
| TF-IDF + LogReg | 96 | 0.9330 | 0.8827 | 52.9 % | 0.00 % | 0.9995 | 0.00 % |
| TF-IDF + Linear SVM | whole | 0.8912 | 0.8110 | 0.2 % | 0.00 % | 1.0000 | 0.00 % |
| TF-IDF + Linear SVM | 32 | 0.9945 | 0.9927 | 99.2 % | 0.25 % | 1.0000 | 0.00 % |
| TF-IDF + Linear SVM | 48 | 0.9895 | 0.9815 | 96.2 % | 0.00 % | 1.0000 | 0.00 % |
| TF-IDF + Linear SVM | 64 | 0.9774 | 0.9605 | 88.3 % | 0.00 % | 1.0000 | 0.00 % |
| TF-IDF + Linear SVM | 96 | 0.9503 | 0.9132 | 65.8 % | 0.00 % | 1.0000 | 0.00 % |
| Hand-crafted + Random Forest | whole | 0.9410 | 0.8960 | 44.8 % | 0.05 % | 0.9975 | 0.15 % |
| Hand-crafted + Random Forest | 32 | 0.9406 | 0.9083 | 85.6 % | 0.35 % | 0.9474 | 0.45 % |
| Hand-crafted + Random Forest | 48 | 0.9322 | 0.8855 | 72.3 % | 0.30 % | 0.9584 | 0.35 % |
| Hand-crafted + Random Forest | 64 | 0.9280 | 0.8771 | 63.5 % | 0.30 % | 0.9649 | 0.35 % |
| Hand-crafted + Random Forest | 96 | 0.9271 | 0.8750 | 52.3 % | 0.30 % | 0.9731 | 0.35 % |
| Hybrid + LogReg | whole | 0.9645 | 0.9391 | 66.9 % | 0.20 % | 0.9969 | 0.20 % |
| Hybrid + LogReg | 32 | 0.9552 | 0.9353 | 91.7 % | 0.20 % | 0.9622 | 0.05 % |
| Hybrid + LogReg | 48 | 0.9688 | 0.9499 | 90.6 % | 0.15 % | 0.9781 | 0.10 % |
| Hybrid + LogReg | 64 | 0.9718 | 0.9538 | 89.4 % | 0.15 % | 0.9843 | 0.10 % |
| Hybrid + LogReg | 96 | 0.9700 | 0.9503 | 84.2 % | 0.15 % | 0.9862 | 0.10 % |

*Table 6. E4: whole-string scoring compared with sliding-window max-pooling
(stride = ⌊w/3⌋). "Padding detected" is the detection rate on the 480 padded
strings in E3. The E1 columns measure the cost of windowing on in-distribution
data.*

![Figure 6](../results/figures/window_sweep.png)

*Figure 6. Left: E3 macro-F1 against window width. Right: E1 benign FPR
against window width.*

**Windowing restores TF-IDF detection of padded payloads.** Max-pooling the
classifier over windows recovers padded payloads that whole-string TF-IDF
misses completely (Table 6, Figure 6). At *w* = 48:

* Linear SVM padding detection rises from 0.2 % to 96.2 %.
* Linear SVM E3 macro-F1 rises from 0.8912 to 0.9895, with 0.00 % FPR in both
  E1 and E3.
* LogReg rises from 0.8845 to 0.9828.

This is the best E3 result of any configuration we tested at a 0.00 % E1 FPR.

**Smaller windows trade false positives for recall.** At *w* = 32 the SVM
reaches 0.9945 E3 macro-F1, but its E3 FPR becomes 0.25 %. Naive Bayes reaches
an E1 FPR of 1.15 % at *w* = 32, because short fragments of passwords and
code-like questions look like attacks out of context.

**Windowing is not universal.** It *hurts* models that depend on whole-string
statistics:

* The hand-crafted RF falls from 0.9975 to 0.9584 E1 macro-F1 at *w* = 48.
* The hybrid falls from 0.9969 to 0.9781.

Features such as length, token count and ratios take values on 48-character
fragments that the models never saw in training. These models would need to
be trained on windows as well.

**Caveat on the choice of *w*.** The sweep is evaluated on the test sets
themselves. We did not tune *w* on separate validation data, so the
highlighted *w* = 48 should be read as a sensitivity analysis rather than an
unbiased estimate. Every width from 32 to 64 gives a large improvement for all
TF-IDF models.

### 4.4 E5 – Unseen attack families

| Held-out family | n | TF-IDF + Naive Bayes | TF-IDF + LogReg | TF-IDF + Linear SVM | Hand-crafted + Random Forest | Hybrid + LogReg |
|---|---|---|---|---|---|---|
| sqli / boolean_blind | 767 | 96.7 % | 92.0 % | 91.7 % | 99.3 % | 99.5 % |
| sqli / error_based | 681 | 97.7 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % |
| sqli / order_by | 700 | 70.9 % | 38.9 % | 47.9 % | 97.1 % | 95.7 % |
| sqli / stacked | 709 | 89.4 % | 40.1 % | 40.2 % | 93.8 % | 95.1 % |
| sqli / tautology | 678 | 88.6 % | 53.4 % | 57.1 % | 80.2 % | 82.3 % |
| sqli / time_blind | 753 | 72.5 % | 83.8 % | 89.1 % | 98.7 % | 97.7 % |
| sqli / union | 712 | 87.2 % | 59.7 % | 66.7 % | 99.4 % | 85.1 % |
| xss / attr_breakout | 767 | 95.4 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % |
| xss / data_uri | 627 | 77.4 % | 61.9 % | 62.4 % | 85.6 % | 65.4 % |
| xss / event_handler | 829 | 95.5 % | 97.1 % | 99.5 % | 99.4 % | 100.0 % |
| xss / js_uri | 707 | 96.2 % | 99.6 % | 99.7 % | 99.2 % | 99.3 % |
| xss / script_tag | 682 | 94.1 % | 99.1 % | 99.4 % | 86.7 % | 99.6 % |
| xss / style_based | 693 | 94.8 % | 99.0 % | 99.3 % | 100.0 % | 100.0 % |
| xss / template_dom | 695 | 96.4 % | 92.5 % | 92.8 % | 95.7 % | 85.8 % |
| **mean detection** |  | **89.5 %** | **79.8 %** | **81.8 %** | **95.4 %** | **93.2 %** |
| mean benign FPR |  | 0.26 % | 0.00 % | 0.00 % | 0.16 % | 0.20 % |

*Table 7. E5: share of a held-out family detected (predicted as either attack
class) when that family is absent from training. Benign FPR is measured on the
2,000 E1 benign test strings and averaged over the 14 runs.*

![Figure 7](../results/figures/family_holdout.png)

*Figure 7. E5 detection rate per held-out family for three representative
models.*

**Learned n-grams transfer worse than expert features.** When an entire
attack family is missing from training, the models that generalise best are
those built on expert knowledge (Table 7, Figure 7):

* Hand-crafted RF: 95.4 % mean detection.
* Hybrid: 93.2 %.
* TF-IDF + Linear SVM: 81.8 %.
* TF-IDF + LogReg: 79.8 %.

The discriminative TF-IDF models learn the n-grams *of the families they
saw*. Unseen families with little lexical overlap are largely missed by both
discriminative TF-IDF models:

| Unseen family | TF-IDF LogReg | TF-IDF SVM | Hand-crafted RF |
|---|---|---|---|
| stacked queries (`; DROP TABLE …`) | 40.1 % | 40.2 % | 93.8 % |
| `ORDER BY` probing | 38.9 % | 47.9 % | 97.1 % |
| tautologies | 53.4 % | 57.1 % | 80.2 % |
| UNION | 59.7 % | 66.7 % | 99.4 % |

The RF's keyword lexicon and its comment, quote and semicolon counts encode
knowledge of SQL syntax *as a whole*, so they transfer to families it has not
seen.

**Some families are hard for every model.**

* *`data:` URIs* (XSS): 62–86 %. The `base64` payload hides the markup.
* *Tautologies* (SQLi): 53–89 %. Short forms such as `" or ""="` and
  `admin'--` contain very little syntax.

The generative Naive Bayes model is more robust than its discriminative
TF-IDF counterparts (89.5 %). It is also the noisiest in-distribution.

### 4.5 What the models learn

| Class | Top n-grams (LogReg weight) |
|---|---|
| benign | `e␣` (2.9), `h` (2.4), `t␣` (2.0), `k` (1.9), `e` (1.8), `i` (1.6), `w` (1.6), `n␣` (1.4) |
| sqli | `'` (6.8), `--` (5.1), `-` (4.6), `and` (4.2), `␣or␣` (4.2), `;␣` (4.1), `or␣` (4.1), `;` (4.0) |
| xss | `<` (4.9), `>` (4.7), `(` (3.8), `t(` (3.4), `)` (3.1), `␣on` (2.7), `aler` (2.7), `lert` (2.7) |

*Table 8. Character n-grams with the largest LogReg coefficients per class
(canonical input; ␣ = space).*

![Figure 8](../results/figures/top_ngrams.png)

*Figure 8. Top 15 n-grams per class from the TF-IDF + LogReg model.*

| Feature | Permutation Δ macro-F1 | Gini importance |
|---|---|---|
| upper_ratio_raw | 0.0342 ± 0.0027 | 0.0861 |
| sql_keyword_count | 0.0303 ± 0.0027 | 0.1194 |
| xss_keyword_count | 0.0239 ± 0.0017 | 0.1446 |
| n_sql_comment | 0.0078 ± 0.0013 | 0.0585 |
| n_parentheses | 0.0058 ± 0.0003 | 0.0861 |
| n_angle_brackets | 0.0023 ± 0.0004 | 0.0716 |
| whitespace_ratio | 0.0015 ± 0.0003 | 0.0257 |
| n_tokens | 0.0015 ± 0.0003 | 0.0250 |
| max_nonalnum_run | 0.0014 ± 0.0003 | 0.0508 |
| n_single_quote | 0.0012 ± 0.0006 | 0.0374 |

*Table 9. The ten most important hand-crafted features: permutation
importance (drop in macro-F1 on the E1 test set, 5 repeats) and impurity
(Gini) importance.*

![Figure 9](../results/figures/feature_importance.png)

*Figure 9. Permutation importance of all 28 hand-crafted features (Random
Forest, E1 test set).*

**The n-gram model learns attack syntax** (Table 8, Figure 8):

* *SQLi*: the single quote (weight 6.8), `--`, `;␣`, and `␣or␣`/`and` as
  separate tokens.
* *XSS*: angle brackets, `(`, `t(` (as in `alert(`), `␣on` (event handlers)
  and the fragments `aler`/`lert`.

The benign class has no positive syntax of its own. Its top features are
common English letter n-grams such as `e␣` and `t␣`. In effect, *benign*
means "prose without attack syntax". This explains why padding works: it
adds strong benign evidence.

**The Random Forest ranks a shortcut first** (Table 9, Figure 9). The three
most important features are:

1. `upper_ratio_raw` (Δ macro-F1 0.034);
2. the SQL keyword count (0.030);
3. the XSS keyword count (0.024).

The first is a **shortcut** [22]. Our SQLi templates, like most published
payload lists, write keywords in upper case, while benign text is mostly
lower or sentence case. Upper-case density is therefore predictive in this
dataset, but an attacker can remove it at no cost: SQL keywords are
case-insensitive. We kept the feature so that the shortcut stays visible,
rather than hiding it, and the paper reports it as a finding. Two facts limit
its effect:

* The `case` obfuscation (random case) is present in training.
* Canonical-only features carry most of the Gini importance: the XSS keyword
  count (0.145) and the SQL keyword count (0.119) both outrank
  `upper_ratio_raw` (0.086).

Permutation importances are small overall, at most 0.034. Many features are
redundant (for example the parenthesis count and the function-call count), so
permuting any one of them costs little.

**Cost.** Every model is cheap. On a 4-core container, training on 16,000
strings takes 1–7 s, and prediction takes 63–108 µs per string, including
canonicalisation and feature extraction (`results/timing.csv`; these times are
machine-dependent). Sliding-window scoring multiplies the cost by the number
of windows, which is about 3·len/*w*.

## 5 Discussion and Limitations

### 5.1 What a practitioner should take away

1. **Do not trust a random split of payload lists.** Every one of our
   TF-IDF models scored ≥ 0.98 macro-F1 in E1, and the best scored 1.0000.
   The same models lost 11 percentage points of accuracy when the encodings
   changed, and up to 20 percentage points of detection when a whole attack
   family was absent. Template-generated (and scraped) payload corpora contain
   many near-duplicates, and a random split measures memorisation [11]. A
   payload-detection evaluation should report at least one shift protocol,
   such as held-out encodings, held-out families or held-out sources.
2. **Normalise before learning.** WAF-style canonicalisation costs almost
   nothing and was worth +0.06 to +0.10 macro-F1 under shift. It also reduced
   in-distribution false positives. Decoding is better done by deterministic
   code than learned from data, because encodings form an open-ended space an
   attacker can always extend.
3. **Combine learned and expert features.** The TF-IDF SVM is the best
   in-distribution model. Hand-crafted features generalise better to new
   attack families. The hybrid is the most robust whole-string model under
   encoding shift. These representations are complementary.
4. **Detect locally, not globally.** Any representation normalised over the
   whole input is open to dilution. Max-pooling over windows is a simple fix
   that restored 96 % padding detection with no FPR cost for the linear
   TF-IDF models. A production design could do two things: train on windows,
   and add a whole-string model, so that statistics features stay
   in-distribution.

### 5.2 Evasion

Our attacker is *non-adaptive*. Every obfuscation is a fixed transform chosen
without access to the model, and even so several transforms succeeded:

* **Padding.** Up to 99.8 % evasion against whole-string TF-IDF.
* **Character-code construction.** About 13 % evasion against the SVM.
  Canonicalisation cannot undo it, because doing so would require evaluating
  code.
* **Novel syntax.** An unseen family evades the TF-IDF SVM between 0 % and
  60 % of the time, depending on the family.

An *adaptive* attacker with query access would do much better. Such an
attacker can search the space of semantics-preserving mutations, as
WAF-A-MoLE does against ML WAFs [16] and as Appelt et al. did against
signature WAFs [8, 15]. Typical mutations:

* swapping in equivalent keywords (`||` for `OR`);
* adding junk comments;
* changing whitespace and case;
* building strings with `CONCAT`/`CHAR`;
* using unusual HTML tags and events.

Linear models over n-grams are particularly exposed to this kind of search:
the attacker only needs to remove high-weight n-grams (`'`, `--`, `alert(`)
or add negative-weight ones. Our window defence closes the padding channel,
but it gives no guarantee against mutation search. Real robustness would
need some combination of:

* adversarial training on mutated payloads;
* ensembles mixing learned and grammar-based detectors (for example
  libinjection-style tokenisation [14]);
* parsing the input in the context it will be used in (SQL grammar or HTML
  tokeniser) rather than as a flat string.

A detector at the WAF position also suffers from *impedance mismatch*. It
canonicalises differently from the backend, which may decode `+`, overlong
UTF-8 or other encodings. One concrete example appears in our own errors: our
canonicaliser keeps `+` as a literal, which is right for path segments and
JSON. Strings such as `redirect=UNION+SELECT%20username…` were then missed by
the hybrid and RF models, and a backend that decodes form data would see
spaces there.

### 5.3 Limitations

* **The data is synthetic.** Templates give clean labels and full control,
  which is what made the shift protocols possible. However, the benign
  distribution is far narrower than real traffic: we have no real user
  sessions, API payloads, rich-text editors, code-hosting or SQL-tutorial
  sites. The attack distribution is limited to our 95 templates. Absolute
  numbers will not transfer to production. The *relative* findings are more
  likely to: a shift collapses random-split scores, canonicalisation helps,
  padding dilutes TF-IDF, and expert features transfer across families. All
  of these should be confirmed on real data, such as WAF logs or public
  corpora like CSIC 2010.
* **There are labelling artefacts.** Labels are by construction, not by
  exploitability. Some "malicious" strings are near-benign in isolation, such
  as `admin'` with an empty comment or `charger973 AND 3=46`. Some "benign"
  strings are SQL fragments typed into a search box, which a strict policy
  might block. A real deployment must decide where that boundary lies.
* **The upper-case shortcut** (Section 4.5) shows that synthetic generators
  carry the stylistic habits of payload lists. Other, less visible shortcuts
  may exist; one candidate is the narrow benign vocabulary.
* **The base rate is unrealistic.** Benign strings are 50 % of our data. In
  production, attacks are a tiny fraction of requests. An FPR of 0.2 %, with
  recall 0.94 and one attack per 10,000 requests, gives an alert precision of
  only about 4.5 %. A measured FPR of "0.00 %" on 2,000 benign test strings
  only bounds the true FPR below about 0.15 % (rule of three, 95 %
  confidence). Much larger benign test sets are needed to certify
  production-grade FPRs.
* **The scope is narrow.** We cover single-parameter, first-order SQLi and
  reflected-style XSS strings. We do not cover:
  * stored or DOM XSS that never crosses the WAF;
  * second-order SQLi;
  * NoSQL, LDAP, command or template injection beyond the few template
    payloads;
  * request-level context, such as the endpoint, the parameter's expected
    type and user history, which real systems use.
* **Models and tuning were limited.** Hyperparameters were fixed defaults,
  and we tested no deep models. Sequence models (CNN/LSTM/transformers [10])
  might reduce the family gap, but they share the dilution and adaptive
  evasion concerns above.

### 5.4 Ethics and safety

All payloads are inert text, built from widely published, textbook patterns.
They use harmless marker functions and reserved domain names. No payload was
executed, rendered or sent to any system during this work. The artefact is
meant for building and evaluating *defences*. It contains no novel bypass
technique: every evasion we evaluate is publicly documented [7, 8, 16, 21].

## 6 Conclusion

We built a reproducible benchmark for ML-based detection of SQLi and XSS
payloads. It is synthetic, fully labelled and inert, and it covers 14 attack
families, 11 obfuscations and a hard-benign subset. On it we compared
character n-gram TF-IDF, hand-crafted features and a hybrid of both, across
five protocols.

The random-split evaluation is saturated, with a Linear SVM at 1.0000
macro-F1, and it hides the differences that matter:

* Under unseen encodings, canonicalisation is worth up to 0.10 macro-F1, and
  the hybrid model leads (0.9645).
* Under benign padding, whole-string TF-IDF detection collapses to at most 0.2 %.
  Sliding-window max-pooling restores it to 96 % at no false-positive cost,
  giving the best overall shift result (0.9895).
* Under unseen attack families, hand-crafted features detect 95.4 %, against
  81.8 % for the best TF-IDF model.

We also found and reported a dataset shortcut: upper-case keyword density.

Three directions follow:

* validating on real traffic;
* replacing the fixed obfuscation set with an adaptive, query-based attacker;
* training window-aware hybrid models, so that the encoding robustness of
  canonicalisation, the family robustness of expert features and the
  dilution robustness of local scoring can be combined in a single detector.

## References

[1] OWASP Foundation. *OWASP Top 10:2021 — A03:2021 Injection.* https://owasp.org/Top10/A03_2021-Injection/, 2021.

[2] W. G. J. Halfond, J. Viegas, and A. Orso. "A Classification of SQL-Injection Attacks and Countermeasures." In *Proc. IEEE International Symposium on Secure Software Engineering (ISSSE)*, 2006.

[3] J. Clarke (ed.). *SQL Injection Attacks and Defense*, 2nd ed. Syngress, 2012.

[4] OWASP Foundation. "Cross Site Scripting (XSS)." *OWASP Community Pages.* https://owasp.org/www-community/attacks/xss/

[5] I. Ristić. *ModSecurity Handbook*, 2nd ed. Feisty Duck, 2017. (Transformation functions: `t:urlDecodeUni`, `t:htmlEntityDecode`, `t:jsDecode`, `t:replaceComments`, `t:compressWhitespace`, `t:lowercase`.)

[6] OWASP Foundation. *OWASP ModSecurity Core Rule Set (CRS).* https://coreruleset.org/

[7] OWASP Foundation. *XSS Filter Evasion Cheat Sheet.* OWASP Cheat Sheet Series. https://cheatsheetseries.owasp.org/cheatsheets/XSS_Filter_Evasion_Cheat_Sheet.html

[8] D. Appelt, C. D. Nguyen, L. C. Briand, and N. Alshahwan. "Automated Testing for SQL Injection Vulnerabilities: An Input Mutation Approach." In *Proc. International Symposium on Software Testing and Analysis (ISSTA)*, 2014.

[9] D. Kar, S. Panigrahi, and S. Sundararajan. "SQLiGoT: Detecting SQL Injection Attacks Using Graph of Tokens and SVM." *Computers & Security*, 60, 2016.

[10] Y. Fang, Y. Li, L. Liu, and C. Huang. "DeepXSS: Cross Site Scripting Detection Based on Deep Learning." In *Proc. International Conference on Computing and Artificial Intelligence (ICCAI)*, 2018.

[11] D. Arp, E. Quiring, F. Pendlebury, A. Warnecke, F. Pierazzi, C. Wressnegger, L. Cavallaro, and K. Rieck. "Dos and Don'ts of Machine Learning in Computer Security." In *Proc. USENIX Security Symposium*, 2022.

[12] M. Heiderich, J. Schwenk, T. Frosch, J. Magazinius, and E. Z. Yang. "mXSS Attacks: Attacking Well-Secured Web-Applications by Using innerHTML Mutations." In *Proc. ACM Conference on Computer and Communications Security (CCS)*, 2013.

[13] T. Berners-Lee, R. Fielding, and L. Masinter. "Uniform Resource Identifier (URI): Generic Syntax." RFC 3986, IETF, 2005.

[14] N. Galbreath. *libinjection: SQL / SQLi tokenizer, parser and analyzer.* https://github.com/client9/libinjection, 2012–.

[15] D. Appelt, C. D. Nguyen, A. Panichella, and L. C. Briand. "A Machine-Learning-Driven Evolutionary Approach for Testing Web Application Firewalls." *IEEE Transactions on Reliability*, 67(3), 2018.

[16] L. Demetrio, A. Valenza, G. Costa, and G. Lagorio. "WAF-A-MoLE: Evading Web Application Firewalls through Adversarial Machine Learning." In *Proc. ACM Symposium on Applied Computing (SAC)*, 2020.

[17] C. Kruegel and G. Vigna. "Anomaly Detection of Web-based Attacks." In *Proc. ACM Conference on Computer and Communications Security (CCS)*, 2003.

[18] P. Likarish, E. Jung, and I. Jo. "Obfuscated Malicious JavaScript Detection Using Classification Techniques." In *Proc. International Conference on Malicious and Unwanted Software (MALWARE)*, 2009.

[19] R. Sommer and V. Paxson. "Outside the Closed World: On Using Machine Learning for Network Intrusion Detection." In *Proc. IEEE Symposium on Security and Privacy*, 2010.

[20] B. Biggio and F. Roli. "Wild Patterns: Ten Years After the Rise of Adversarial Machine Learning." *Pattern Recognition*, 84, 2018.

[21] D. Lowd and C. Meek. "Good Word Attacks on Statistical Spam Filters." In *Proc. Conference on Email and Anti-Spam (CEAS)*, 2005.

[22] R. Geirhos, J.-H. Jacobsen, C. Michaelis, R. Zemel, W. Brendel, M. Bethge, and F. A. Wichmann. "Shortcut Learning in Deep Neural Networks." *Nature Machine Intelligence*, 2(11), 2020.

[23] D. Eastlake 3rd and A. Panitz. "Reserved Top Level DNS Names." RFC 2606, IETF, 1999.

[24] C. E. Shannon. "A Mathematical Theory of Communication." *Bell System Technical Journal*, 27(3), 1948.

[25] F. Pedregosa et al. "Scikit-learn: Machine Learning in Python." *Journal of Machine Learning Research*, 12, 2011.

[26] R.-E. Fan, K.-W. Chang, C.-J. Hsieh, X.-R. Wang, and C.-J. Lin. "LIBLINEAR: A Library for Large Linear Classification." *Journal of Machine Learning Research*, 9, 2008.

[27] L. Breiman. "Random Forests." *Machine Learning*, 45(1), 2001.
