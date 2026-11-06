# EE6483 Mini Project Option 1

This project classifies product reviews as negative (0) or positive (1). The reference baseline adapts a standard movie-review sentiment pipeline to the course data using word-unigram TF-IDF and logistic regression. The final model combines word and character TF-IDF with class-weighted logistic regression. A compact SVD and MLP pipeline is included as a neural comparison.

## Results

These results come from the same 1,453 labeled reviews held out from development. The official test set has 1,851 reviews without labels.

| Model | Accuracy | Macro F1 | Negative recall |
|---|---:|---:|---:|
| Always positive | 85.34% | 46.05% | 0.00% |
| Unigram logistic baseline | 88.92% | 67.67% | 26.76% |
| SVD and MLP | 89.81% | 80.47% | 70.42% |
| Weighted word and character logistic | 91.74% | 83.62% | 72.77% |

Before training, normalized text matching identified 87 review groups shared across the course train and test files. The experiment excludes 136 matching labeled rows, retains all 1,851 test rows, and keeps duplicates within each internal split. It selects models on development macro F1. The final classifier is refitted on all 7,265 eligible labeled reviews for test inference.

## Reproduce the experiment

Get `train.json`, `test.json` and the original sample `submission.csv` from the [NTULearn Option 1 attachments](https://ntulearn.ntu.edu.sg/ultra/courses/_2706627_1/document/_5882846_1?view=content&state=view). Save them in `data/` and rename the original sample `sample_submission.csv`. The sample supplies column names and row order. Its predictions are placeholders.

Use Python 3.11 with the pinned dependencies. The run uses CPU and seed 6483.

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python experiment.py \
  --train data/train.json \
  --test data/test.json \
  --sample data/sample_submission.csv \
  --output rerun
```

The script validates exact test row order, fits the feature vocabulary and model on the relevant training split, and writes `submission.csv`, `metrics.json`, `audit_predictions.csv` and `test_scores.csv`. A repeat run matched the submitted prediction CSV and held-out prediction CSV byte for byte in the recorded environment. Feature preparation times vary between runs.

## Files

- [Report](EE6483_Option1_Report.docx) covers the literature, method, evaluation, error cases, hotel transfer and noisy labels. Group details are blank for the team to complete.
- [submission.csv](submission.csv) contains 1,851 course test predictions with the official header and original review order.
- [metrics.json](metrics.json) records every development setting, resource measurements, holdout metrics and dataset checksums.
- `audit_predictions.csv` and `test_scores.csv` record the original zero-based row indices and model scores.
