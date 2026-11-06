"""Train and evaluate EE6483 Option 1 product-review sentiment models."""

import argparse
import csv
import hashlib
import html
import json
import platform
import re
import unicodedata
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
import sklearn
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

SEED = 6483
EPOCHS = 30
BREAK = re.compile(r"</?(?:br|p)\b[^>]*>", re.IGNORECASE)
SPACE = re.compile(r"\s+")


def clean_text(text):
    """Normalize encoding and whitespace while retaining punctuation and negation."""
    text = unicodedata.normalize("NFKC", html.unescape(text))
    return SPACE.sub(" ", BREAK.sub(" ", text)).strip()


def read_data(path, labeled):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"Expected a nonempty list of reviews in {path}")
    reviews = [row["reviews"] for row in data]
    if any(not isinstance(review, str) or not clean_text(review) for review in reviews):
        raise ValueError(f"Blank or invalid review in {path}")
    labels = None
    if labeled:
        labels = np.asarray([row["sentiments"] for row in data], dtype=np.int64)
        if set(labels) != {0, 1}:
            raise ValueError(f"Expected both binary labels in {path}")
    return reviews, labels


def check_sample(path, reviews):
    with path.open(newline="", encoding="utf-8-sig") as source:
        template = csv.DictReader(source)
        if template.fieldnames != ["reviews", "predictions"]:
            raise ValueError(f"Unexpected official submission header: {template.fieldnames}")
        if [row["reviews"] for row in template] != reviews:
            raise ValueError("The official sample does not match the test review strings and order.")


def group_split(train_reviews, labels, test_reviews):
    train_groups = np.asarray([review.casefold() for review in train_reviews], dtype=object)
    test_groups = {review.casefold() for review in test_reviews}
    overlap = set(train_groups) & test_groups
    retained = np.flatnonzero(~np.isin(train_groups, list(overlap)))
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = [retained[held] for _, held in splitter.split(
        np.zeros(len(retained)), labels[retained], train_groups[retained]
    )]
    fit, dev, audit = np.sort(np.concatenate(folds[2:])), np.sort(folds[0]), np.sort(folds[1])
    parts = [set(train_groups[part]) for part in (fit, dev, audit)]
    if parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2]:
        raise ValueError("A duplicate group crossed an internal split.")
    diagnostics = {
        "raw_train": len(train_reviews),
        "raw_test": len(test_reviews),
        "test_duplicate_rows": len(test_reviews) - len(test_groups),
        "train_duplicate_rows": len(train_reviews) - len(set(train_groups)),
        "overlap_groups": len(overlap),
        "overlap_train_rows_excluded": len(train_reviews) - len(retained),
        "overlap_test_rows": sum(review.casefold() in overlap for review in test_reviews),
        "train_class_counts": class_counts(labels),
        "retained_class_counts": class_counts(labels[retained]),
    }
    return fit, dev, audit, np.sort(retained), diagnostics


def class_counts(labels):
    return np.bincount(labels, minlength=2).tolist()


def vectorizer(mode):
    common = dict(min_df=2, sublinear_tf=True, dtype=np.float32)
    if mode == "word_unigram":
        return TfidfVectorizer(ngram_range=(1, 1), max_features=20000, **common)
    if mode == "word_bigram":
        return TfidfVectorizer(ngram_range=(1, 2), max_features=28000, **common)
    if mode == "character":
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=40000, **common)
    return FeatureUnion(
        [("word", vectorizer("word_bigram")), ("char", vectorizer("character"))],
        transformer_weights={"word": 1.0, "char": 0.7},
    )


def representation_size(matrix):
    if sparse.issparse(matrix):
        return matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    return matrix.nbytes


def scores(actual, predicted):
    return {
        "n": int(len(actual)),
        "accuracy": float(np.mean(actual == predicted)),
        "macro_f1": float(f1_score(actual, predicted, average="macro")),
        "negative_recall": float(recall_score(actual, predicted, pos_label=0, zero_division=0)),
        "positive_recall": float(recall_score(actual, predicted, pos_label=1, zero_division=0)),
        "confusion": confusion_matrix(actual, predicted, labels=[0, 1]).tolist(),
    }


def fit_linear(x, y, config):
    return LogisticRegression(
        C=config["C"], class_weight="balanced" if config["balanced"] else None,
        solver="liblinear", max_iter=1500, tol=1e-5, random_state=SEED,
    ).fit(x, y)


def fit_neural(x, y, config):
    net = MLPClassifier(
        hidden_layer_sizes=(64,), activation="relu", solver="adam",
        alpha=0.003, batch_size=128, learning_rate_init=config["learning_rate"], random_state=SEED,
    )
    weights = compute_sample_weight("balanced", y) if config["balanced"] else None
    for _ in range(EPOCHS):
        net.partial_fit(x, y, classes=[0, 1], sample_weight=weights)
    return net


def project_features(x, other):
    n_components = min(128, x.shape[0] - 1, x.shape[1] - 1)
    projector = TruncatedSVD(n_components=n_components, n_iter=5, random_state=SEED)
    z = projector.fit_transform(x)
    scaler = StandardScaler()
    return scaler.fit_transform(z), scaler.transform(projector.transform(other))


def fit_evaluate(config, texts, labels, eval_text):
    extract = vectorizer(config["features"])
    x = extract.fit_transform(texts)
    eval_x = extract.transform(eval_text)
    if config["model"] == "MLP":
        x, eval_x = project_features(x, eval_x)
        model = fit_neural(x, labels, config)
    else:
        model = fit_linear(x, labels, config)
    return model.predict(eval_x), model.predict_proba(eval_x)[:, 1], x.shape[1]


def evaluate_candidates(train_text, train_y, dev_text, dev_y):
    candidates = []
    resource = []
    for name in ("word_unigram", "word_bigram", "character", "hybrid"):
        began = perf_counter()
        extractor = vectorizer(name)
        x = extractor.fit_transform(train_text)
        val = extractor.transform(dev_text)
        elapsed = perf_counter() - began
        resource.append({
            "features": name,
            "dimension": x.shape[1],
            "fit_transform_seconds": elapsed,
            "train_matrix_bytes": representation_size(x),
            "train_nonzero": x.nnz,
        })
        for c in (0.3, 1.0, 3.0, 10.0):
            for balanced in (False, True):
                config = {"model": "Logistic regression", "features": name, "C": c, "balanced": balanced}
                start = perf_counter()
                model = fit_linear(x, train_y, config)
                metric = scores(dev_y, model.predict(val))
                candidates.append({"config": config, "dev": metric,
                                   "fit_seconds": perf_counter() - start,
                                   "iterations": int(model.n_iter_[0])})
        print(f"{name}: {x.shape[1]:,} features, best dev F1 "
              f"{max(row['dev']['macro_f1'] for row in candidates if row['config']['features'] == name):.4f}", flush=True)
        if name == "word_bigram":
            began = perf_counter()
            z, val_z = project_features(x, val)
            resource.append({
                "features": "word_bigram_svd",
                "dimension": z.shape[1],
                "fit_transform_seconds": elapsed + perf_counter() - began,
                "train_matrix_bytes": representation_size(z),
                "train_nonzero": int(np.count_nonzero(z)),
            })
            for lr in (0.001, 0.003):
                for balanced in (False, True):
                    config = {"model": "MLP", "features": name, "learning_rate": lr, "balanced": balanced}
                    start = perf_counter()
                    model = fit_neural(z, train_y, config)
                    candidates.append({"config": config, "dev": scores(dev_y, model.predict(val_z)),
                                       "fit_seconds": perf_counter() - start,
                                       "epochs": EPOCHS, "last_epoch_loss": float(model.loss_)})
    return candidates, resource


def rank(candidate):
    metric = candidate["dev"]
    return metric["macro_f1"], metric["accuracy"], metric["negative_recall"]


def sha256(path):
    with path.open("rb") as data:
        return hashlib.file_digest(data, "sha256").hexdigest()


def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def run(args):
    original_train, y = read_data(args.train, True)
    original_test, _ = read_data(args.test, False)
    check_sample(args.sample, original_test)
    train = np.asarray([clean_text(text) for text in original_train], dtype=object)
    test = [clean_text(text) for text in original_test]
    fit, dev, audit, retained, data = group_split(train, y, test)
    data["split_counts"] = {name: class_counts(y[index]) for name, index in
                            [("fit", fit), ("dev", dev), ("audit", audit)]}
    data["sha256"] = {name: sha256(getattr(args, name)) for name in ("train", "test", "sample")}
    print("Data split counts [negative, positive]:", data["split_counts"], flush=True)

    candidates, resource = evaluate_candidates(train[fit], y[fit], train[dev], y[dev])
    best = max(candidates, key=rank)
    baseline = next(row for row in candidates if row["config"] == {
        "model": "Logistic regression", "features": "word_unigram", "C": 1.0, "balanced": False,
    })
    best_neural = max((row for row in candidates if row["config"]["model"] == "MLP"), key=rank)
    train_dev = np.sort(np.concatenate([fit, dev]))
    audit_results = {}
    for name, candidate in (("baseline", baseline), ("best_neural", best_neural), ("selected", best)):
        labels, probability, n_features = fit_evaluate(candidate["config"], train[train_dev], y[train_dev], train[audit])
        audit_results[name] = {"config": candidate["config"], "dimension": n_features,
                               **scores(y[audit], labels)}
        if name == "selected":
            write_csv(args.output / "audit_predictions.csv",
                      ["train_index", "actual", "predicted", "p_positive"],
                      ((i, a, p, f"{score:.9f}") for i, a, p, score in zip(audit, y[audit], labels, probability)))
    dummy = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(train_dev), 1)), y[train_dev])
    audit_results["majority"] = scores(y[audit], dummy.predict(np.zeros((len(audit), 1))))

    prediction, p, dim = fit_evaluate(best["config"], train[retained], y[retained], test)
    write_csv(args.output / "submission.csv", ["reviews", "predictions"], zip(original_test, prediction.tolist()))
    write_csv(args.output / "test_scores.csv", ["test_index", "prediction", "p_positive"],
              ((i, label, f"{prob:.9f}") for i, (label, prob) in enumerate(zip(prediction, p))))
    result = {
        "data": data, "selection_rule": "dev macro-F1, then accuracy, then negative recall; stable grid order on ties",
        "resource": resource, "candidates": candidates, "selected": best,
        "audit": audit_results, "final_dimension": dim, "test_predicted_counts": class_counts(prediction),
        "environment": {"seed": SEED, "device": "CPU", "platform": platform.platform(),
                        "python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "sklearn": sklearn.__version__},
    }
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("Selected:", best["config"], flush=True)
    print("Audit:", audit_results["selected"], flush=True)
    print("Predictions [negative, positive]:", result["test_predicted_counts"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=4):
        run(args)


if __name__ == "__main__":
    main()
