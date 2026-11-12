import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.core.feature_validation import require


def fit_logistic(x, y, c=1.0, class_weight="balanced", max_iter=1000, tolerance_grad=1e-7):
    """표준화한 특징에서 L2 정규화 로지스틱 회귀 p = σ(wᵀz + b)를 float64 LBFGS로 학습한다.

    목적함수: 0.5·||w||² + c·Σ sample_weight·BCE (bias는 정규화하지 않음)
    """
    require(c > 0, "Regularization constant c must be positive")
    require(class_weight in ("balanced", None), f"Unsupported class_weight: {class_weight}")
    x, y = x.double(), y.double()
    positives = int(y.sum())
    negatives = len(y) - positives
    require(positives > 0 and negatives > 0, "Training data must contain both labels")
    mean = x.mean(dim=0)
    scale = x.std(dim=0, unbiased=False)
    scale = torch.where(scale > 1e-12, scale, torch.ones_like(scale))
    z = (x - mean) / scale
    if class_weight == "balanced":
        sample_weight = torch.where(y > 0.5, len(y) / (2 * positives), len(y) / (2 * negatives))
    else:
        sample_weight = torch.ones_like(y)

    weight = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    bias = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [weight, bias], lr=1, max_iter=max_iter, tolerance_grad=tolerance_grad, tolerance_change=1e-12,
        history_size=20, line_search_fn="strong_wolfe",
    )

    def objective():
        logits = z @ weight + bias
        loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        return c * (sample_weight * loss).sum() + 0.5 * weight.dot(weight)

    def closure():
        optimizer.zero_grad()
        loss = objective()
        loss.backward()
        return loss

    optimizer.step(closure)
    loss = objective()
    gradients = torch.autograd.grad(loss, [weight, bias])
    iterations = optimizer.state[weight]["n_iter"]
    return {
        "mean": mean, "scale": scale, "weight": weight.detach(), "bias": float(bias.detach()),
        "loss": float(loss.detach()), "iterations": iterations,
        "grad_max": max(float(gradient.abs().max()) for gradient in gradients),
        "converged": iterations < max_iter,
    }


def predict(model, x):
    z = (x.double() - model["mean"]) / model["scale"]
    return torch.sigmoid(z @ model["weight"] + model["bias"])


def auroc(scores, labels):
    """Mann-Whitney 방식 AUROC (동점은 0.5). 한 라벨만 있으면 None."""
    scores = torch.as_tensor(scores, dtype=torch.float64)
    labels = torch.as_tensor(labels, dtype=torch.bool)
    positive, negative = scores[labels], scores[~labels]
    if not len(positive) or not len(negative):
        return None
    greater = (positive[:, None] > negative[None, :]).double().sum()
    ties = (positive[:, None] == negative[None, :]).double().sum()
    return float((greater + 0.5 * ties) / (len(positive) * len(negative)))


def score(probabilities, labels, threshold):
    """labels: True = P. 예측은 p >= threshold이면 P."""
    probabilities = torch.as_tensor(probabilities, dtype=torch.float64)
    labels = torch.as_tensor(labels, dtype=torch.bool)
    n = len(labels)
    n_positive = int(labels.sum())
    if not n:
        return {"n": 0, "n_P": 0, "n_N": 0, "accuracy": None, "auroc": None}
    correct = (probabilities >= threshold) == labels
    return {
        "n": n, "n_P": n_positive, "n_N": n - n_positive,
        "accuracy": float(correct.double().mean()), "auroc": auroc(probabilities, labels),
    }


def cross_fit(vectors, matrix, layers, image_fold, folds, settings, threshold):
    """Fold × 클래스 × 레이어별 Probe 학습과 Out-of-Fold 예측.

    Train (엄격): 대상 또는 prefix에 Fold k의 이미지가 하나라도 있는 벡터는 제외한다.
    Validation: 대상 이미지가 Fold k에 속하는 벡터. prefix의 Train 이미지 포함 여부는 context로 기록한다.
    모든 문맥 벡터는 자기 대상 이미지가 속한 Fold의 Probe로 정확히 한 번 예측된다.
    """
    oof = torch.full((len(vectors), len(layers)), float("nan"), dtype=torch.float64)
    probes, checkpoints = [], []
    classes = sorted({vector["class"] for vector in vectors})
    for fold in range(folds):
        for name in classes:
            members = [vector for vector in vectors if vector["class"] == name]
            train = [v["vector"] for v in members if all(image_fold[i] != fold for i in v["prefix"])]
            val = [v["vector"] for v in members if v["fold"] == fold]
            excluded = [v for v in members if v["fold"] != fold and any(image_fold[i] == fold for i in v["prefix"])]
            require(len(train) + len(val) + len(excluded) == len(members), "Fold partition is inconsistent")
            train_images = {i for index in train for i in vectors[index]["prefix"]}
            val_targets = {vectors[index]["image_id"] for index in val}
            require(bool(val), f"No validation vectors: fold {fold}, {name}")
            require(not train_images & val_targets, f"Validation target appears in training data: fold {fold}, {name}")
            train_labels = torch.tensor([vectors[index]["label"] == "P" for index in train])
            val_labels = torch.tensor([vectors[index]["label"] == "P" for index in val])
            val_first = [j for j, index in enumerate(val) if vectors[index]["position"] == 1]
            for column, layer in enumerate(layers):
                model = fit_logistic(matrix[train, column], train_labels, **settings)
                train_probability = predict(model, matrix[train, column])
                val_probability = predict(model, matrix[val, column])
                require(bool(torch.isnan(oof[val, column]).all()), "A context vector was predicted twice")
                oof[val, column] = val_probability
                train_score = score(train_probability, train_labels, threshold)
                val_score = score(val_probability, val_labels, threshold)
                image_score = score(val_probability[val_first], val_labels[val_first], threshold)
                probes.append({
                    "fold": fold, "class": name, "layer": layer,
                    "n_train_vectors": len(train), "n_train_P": train_score["n_P"], "n_train_N": train_score["n_N"],
                    "n_train_images": len({vectors[index]["image_id"] for index in train}),
                    "n_excluded_prefix": len(excluded),
                    "n_val_vectors": len(val), "n_val_images": len(val_targets),
                    "train_accuracy": train_score["accuracy"],
                    "val_context_accuracy": val_score["accuracy"], "val_context_auroc": val_score["auroc"],
                    "val_image_accuracy": image_score["accuracy"], "val_image_auroc": image_score["auroc"],
                    "iterations": model["iterations"], "grad_max": model["grad_max"],
                    "converged": model["converged"], "loss": model["loss"],
                })
                checkpoints.append({
                    "fold": fold, "class": name, "layer": layer,
                    "mean": model["mean"], "scale": model["scale"],
                    "weight": model["weight"], "bias": model["bias"],
                    "train_vectors": train, "val_vectors": val,
                })
    require(not bool(torch.isnan(oof).any()), "Some context vectors received no out-of-fold prediction")
    return oof, probes, checkpoints


def fit_final(vectors, matrix, layers, settings, threshold, progress=False):
    """클래스 × 레이어별로 모든 문맥 벡터를 써서 최종 Probe를 학습한다 (독립 검증용 고정 Probe)."""
    probes, rows = [], []
    classes = sorted({vector["class"] for vector in vectors})
    bar = tqdm(total=len(classes) * len(layers), desc="Fit z_i probes", unit="probe", disable=not progress)
    for name in classes:
        members = [vector["vector"] for vector in vectors if vector["class"] == name]
        labels = torch.tensor([vectors[index]["label"] == "P" for index in members])
        images = sorted({vectors[index]["image_id"] for index in members})
        for column, layer in enumerate(layers):
            model = fit_logistic(matrix[members, column], labels, **settings)
            train = score(predict(model, matrix[members, column]), labels, threshold)
            probes.append({"class": name, "layer": layer, "mean": model["mean"], "scale": model["scale"],
                           "weight": model["weight"], "bias": model["bias"], "train_image_ids": images})
            rows.append({"class": name, "layer": layer, "n_train_vectors": len(members),
                         "n_train_P": train["n_P"], "n_train_N": train["n_N"], "n_train_images": len(images),
                         "train_accuracy": train["accuracy"], "iterations": model["iterations"],
                         "grad_max": model["grad_max"], "converged": model["converged"], "loss": model["loss"]})
            bar.update()
    bar.close()
    return probes, rows
