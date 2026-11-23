"""多ラベル分類の入力、学習、採点と提出。"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from torch import nn

SEED = 42
TRAIN_FILES = ("train_features.csv", "train_targets_scored.csv", "train_drug.csv")
FILES = (*TRAIN_FILES, "test_features.csv", "sample_submission.csv")
HYPOTHESES = {
    "E01": "処理群の独立ロジスティック回帰",
    "E02": "遺伝子発現を除くと Log Loss が悪化する",
    "E03": "出現率へ 10% 縮小すると希少ラベルの過信が減る",
    "E04": "共有隠れ層で希少ラベルの損失が下がる",
    "E05": "遺伝子発現の PCA 圧縮で独立回帰の損失が下がる",
}


def aligned(path: Path, ids: pd.Index) -> pd.DataFrame:
    table = pd.read_csv(path, index_col="sig_id")
    if not table.index.is_unique or len(table) != len(ids) or not ids.isin(table.index).all():
        raise ValueError(f"ID 集合の不一致: {path.name}")
    return table.loc[ids]


def load_training(data: Path) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    train = pd.read_csv(data / TRAIN_FILES[0], index_col="sig_id").sort_index()
    if not train.index.is_unique:
        raise ValueError("学習 ID が重複しています")
    labels = aligned(data / TRAIN_FILES[1], train.index)
    groups = aligned(data / TRAIN_FILES[2], train.index)["drug_id"]
    target = labels.to_numpy()
    if groups.isna().any() or not np.isin(target, [0, 1]).all():
        raise ValueError("化合物 ID またはラベルが不正です")
    if target[~treated(train)].any():
        raise ValueError("対照群に陽性ラベルがあります")
    return train, labels, groups.to_numpy()


def treated(frame: pd.DataFrame) -> np.ndarray:
    status = frame["cp_type"]
    if not status.isin(["trt_cp", "ctl_vehicle"]).all():
        raise ValueError("不明な cp_type です")
    return status.eq("trt_cp").to_numpy(dtype=bool)


def feature_columns(frame: pd.DataFrame, trial: str) -> list[str]:
    prefixes = ("c-",) if trial == "E02" else ("g-", "c-")
    columns = sorted(
        (column for column in frame.columns if column.startswith(prefixes)),
        key=lambda column: (column[0] != "g", int(column[2:])),
    )
    return [*columns, "cp_time"]


def features(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    dose = frame["cp_dose"].map({"D1": 0, "D2": 1}).to_numpy(dtype=float)
    values = np.column_stack([frame[columns].to_numpy(dtype=float), dose])
    if not np.isfinite(values).all():
        raise ValueError("特徴量に欠損または非有限値があります")
    return values


def independent(x: np.ndarray, y: np.ndarray, prior: np.ndarray) -> dict[str, np.ndarray]:
    positive = y.sum(axis=0)
    active = (positive > 0) & (positive < len(y))
    if not active.any():
        return {"active": active, "weight": np.empty((x.shape[1], 0)), "bias": np.empty(0)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(8)
    inputs = torch.as_tensor(x, dtype=torch.float32, device=device)
    labels = torch.as_tensor(y[:, active], dtype=torch.float32, device=device)
    weights = torch.zeros((x.shape[1], labels.shape[1]), device=device, requires_grad=True)
    bias = torch.as_tensor(prior[active], dtype=torch.float32, device=device).logit()
    bias.requires_grad_()
    optimizer = torch.optim.LBFGS(
        [weights, bias], max_iter=200, history_size=15, line_search_fn="strong_wolfe"
    )

    def objective() -> torch.Tensor:
        optimizer.zero_grad()
        logits = inputs @ weights + bias
        data_loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="sum")
        loss = (data_loss + weights.square().sum() / 2) / len(x)
        loss.backward()
        return loss

    optimizer.step(objective)
    return {
        "active": active,
        "weight": weights.detach().cpu().numpy(),
        "bias": bias.detach().cpu().numpy(),
    }


def shared(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(SEED)
    torch.manual_seed(SEED)
    inputs = torch.as_tensor(x, dtype=torch.float32)
    labels = torch.as_tensor(y, dtype=torch.float32)
    model = nn.Sequential(nn.Linear(x.shape[1], 256), nn.ReLU(), nn.Linear(256, y.shape[1]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    loss = nn.BCEWithLogitsLoss()
    for _ in range(20):
        for batch in torch.randperm(len(x), generator=generator).split(256):
            optimizer.zero_grad()
            loss(model(inputs[batch]), labels[batch]).backward()
            optimizer.step()
    return {name: value.detach().numpy() for name, value in model.state_dict().items()}


def project(x: np.ndarray, state: dict[str, np.ndarray]) -> np.ndarray:
    gene_mask = state["gene_mask"]
    genes = (x[:, gene_mask] - state["pca_mean"]) @ state["pca_components"].T
    return np.column_stack([genes, x[:, ~gene_mask]])


def fit_model(train: pd.DataFrame, y: np.ndarray, trial: str) -> dict[str, np.ndarray]:
    training = treated(train)
    columns = feature_columns(train, trial)
    x = features(train.loc[training], columns)
    label = y[training]
    prior = (label.sum(axis=0) + 0.5) / (len(label) + 1)
    state = {
        "trial": np.array(trial),
        "columns": np.asarray(columns, dtype=str),
        "prior": prior,
    }
    with threadpool_limits(limits=2):
        scale = StandardScaler().fit(x)
        state.update(
            mean=np.asarray(scale.mean_, dtype=float), scale=np.asarray(scale.scale_, dtype=float)
        )
        x = np.asarray(scale.transform(x))
        if trial == "E05":
            genes = [column.startswith("g-") for column in columns]
            gene_mask = np.array([*genes, False])
            pca = PCA(n_components=64, svd_solver="randomized", random_state=SEED)
            pca.fit(x[:, gene_mask])
            state.update(
                gene_mask=gene_mask,
                pca_mean=np.asarray(pca.mean_, dtype=float),
                pca_components=np.asarray(pca.components_, dtype=float),
            )
            x = project(x, state)
        state.update(shared(x, label) if trial == "E04" else independent(x, label, prior))
    return state


def apply_model(query: pd.DataFrame, state: dict[str, np.ndarray]) -> np.ndarray:
    trial = str(state["trial"].item())
    prior = state["prior"]
    inference = treated(query)
    result = np.zeros((len(query), len(prior)))
    if not inference.any():
        return result
    raw = features(query.loc[inference], state["columns"].tolist())
    with threadpool_limits(limits=2):
        x = (raw - state["mean"]) / state["scale"]
        if trial == "E05":
            x = project(x, state)
        if trial == "E04":
            hidden = np.maximum(x @ state["0.weight"].T + state["0.bias"], 0)
            prediction = expit(hidden @ state["2.weight"].T + state["2.bias"])
        else:
            prediction = np.tile(prior, (len(x), 1))
            prediction[:, state["active"]] = expit(x @ state["weight"] + state["bias"])
    if trial == "E03":
        prediction = 0.9 * prediction + 0.1 * prior
    result[inference] = prediction
    return result


def predict(train: pd.DataFrame, y: np.ndarray, query: pd.DataFrame, trial: str) -> np.ndarray:
    return apply_model(query, fit_model(train, y, trial))


def losses(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    p = np.clip(pred.astype(np.float64), 1e-15, 1 - 1e-15)
    return -np.where(y == 1, np.log(p), np.log1p(-p))


def log_loss(y: np.ndarray, pred: np.ndarray) -> float:
    return float(losses(y, pred).mean())


def write_submission(data: Path, state: dict[str, np.ndarray], output: Path) -> None:
    sample = pd.read_csv(data / FILES[-1], index_col="sig_id")
    if not sample.index.is_unique or not np.array_equal(sample.columns, state["labels"]):
        raise ValueError("提出ラベル列が学習ラベル列と一致しません")
    test = aligned(data / FILES[-2], sample.index)
    pred = apply_model(test, state)
    if pred.shape != sample.shape or not (np.isfinite(pred) & (pred >= 0) & (pred <= 1)).all():
        raise ValueError("提出確率が不正です")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pred, index=sample.index, columns=sample.columns).to_csv(output)
