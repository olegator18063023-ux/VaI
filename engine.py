from dataclasses import dataclass
from itertools import permutations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    mean_absolute_error,
    roc_auc_score,
)

SEED = 42
WINDOW = 12
HORIZON = 24

SENSORS = ["temperature", "vibration", "pressure"]
LABELS = {
    "temperature": "Температура подшипника",
    "vibration": "Вибрация",
    "pressure": "Давление масла",
}
UNITS = {
    "temperature": "°C",
    "vibration": "мм/с",
    "pressure": "бар",
}


def generate_unit(unit_id: int, seed: int) -> pd.DataFrame:
    """Полный синтетический жизненный цикл; один шаг равен часу."""
    rng = np.random.default_rng(seed)
    lifetime = int(rng.integers(190, 321))
    hour = np.arange(lifetime + 1)
    progress = hour / lifetime

    # Общая деградация и независимые особенности экземпляра.
    wear = progress ** rng.uniform(1.15, 1.65)
    load = np.sin(hour / 15 + rng.uniform(0, 6.28))
    noise_scale = 0.4 + wear

    temperature = (
        49
        + rng.normal(0, 2)
        + rng.uniform(39, 47) * wear
        + 1.4 * load
        + rng.normal(0, 0.8, len(hour)) * noise_scale
    )
    vibration = (
        0.8
        + rng.normal(0, 0.08)
        + rng.uniform(6.7, 8.0) * wear ** 1.25
        + 0.08 * load
        + rng.normal(0, 0.10, len(hour)) * noise_scale
    )
    pressure = (
        5.3
        + rng.normal(0, 0.12)
        - rng.uniform(3.0, 3.6) * wear
        + 0.08 * load
        + rng.normal(0, 0.04, len(hour)) * noise_scale
    )

    return pd.DataFrame({
        "unit_id": unit_id,
        "hour": hour,
        "temperature": temperature,
        "vibration": np.maximum(vibration, 0.05),
        "pressure": np.maximum(pressure, 0.1),
        "rul": lifetime - hour,
    })


def make_features(history: pd.DataFrame) -> pd.DataFrame:
    """Причинные оконные признаки; вызывать отдельно для каждого узла."""
    missing = set(SENSORS) - set(history.columns)
    if missing:
        raise ValueError(f"Отсутствуют датчики: {sorted(missing)}")
    if len(history) < WINDOW:
        raise ValueError(f"Необходимо минимум {WINDOW} измерений")
    if not np.isfinite(history[SENSORS].to_numpy()).all():
        raise ValueError("Телеметрия содержит NaN или бесконечность")

    result = pd.DataFrame(index=history.index)
    for sensor in SENSORS:
        series = history[sensor]
        rolling = series.rolling(WINDOW, min_periods=WINDOW)
        result[f"{sensor}_last"] = series
        result[f"{sensor}_mean"] = rolling.mean()
        result[f"{sensor}_std"] = rolling.std(ddof=0)
        result[f"{sensor}_trend"] = (
            series - series.shift(WINDOW - 1)
        ) / (WINDOW - 1)

    return result.dropna()


def build_dataset(count: int = 80) -> pd.DataFrame:
    chunks = []
    for unit_id in range(count):
        frame = generate_unit(unit_id, SEED + unit_id)
        features = make_features(frame)
        features["unit_id"] = unit_id
        features["rul"] = frame.loc[features.index, "rul"]

        # Уменьшаем объем обучения; не используем измерение в момент отказа.
        chunks.append(features.iloc[::3].query("rul > 0"))

    return pd.concat(chunks, ignore_index=True)


@dataclass
class ModelBundle:
    regressor: RandomForestRegressor
    classifier: RandomForestClassifier
    calibrator: LogisticRegression
    columns: list
    reference: pd.Series
    lower: pd.Series
    upper: pd.Series
    metrics: dict

    def risk(self, features: pd.DataFrame) -> np.ndarray:
        raw = self.classifier.predict_proba(
            features[self.columns]
        )[:, 1]
        return self.calibrator.predict_proba(
            raw.reshape(-1, 1)
        )[:, 1]

    def predict(self, features: pd.DataFrame):
        x = features[self.columns]
        rul = np.maximum(self.regressor.predict(x), 0)
        return rul, self.risk(x)


def train_models() -> ModelBundle:
    dataset = build_dataset()
    ids = np.random.default_rng(SEED).permutation(80)

    train_ids = set(ids[:52])
    calibration_ids = set(ids[52:66])
    test_ids = set(ids[66:])

    assert train_ids.isdisjoint(calibration_ids)
    assert train_ids.isdisjoint(test_ids)
    assert calibration_ids.isdisjoint(test_ids)

    train = dataset[dataset.unit_id.isin(train_ids)]
    calibration = dataset[dataset.unit_id.isin(calibration_ids)]
    test = dataset[dataset.unit_id.isin(test_ids)]

    columns = [
        column for column in dataset.columns
        if column not in {"unit_id", "rul"}
    ]

    regressor = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=4,
        max_features=0.9,
        n_jobs=-1,
        random_state=SEED,
    )
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        max_features=0.9,
        n_jobs=-1,
        random_state=SEED,
    )

    regressor.fit(train[columns], train.rul)
    classifier.fit(train[columns], (train.rul <= HORIZON).astype(int))

    # Сигмоидная калибровка на отдельных экземплярах оборудования.
    raw_calibration = classifier.predict_proba(
        calibration[columns]
    )[:, 1].reshape(-1, 1)

    calibrator = LogisticRegression(C=100, solver="lbfgs")
    calibrator.fit(
        raw_calibration,
        (calibration.rul <= HORIZON).astype(int),
    )

    # Реальное учебное окно с высоким RUL как фиксированный XAI-эталон.
    healthy = train[train.rul >= train.rul.quantile(0.8)]
    center = healthy[columns].median()
    scale = train[columns].std().replace(0, 1)
    distance = (((healthy[columns] - center) / scale) ** 2).sum(axis=1)
    reference = healthy.loc[distance.idxmin(), columns].copy()

    bundle = ModelBundle(
        regressor=regressor,
        classifier=classifier,
        calibrator=calibrator,
        columns=columns,
        reference=reference,
        lower=train[columns].min(),
        upper=train[columns].max(),
        metrics={},
    )

    predicted_rul, predicted_risk = bundle.predict(test[columns])
    truth = (test.rul <= HORIZON).astype(int)

    bundle.metrics = {
        "MAE RUL, ч": float(mean_absolute_error(test.rul, predicted_rul)),
        "MAE константного baseline, ч": float(
            mean_absolute_error(
                test.rul,
                np.full(len(test), train.rul.median()),
            )
        ),
        "ROC-AUC": float(roc_auc_score(truth, predicted_risk)),
        "Brier score": float(brier_score_loss(truth, predicted_risk)),
        "Узлов: обучение": len(train_ids),
        "Узлов: калибровка": len(calibration_ids),
        "Узлов: тест": len(test_ids),
        "Тестовых окон": len(test),
    }
    return bundle


def simulate(
    history: pd.DataFrame,
    temperature_delta: float,
    vibration_delta: float,
    pressure_delta: float,
) -> pd.DataFrame:
    """Плавно вводит возмущение в последние WINDOW часов."""
    result = history.copy(deep=True)
    ramp = np.linspace(0, 1, min(WINDOW, len(result)))

    for sensor, delta in zip(
        SENSORS,
        [temperature_delta, vibration_delta, pressure_delta],
    ):
        column = result.columns.get_loc(sensor)
        result.iloc[-len(ramp):, column] += ramp * delta

    result["vibration"] = result["vibration"].clip(lower=0.05)
    result["pressure"] = result["pressure"].clip(lower=0.1)
    return result


def local_explanation(bundle: ModelBundle, row: pd.Series):
    """
    Точные Shapley-вклады трех групп датчиков относительно одного эталона.
    Все признаки одного датчика заменяются совместно.
    """
    coalitions = []
    for mask in range(8):
        hybrid = bundle.reference.copy()
        for index, sensor in enumerate(SENSORS):
            if mask & (1 << index):
                names = [
                    name for name in bundle.columns
                    if name.startswith(sensor + "_")
                ]
                hybrid.loc[names] = row.loc[names]
        coalitions.append(hybrid)

    values = bundle.risk(pd.DataFrame(coalitions)[bundle.columns])
    contributions = np.zeros(3)

    for order in permutations(range(3)):
        mask = 0
        for index in order:
            next_mask = mask | (1 << index)
            contributions[index] += values[next_mask] - values[mask]
            mask = next_mask

    contributions /= 6
    return contributions, float(values[0]), float(values[7])


def outside_training_range(bundle: ModelBundle, row: pd.Series) -> list:
    bad = (row[bundle.columns] < bundle.lower) | (
        row[bundle.columns] > bundle.upper
    )
    return bad.index[bad].tolist()
