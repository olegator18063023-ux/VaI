import numpy as np
import pandas as pd
import pytest

from engine import (
    SENSORS,
    generate_unit,
    local_explanation,
    make_features,
    simulate,
    train_models,
)


@pytest.fixture(scope="module")
def bundle():
    return train_models()


def test_features_do_not_use_future():
    full = generate_unit(900, 900)
    prefix = full.iloc[:100].copy()

    expected = make_features(prefix)
    actual = make_features(full).loc[expected.index]

    pd.testing.assert_frame_equal(expected, actual)


def test_targets_are_not_features():
    frame = generate_unit(901, 901)
    features = make_features(frame)

    assert "rul" not in features.columns
    assert "unit_id" not in features.columns
    assert "hour" not in features.columns
    assert np.isfinite(features.to_numpy()).all()


def test_simulator_does_not_mutate_input():
    original = generate_unit(902, 902).iloc[:100].copy()
    snapshot = original.copy(deep=True)

    changed = simulate(original, 10.0, 2.0, -1.0)

    pd.testing.assert_frame_equal(original, snapshot)
    pd.testing.assert_frame_equal(
        changed.iloc[:-12],
        original.iloc[:-12],
    )
    assert changed.temperature.iloc[-1] == pytest.approx(
        original.temperature.iloc[-1] + 10.0
    )


def test_zero_intervention():
    frame = generate_unit(903, 903).iloc[:100].copy()
    pd.testing.assert_frame_equal(
        frame,
        simulate(frame, 0.0, 0.0, 0.0),
    )


def test_prediction_and_local_explanation(bundle):
    frame = generate_unit(904, 904).iloc[:150]
    features = make_features(frame).tail(1)

    rul, risk = bundle.predict(features)
    assert np.isfinite(rul).all()
    assert rul[0] >= 0
    assert 0 <= risk[0] <= 1

    contributions, reference, current = local_explanation(
        bundle, features.iloc[0]
    )
    assert len(contributions) == len(SENSORS)
    assert reference + contributions.sum() == pytest.approx(
        current, abs=1e-8
    )
    assert current == pytest.approx(risk[0], abs=1e-8)
