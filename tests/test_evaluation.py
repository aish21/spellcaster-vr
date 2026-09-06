import numpy as np
import pytest

from spellcaster.gestures.spells import Spell
from spellcaster.ml.dataset import (
    MLDataset,
)
from spellcaster.ml.evaluation import (
    evaluate_knn_cross_validation,
)


def test_knn_cross_validation_tracks_predictions_and_neighbors():

    features = np.asarray(
        [
            # Fireball cluster
            [0.00, 0.00],
            [0.05, 0.00],
            [0.00, 0.05],
            [0.05, 0.05],
            [0.02, 0.02],
            # Shield cluster
            [10.00, 10.00],
            [10.05, 10.00],
            [10.00, 10.05],
            [10.05, 10.05],
            [10.02, 10.02],
        ],
        dtype=np.float64,
    )

    labels = np.asarray(
        [
            Spell.FIREBALL.value,
            Spell.FIREBALL.value,
            Spell.FIREBALL.value,
            Spell.FIREBALL.value,
            Spell.FIREBALL.value,
            Spell.SHIELD.value,
            Spell.SHIELD.value,
            Spell.SHIELD.value,
            Spell.SHIELD.value,
            Spell.SHIELD.value,
        ],
        dtype=np.str_,
    )

    sample_ids = tuple(f"sample-{index}" for index in range(10))

    dataset = MLDataset(
        features=features,
        labels=labels,
        sample_ids=sample_ids,
        session_ids=tuple("test-session" for _ in sample_ids),
    )

    evaluation = evaluate_knn_cross_validation(
        dataset=dataset,
        n_neighbors=1,
        n_splits=5,
        random_state=42,
    )

    assert evaluation.accuracy == pytest.approx(1.0)

    assert len(evaluation.predictions) == 10

    assert evaluation.confusion_matrix.tolist() == [
        [5, 0],
        [0, 5],
    ]

    predicted_ids = {prediction.sample_id for prediction in evaluation.predictions}

    assert predicted_ids == set(sample_ids)

    for prediction in evaluation.predictions:

        assert len(prediction.neighbor_ids) == 1

        assert len(prediction.neighbor_labels) == 1

        assert len(prediction.neighbor_distances) == 1


def test_knn_cross_validation_requires_enough_samples_per_class():

    features = np.zeros(
        (
            8,
            2,
        ),
        dtype=np.float64,
    )

    labels = np.asarray(
        [
            Spell.FIREBALL.value,
            Spell.FIREBALL.value,
            Spell.FIREBALL.value,
            Spell.FIREBALL.value,
            Spell.SHIELD.value,
            Spell.SHIELD.value,
            Spell.SHIELD.value,
            Spell.SHIELD.value,
        ],
        dtype=np.str_,
    )
    sample_ids = tuple(f"sample-{index}" for index in range(8))

    dataset = MLDataset(
        features=features,
        labels=labels,
        sample_ids=sample_ids,
        session_ids=tuple("test-session" for _ in sample_ids),
    )

    with pytest.raises(
        ValueError,
        match="at least 5 samples",
    ):

        evaluate_knn_cross_validation(
            dataset=dataset,
            n_neighbors=1,
            n_splits=5,
        )


def test_knn_cross_validation_rejects_unknown_weighting():

    features = np.zeros(
        (
            10,
            2,
        ),
        dtype=np.float64,
    )

    labels = np.asarray(
        [
            Spell.FIREBALL.value,
            Spell.FIREBALL.value,
            Spell.FIREBALL.value,
            Spell.FIREBALL.value,
            Spell.FIREBALL.value,
            Spell.SHIELD.value,
            Spell.SHIELD.value,
            Spell.SHIELD.value,
            Spell.SHIELD.value,
            Spell.SHIELD.value,
        ],
        dtype=np.str_,
    )

    sample_ids = tuple(f"sample-{index}" for index in range(8))

    dataset = MLDataset(
        features=features,
        labels=labels,
        sample_ids=sample_ids,
        session_ids=tuple("test-session" for _ in sample_ids),
    )

    with pytest.raises(
        ValueError,
        match="uniform.*distance",
    ):

        evaluate_knn_cross_validation(
            dataset=dataset,
            n_neighbors=1,
            n_splits=5,
            weights="banana",  # type: ignore[arg-type]
        )
