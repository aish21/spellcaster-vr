import numpy as np
import pytest

from spellcaster.ml.rejection import (
    learn_rejection_threshold,
    nearest_same_class_distances,
    evaluate_open_set_leave_one_session_out,
)
from spellcaster.ml.dataset import MLDataset


def test_nearest_same_class_distances():

    features = np.asarray(
        [
            # Class A
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            # Class B
            [10.0, 0.0],
            [12.0, 0.0],
            [14.0, 0.0],
        ],
        dtype=np.float64,
    )

    labels = np.asarray(
        [
            "a",
            "a",
            "a",
            "b",
            "b",
            "b",
        ],
        dtype=np.str_,
    )

    distances = nearest_same_class_distances(
        features,
        labels,
    )

    expected = np.asarray(
        [
            1.0,
            1.0,
            1.0,
            2.0,
            2.0,
            2.0,
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        distances,
        expected,
    )


def test_rejection_threshold_uses_largest_known_distance():

    features = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [10.0, 0.0],
            [12.0, 0.0],
            [14.0, 0.0],
        ],
        dtype=np.float64,
    )

    labels = np.asarray(
        [
            "a",
            "a",
            "a",
            "b",
            "b",
            "b",
        ],
        dtype=np.str_,
    )

    threshold = learn_rejection_threshold(
        features,
        labels,
    )

    assert threshold == pytest.approx(2.0)


def test_leave_one_session_out_never_uses_held_out_session():

    features = np.asarray(
        [
            # ================================================
            # Session A
            # ================================================
            [0.00, 0.00],  # fireball
            [5.00, 0.00],  # shield
            [50.0, 50.0],  # invalid
            # ================================================
            # Session B
            # ================================================
            [0.10, 0.00],  # fireball
            [5.10, 0.00],  # shield
            [50.0, 49.0],  # invalid
            # ================================================
            # Session C
            # ================================================
            [0.05, 0.00],  # fireball
            [5.05, 0.00],  # shield
            [49.0, 50.0],  # invalid
        ],
        dtype=np.float64,
    )

    labels = np.asarray(
        [
            "fireball",
            "shield",
            "invalid",
            "fireball",
            "shield",
            "invalid",
            "fireball",
            "shield",
            "invalid",
        ],
        dtype=np.str_,
    )

    sample_ids = (
        "a-fireball",
        "a-shield",
        "a-invalid",
        "b-fireball",
        "b-shield",
        "b-invalid",
        "c-fireball",
        "c-shield",
        "c-invalid",
    )

    session_ids = (
        "session-a",
        "session-a",
        "session-a",
        "session-b",
        "session-b",
        "session-b",
        "session-c",
        "session-c",
        "session-c",
    )

    dataset = MLDataset(
        features=features,
        labels=labels,
        sample_ids=sample_ids,
        session_ids=session_ids,
    )

    evaluation = evaluate_open_set_leave_one_session_out(dataset)

    # --------------------------------------------------------
    # Every sample should have been tested exactly once.
    # --------------------------------------------------------

    assert len(evaluation.overall.predictions) == 9

    # --------------------------------------------------------
    # Build a lookup:
    #
    # sample UUID -> collection session
    # --------------------------------------------------------

    sample_to_session = {
        sample_id: session_id
        for sample_id, session_id in zip(
            sample_ids,
            session_ids,
        )
    }

    # --------------------------------------------------------
    # A held-out gesture's nearest training neighbour must
    # NEVER belong to the same collection session.
    # --------------------------------------------------------

    for prediction in evaluation.overall.predictions:

        test_session = prediction.session_id

        neighbor_session = sample_to_session[prediction.nearest_known_id]

        assert neighbor_session != test_session


def test_leave_one_session_out_requires_multiple_sessions():

    dataset = MLDataset(
        features=np.asarray(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [5.0, 0.0],
                [5.1, 0.0],
                [50.0, 50.0],
            ],
            dtype=np.float64,
        ),
        labels=np.asarray(
            [
                "fireball",
                "fireball",
                "shield",
                "shield",
                "invalid",
            ],
            dtype=np.str_,
        ),
        sample_ids=(
            "sample-1",
            "sample-2",
            "sample-3",
            "sample-4",
            "sample-5",
        ),
        session_ids=(
            "only-session",
            "only-session",
            "only-session",
            "only-session",
            "only-session",
        ),
    )

    with pytest.raises(
        ValueError,
        match="at least 2 sessions",
    ):

        evaluate_open_set_leave_one_session_out(dataset)
