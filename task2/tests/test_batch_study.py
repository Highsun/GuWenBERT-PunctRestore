import pytest

from guwenbert_task2.batch_study import select_batches, variants


def entry(batch, seconds, f1=.7, stable=True):
    return {"condition": {"id": f"example-b{batch}", "window_id": "example", "batch_size": batch},
            "median_seconds": seconds, "repeat_predictions_identical": stable,
            "quality": {"overall": {"punctuation_micro": {"f1": f1}}}}


def config():
    return {"windows": [{"id": "example", "max_length": 222, "core_tokens": 192, "anchored": True}],
            "batch_sizes": [1, 4, 8, 16, 32, 64], "near_fastest_ratio": 1.03, "quality_drift_limit_pp": .02}


def test_prefer_smaller_batch_when_within_three_percent_of_fastest():
    records = [entry(16, 2), entry(32, 1.02), entry(64, 1)]
    selected = select_batches(records, config())["example"]
    assert selected["batch_size"] == 32
    assert selected["near_fastest_batches"] == [32, 64]


def test_exclude_quality_drift_and_unstable_predictions():
    records = [entry(16, 2), entry(32, 1, f1=.6997), entry(64, .8, stable=False)]
    selected = select_batches(records, config())["example"]
    assert selected["batch_size"] == 16
    assert selected["eligible_batches"] == [16]


def test_quality_improvements_also_count_as_batch_numerical_drift():
    records = [entry(16, 2), entry(64, 1, f1=.7003)]
    assert select_batches(records, config())["example"]["batch_size"] == 16


def test_require_stable_reference_or_eligible_other_batch():
    with pytest.raises(ValueError, match="No stable"):
        select_batches([entry(16, 2, stable=False)], config())


def test_test_conditions_only_include_locked_batches_and_matching_batch1():
    dev = variants(config())
    assert len(dev) == 8
    test = variants(config(), {"example": {"batch_size": 32}})
    assert [c["id"] for c in test] == ["example-b1", "example-b32", "serial256-b1", "serial400-b1"]
    assert test[1]["padded_limit"] == 224
    assert test[1]["max_length"] == 222
    assert test[1]["anchored"]


def test_selected_batch1_is_not_evaluated_twice():
    conditions = variants(config(), {"example": {"batch_size": 1}})
    assert len(conditions) == len({c["id"] for c in conditions}) == 3
