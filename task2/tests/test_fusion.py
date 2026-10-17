import numpy as np
import pytest

from guwenbert_task2.fusion import center_weights, combine_probabilities
from guwenbert_task2.windows import EncodedText, Window
from guwenbert_task2.fusion_study import select_weighting, conditions


def examples():
    encoded = [EncodedText.build("甲乙丙丁", lambda c: (1,))]
    windows = [Window(0, 0, 3, 0, 2), Window(0, 1, 4, 2, 4)]
    a = np.zeros((3, 8)); b = np.zeros((3, 8))
    a[:, 1] = .8; a[:, 0] = .2
    b[:, 2] = .9; b[:, 0] = .1
    return encoded, windows, [a, b]


def test_core_control_ignores_overlapping_noncore_predictions():
    e,w,p = examples()
    labels,sums,coverage = combine_probabilities(e,w,p,"prob-core")
    assert labels == [[1,1,2,2]]
    assert coverage[0].tolist() == [1,1,1,1]
    assert sums[0].tolist() == [1,1,1,1]


def test_uniform_average_uses_both_windows_and_counts_each_output_once():
    e,w,p = examples()
    labels,sums,coverage = combine_probabilities(e,w,p,"uniform")
    assert labels == [[1,2,2,2]]
    assert coverage[0].tolist() == [1,2,2,1]
    assert sums[0].tolist() == [1,2,2,1]


def test_center_weighting_favors_window_interior_over_edge():
    e,w,p = examples()
    labels,_,_ = combine_probabilities(e,w,p,"weighted",1)
    assert labels == [[1,1,2,2]]
    assert center_weights(e[0],w[0],1).tolist() == [.5,1,.5]
    assert center_weights(e[0],w[0],2).tolist() == [.25,1,.25]


def test_multisubtoken_weights_use_complete_character_boundaries():
    e = EncodedText.build("甲乙丙",lambda c: (1,2) if c=="乙" else (1,))
    assert center_weights(e,Window(0,0,3,0,3),1).tolist() == [.5,1,.5]


@pytest.mark.parametrize("mode",["prob-core","uniform","weighted"])
def test_single_window_prediction_unchanged_by_weight_normalization(mode):
    e,w,p=examples()
    labels,_,_=combine_probabilities(e,[Window(0,0,4,0,4)],[np.vstack([p[0],p[0][0]])],mode)
    assert labels == [[1,1,1,1]]


def test_missing_or_duplicate_core_coverage_rejected():
    e,w,p=examples()
    with pytest.raises(AssertionError,match="Uncovered"):
        combine_probabilities(e,w[:1],p[:1],"prob-core")
    with pytest.raises(AssertionError,match="exactly once"):
        combine_probabilities(e,[*w,w[0]],[*p,p[0]],"prob-core")


def test_invalid_probability_distribution_rejected():
    e,w,p=examples();p[0][0,0]=2
    with pytest.raises(ValueError,match="distribution"):
        combine_probabilities(e,w,p,"uniform")


def test_window_order_does_not_change_non_tied_fusion_predictions():
    e,w,p=examples()
    a=combine_probabilities(e,w,p,"weighted",2)[0]
    b=combine_probabilities(e,list(reversed(w)),list(reversed(p)),"weighted",2)[0]
    assert a==b


def test_weight_selection_prefers_faster_within_quality_tolerance():
    config={"windows":[{"id":"example"}],"selection_quality_tolerance_pp":.02}
    def entry(power,f1,seconds):
        return {"condition":{"window_id":"example","mode":"weighted","power":power,"id":str(power)},
                "repeat_predictions_identical":True,"quality":{"overall":{"punctuation_micro":{"f1":f1}}},
                "median_seconds":seconds}
    assert select_weighting([entry(1,.7,1),entry(2,.7001,2)],config)["example"]["power"]==1
    assert select_weighting([entry(1,.7,1),entry(2,.7003,2)],config)["example"]["power"]==2


def test_test_stage_only_uses_locked_center_weight_power():
    config={"windows":[{"id":"example","max_length":222,"core_tokens":192,"batch_size":32,"anchored":True}],
            "weighted_powers":[1,2]}
    assert len(conditions(config))==6
    selected={"selected":{"example":{"power":2}}}
    result=conditions(config,selected)
    assert len(result)==5
    assert [c["power"] for c in result if c["mode"]=="weighted"]==[2]
