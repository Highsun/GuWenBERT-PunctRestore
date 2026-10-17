from dataclasses import replace
import numpy as np
import pytest

from guwenbert_task2.review import Candidate,apply_reviews,centered_window,choose_candidates,first_posterior
from guwenbert_task2.review_study import conditions,select_policy
from guwenbert_task2.windows import EncodedText,Window


def candidate(cut,doc=0):
    return Candidate(doc,cut,0,20,max(0,cut-2),min(20,cut+2),cut/20,1-cut/20,cut/21)


def test_each_document_has_exact_floor_budget_independent_of_input_order():
    cs=[candidate(i,doc) for doc in (0,1) for i in range(1,11)]
    selected=choose_candidates(cs,"uncertainty",.25)
    assert [(c.document,c.cut) for c in selected]==[(0,9),(0,10),(1,9),(1,10)]
    assert choose_candidates(list(reversed(cs)),"uncertainty",.25)==selected
    assert choose_candidates(cs,"random",.25)==choose_candidates(cs,"random",.25)


def test_disagreement_ties_use_uncertainty_then_cut():
    cs=[replace(candidate(i),disagreement=.5,uncertainty=.5) for i in (5,3,8)]
    assert choose_candidates(cs,"disagreement",.5)[0].cut==3


@pytest.mark.parametrize("cut",[1,50,199])
def test_review_windows_center_and_shift_within_token_budget(cut):
    e=EncodedText.build("甲"*200,lambda c:(1,))
    w=centered_window(e,0,cut,64,8)
    assert w.input_tokens(e)<=64
    assert w.start<=w.core_start<w.core_end<=w.end
    assert w.start<=cut<=w.end
    if cut==50:assert w.start==19 and w.end==81


def test_review_handles_multisubtoken_characters_without_splitting_them():
    e=EncodedText.build("甲乙丙丁戊己庚辛",lambda c:(1,2,3) if c=="乙" else (1,))
    w=centered_window(e,0,4,10,2)
    assert w.input_tokens(e)<=10 and w.core_start<=4<=w.core_end


def test_review_only_changes_bands_and_uses_order_independent_closest_owner():
    e=EncodedText.build("甲"*20,lambda c:(1,))
    cs=[candidate(8),candidate(10)]
    p=[]
    for label in (1,2):
        x=np.zeros((20,8));x[:,label]=1;p.append(x)
    a,owners=apply_reviews([e],[[0]*20],cs,p)
    b,_=apply_reviews([e],[[0]*20],list(reversed(cs)),list(reversed(p)))
    assert a==b
    assert a[0][:6]==[0]*6 and a[0][12:]==[0]*8
    assert owners[0][9]==8
    assert a[0][8]==1 and a[0][10]==2


def test_first_pass_matches_single_window_and_detects_view_disagreement():
    e=EncodedText.build("甲"*4,lambda c:(1,))
    ws=[Window(0,0,3,0,2),Window(0,1,4,2,4)]
    ps=[]
    for label in (1,2):
        p=np.zeros((3,8));p[:,label]=1;ps.append(p)
    posterior,disputes=first_posterior([e],ws,ps,True)
    assert disputes[0].tolist()==[False,True,True,False]
    assert posterior[0].argmax(axis=1).tolist()==[1,1,2,2]


def test_test_phase_has_matched_random_policy_and_six_conditions():
    c={"selectors":["uncertainty","disagreement","random"],"fractions":[.1,.25,.5]}
    assert len(conditions(c))==13
    selected={"selected":{"condition":{"id":"disagreement-25","mode":"review","selector":"disagreement","fraction":.25}}}
    ids=[x["id"] for x in conditions(c,selected)]
    assert ids==["baseline","score-only","disagreement-25","random-25","all-review","serial"]


def test_policy_selection_excludes_expensive_and_random_candidates():
    def row(name,selector,seconds,f1,mode="review"):
        return {"condition":{"id":name,"selector":selector,"fraction":.25,"mode":mode},"median_seconds":seconds,
                "repeat_predictions_identical":True,"quality":{"overall":{"punctuation_micro":{"f1":f1}}}}
    rows=[row("baseline","uncertainty",2,.7,"baseline"),row("a","uncertainty",2.2,.701),
          row("b","disagreement",3,.72),row("r","random",2.1,.73)]
    config={"selection_slowdown_limit":1.25,"selection_quality_tolerance_pp":.02}
    assert select_policy(rows,config)["condition"]["id"]=="a"
