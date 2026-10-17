"""Independent cache replay of selection, local replacement and paired evaluation."""
from pathlib import Path
from bisect import bisect_left,bisect_right
import argparse,hashlib,json,math,statistics,sys
import xml.etree.ElementTree as ET

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from guwenbert_task2.common import ROOT,TASK,digest,load_config,read_json,read_rows,verify_task1,write_json
from guwenbert_task2.batch_study import checked_dataset
from guwenbert_task2.review_study import protect_previous,sources,select_policy
import numpy as np


def audit(config):
    report=TASK/"reports"/config["experiment"]
    if (report/"validation.json").exists():raise FileExistsError("Audit already exists")
    receipt=verify_task1(config);protect_previous(config);selection=read_json(report/"selection.json")
    assert not selection["test_labels_used"]
    assert digest(report/"development.json")==selection["development_results_sha256"]
    artifacts,phases,changes={},{},{}
    for phase in ("development","test"):
        result=read_json(report/f"{phase}.json");output=ROOT/"outputs/task2"/config["experiment"]/phase
        assert result==read_json(output/"results.json") and result["source_files_sha256"]==sources()
        assert result["previous_checkpoint_sha256"]==protect_previous(config)
        if phase=="test":assert result["selection_sha256"]==digest(report/"selection.json")
        for name,sha in result["source_files_sha256"].items():
            assert digest(ROOT/name)==sha and digest(ROOT/"outputs/task2/source-snapshots"/config["experiment"]/name)==sha
        rows,dataset=checked_dataset(config,phase);assert dataset==result["dataset"]
        cache=np.load(output/"baseline/first-pass.npz",allow_pickle=False)
        probs,windows,offsets=cache["probabilities"],cache["windows"],cache["offsets"]
        assert np.all(np.isfinite(probs)) and np.all((probs>=0)&(probs<=1)) and np.allclose(probs.sum(1),1,atol=1e-6)
        doc_offsets=np.cumsum([0,*[len(d["text"]) for d in rows]])
        token_lengths=cache["character_token_lengths"]
        prefix=[np.r_[0,np.cumsum(token_lengths[a:b])] for a,b in zip(doc_offsets[:-1],doc_offsets[1:])]
        sums=[np.zeros(len(d["text"])) for d in rows];scores=[np.zeros((len(d["text"]),8)) for d in rows]
        low=[np.full(len(d["text"]),8) for d in rows];high=[np.full(len(d["text"]),-1) for d in rows]
        first_hash=hashlib.sha256();cuts=[[] for _ in rows]
        for interval,a,b in zip(windows,offsets[:-1],offsets[1:],strict=True):
            d,start,end,core_start,core_end=map(int,interval);p=probs[a:b]
            first_hash.update(np.asarray(interval,dtype="<i8").tobytes());first_hash.update(np.asarray(p,dtype="<f4").tobytes())
            assert len(p)==end-start
            length=prefix[d][end]-prefix[d][start]
            weight=(np.minimum(prefix[d][start:end]-prefix[d][start],prefix[d][end]-prefix[d][start+1:end+1])+1)/((length+1)//2)
            scores[d][start:end]+=p*weight[:,None];sums[d][start:end]+=weight
            votes=p.argmax(1);low[d][start:end]=np.minimum(low[d][start:end],votes);high[d][start:end]=np.maximum(high[d][start:end],votes)
            if core_end<len(rows[d]["text"]):cuts[d].append(core_end)
        posterior=[s/m[:,None] for s,m in zip(scores,sums)];baseline=[p.argmax(1) for p in posterior]
        expected_candidates=[]
        for d,positions in enumerate(cuts):
            ordered=sorted(set(positions));random_values=np.random.default_rng(config["seed"]+d).random(len(ordered))
            for cut,random_value in zip(ordered,random_values,strict=True):
                budget=config["review_max_length"]-2;origin=max(0,min(prefix[d][cut]-budget//2,prefix[d][-1]-budget))
                start=bisect_left(prefix[d],origin);end=bisect_right(prefix[d],prefix[d][start]+budget)-1
                left=max(start,cut-config["review_radius"]);right=min(end,cut+config["review_radius"])
                top=np.partition(posterior[d][left:right],-2,axis=1)[:,-2:]
                expected_candidates.append({"document":d,"cut":cut,"start":start,"end":end,"band_start":left,"band_end":right,
                    "uncertainty":float(np.mean(1-(top[:,1]-top[:,0]))),"disagreement":float(np.mean(low[d][left:right]!=high[d][left:right])),"random":float(random_value)})
        records={r["condition"]["id"]:r for r in result["conditions"]}
        original=read_rows(ROOT/"outputs/task2/cclue-window-fusion-v0.4-optimized"/phase/"original256-weighted1/predictions.jsonl")
        assert [p.tolist() for p in baseline]==[d["predicted"] for d in original]
        draws=np.random.default_rng(config["seed"]+2).integers(0,len(rows),size=(config["bootstrap_samples"],len(rows)))
        boot={};quotas={}
        for name,r in records.items():
            path=output/name;c=r["condition"];saved=read_rows(path/"predictions.jsonl");trace=read_rows(path/"windows.jsonl")
            assert len(saved)==len(rows);matrix=np.zeros((8,8),dtype=np.int64);counts=[];h=hashlib.sha256()
            for doc,p in zip(rows,saved,strict=True):
                assert doc["document_id"]==p["document_id"] and doc["labels"]==p["gold"]
                assert len(doc["text"])==len(p["predicted"])
                gold,value=np.asarray(p["gold"]),np.asarray(p["predicted"])
                assert np.all((value>=0)&(value<8));matrix+=np.bincount(gold*8+value,minlength=64).reshape(8,8)
                counts.append([int(np.sum((gold==value)&(gold!=0))),int(np.sum(value!=0)),int(np.sum(gold!=0))])
                h.update(len(value).to_bytes(8,"little"));h.update(bytes(p["predicted"]))
            assert matrix.tolist()==r["quality"]["overall"]["confusion_gold_rows_predicted_columns"] and matrix.sum()==result["characters"]
            assert counts==r["document_counts"] and h.hexdigest()==r["prediction_sha256"]
            assert r["repeat_predictions_identical"] and r["repeat_probabilities_identical"]
            assert len(r["measurements"])==config["repetitions"]
            assert all(s["prediction_sha256"]==h.hexdigest() for s in r["measurements"])
            assert statistics.median(s["seconds"] for s in r["measurements"])==r["median_seconds"]
            totals=np.asarray(counts)[draws].sum(1);boot[name]=2*totals[:,0]/(totals[:,1]+totals[:,2])
            assert np.allclose(np.quantile(boot[name],[.025,.975]),r["micro_f1_bootstrap_95ci"],atol=1e-12)
            cursor=[0]*len(rows)
            for win in trace:
                d=win["document"];assert win["core_start"]==cursor[d];cursor[d]=win["core_end"]
            assert cursor==[len(d["text"]) for d in rows]
            if c["mode"]!="serial":
                assert r["first_probability_sha256"]==first_hash.hexdigest()
                assert all(s["first_probability_sha256"]==first_hash.hexdigest() for s in r["measurements"])
                assert trace==read_rows(output/"baseline/windows.jsonl")
                assert r["candidate_count"]==len(expected_candidates)
                candidates=read_rows(path/"candidates.jsonl");chosen=read_rows(path/"selected.jsonl")
                if c["mode"]=="baseline":assert candidates==[]
                else:
                    assert len(candidates)==len(expected_candidates)
                    for a,b in zip(candidates,expected_candidates,strict=True):
                        for key in ("document","cut","start","end","band_start","band_end"):assert a[key]==b[key]
                        for key in ("uncertainty","disagreement","random"):assert abs(a[key]-b[key])<1e-12
                selected=[];by_doc={}
                if c["mode"]=="review":
                    for cand in expected_candidates:by_doc.setdefault(cand["document"],[]).append(cand)
                    for values in by_doc.values():
                        if c["selector"]=="all":selected.extend(values)
                        else:
                            ordered=sorted(values,key=lambda x:(-x[c["selector"]],-x["uncertainty"] if c["selector"]=="disagreement" else 0,x["cut"]))
                            selected.extend(ordered[:math.floor(len(values)*c["fraction"])])
                expected_keys={(x["document"],x["cut"]) for x in selected}
                assert {(x["document"],x["cut"]) for x in chosen}==expected_keys
                assert len(chosen)==r["review_windows"]
                per_doc=[sum(x["document"]==d for x in chosen) for d in range(len(rows))]
                if c["mode"]=="review" and c["selector"]!="all":
                    if c["fraction"] in quotas:assert quotas[c["fraction"]]==per_doc
                    else:quotas[c["fraction"]]=per_doc
                cached=np.load(path/"review.npz",allow_pickle=False);rv=cached["probabilities"];rw=cached["windows"];ro=cached["offsets"]
                assert len(chosen)==len(rw) and len(ro)==len(rw)+1 and ro[-1]==len(rv)
                if len(rv):assert np.all(np.isfinite(rv)) and np.all((rv>=0)&(rv<=1)) and np.allclose(rv.sum(1),1,atol=1e-6)
                predicted=[x.copy() for x in baseline];owners=[np.full(len(doc["text"]),-1,dtype=np.int64) for doc in rows]
                distance=[np.full(len(doc["text"]),np.inf) for doc in rows];rh=hashlib.sha256()
                for cand,win,a,b in zip(chosen,rw,ro[:-1],ro[1:],strict=True):
                    d,start,end,left,right,cut=map(int,win)
                    assert [d,start,end,left,right,cut]==[cand[k] for k in ("document","start","end","band_start","band_end","cut")]
                    assert prefix[d][end]-prefix[d][start]+2<=config["review_max_length"]
                    p=rv[a:b];assert len(p)==end-start
                    rh.update(np.asarray(win,dtype="<i8").tobytes());rh.update(np.asarray(p,dtype="<f4").tobytes())
                    for i in range(left,right):
                        dist=abs(prefix[d][i]-prefix[d][cut])
                        if (dist,cut)<(distance[d][i],owners[d][i]):
                            predicted[d][i]=p[i-start].argmax();distance[d][i]=dist;owners[d][i]=cut
                assert rh.hexdigest()==r["review_probability_sha256"]
                assert all(s["review_probability_sha256"]==rh.hexdigest() for s in r["measurements"])
                assert np.array_equal(np.concatenate(owners),cached["owners"])
                assert [x.tolist() for x in predicted]==[x["predicted"] for x in saved]
                assert r["first_forward_calls"]==(len(windows)+config["batch_size"]-1)//config["batch_size"]
                assert r["review_forward_calls"]==(len(chosen)+config["batch_size"]-1)//config["batch_size"]
                assert r["forward_calls"]==r["first_forward_calls"]+r["review_forward_calls"]
                touched=sum(int(np.sum(o>=0)) for o in owners);assert touched==r["reviewed_characters"]
                before_errors=after_errors=corrected=new_errors=changed=0
                for doc,old,new,owner in zip(rows,baseline,predicted,owners,strict=True):
                    gold=np.asarray(doc["labels"]);mask=owner>=0
                    assert np.array_equal(old[~mask],new[~mask])
                    before_errors+=int(np.sum((old!=gold)&mask));after_errors+=int(np.sum((new!=gold)&mask))
                    corrected+=int(np.sum((old!=gold)&(new==gold)));new_errors+=int(np.sum((old==gold)&(new!=gold)));changed+=int(np.sum(old!=new))
                changes[f"{phase}/{name}"]={"reviewed_characters":touched,"baseline_errors_in_reviewed_bands":before_errors,
                    "errors_after_review":after_errors,"corrected_characters":corrected,"newly_incorrect_characters":new_errors,
                    "changed_characters":changed,"unreviewed_characters_unchanged":True}
            for f in path.iterdir():artifacts[str(f.relative_to(ROOT))]=digest(f)
        for name,r in records.items():
            for comp in r["comparisons"].values():
                ref=records[comp["reference"]]
                assert np.allclose(100*np.quantile(boot[name]-boot[comp["reference"]],[.025,.975]),comp["paired_95ci_pp"],atol=1e-12)
                assert abs(comp["latency_ratio"]-r["median_seconds"]/ref["median_seconds"])<1e-12
        if phase=="development":assert select_policy(result["conditions"],config)==selection["selected"]
        progress=read_rows(output/"progress.jsonl");assert sum(x["event"]=="measurement" for x in progress)==len(records)*config["repetitions"]
        phases[phase]={"conditions":len(records),"measurements":len(records)*config["repetitions"],"documents":len(rows),"characters":result["characters"],
            "first_pass_matches_previous_fusion":True,"selection_replayed":True,"review_predictions_replayed":True,"same_budget_random_verified":True,
            "owner_assignment_verified":True,"unreviewed_characters_unchanged":True,"independent_confusion_recomputation":True,"bootstrap_recomputed":True}
    tests=read_json(TASK/"reports/unit-tests-v0.5.json");assert digest(ROOT/tests["junit_xml"]["path"])==tests["junit_xml"]["sha256"]
    suites=ET.parse(ROOT/tests["junit_xml"]["path"]).getroot().findall("testsuite")
    assert sum(int(s.attrib["tests"]) for s in suites)==56 and all(int(s.attrib["failures"])+int(s.attrib["errors"])==0 for s in suites)
    write_json(report/"validation.json",{"passed":True,"phases":phases,"task1_unchanged":True,"task1_files_checked":len(receipt["verified_files"]),
        "previous_experiments_unchanged":True,"unit_tests_passed":56,"changes":changes,"artifact_sha256":artifacts,
        "audit_script":{"path":str(Path(__file__).resolve().relative_to(ROOT)),"sha256":digest(Path(__file__))}})
    print({"passed":True,"phases":phases,"artifact_files":len(artifacts)})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--config",default="configs/review-study.toml")
    args=parser.parse_args();audit(load_config(args.config))
