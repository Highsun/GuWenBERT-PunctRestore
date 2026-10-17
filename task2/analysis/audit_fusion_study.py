"""Replay fusion from saved probabilities without loading any model or accessing CUDA."""
from pathlib import Path
import argparse
import hashlib
import json
import statistics
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from guwenbert_task2.common import ROOT,TASK,digest,load_config,read_json,read_rows,verify_task1,write_json
from guwenbert_task2.batch_study import checked_dataset
from guwenbert_task2.fusion_study import history_checks,select_weighting,study_sources
import numpy as np


def audit(config):
    report=TASK/"reports"/config["experiment"]
    if (report/"validation.json").exists():raise FileExistsError("Audit already exists")
    receipt=verify_task1(config);history=history_checks(config)
    selection=read_json(report/"selection.json")
    assert not selection["test_labels_used"]
    assert digest(report/"development.json")==selection["development_results_sha256"]
    artifacts,phases,changes={},{},{}
    for phase in ("development","test"):
        result=read_json(report/f"{phase}.json")
        assert result["history_checkpoints"]==history and result["source_files_sha256"]==study_sources()
        assert result["task1_unchanged"] and result["previous_experiments_unchanged"]
        if phase=="test":assert result["selection_sha256"]==digest(report/"selection.json")
        rows,dataset=checked_dataset(config,phase);assert dataset==result["dataset"]
        output=ROOT/"outputs/task2"/config["experiment"]/phase
        assert read_json(output/"results.json")==result
        for name,sha in result["source_files_sha256"].items():
            assert digest(ROOT/name)==sha
            assert digest(ROOT/"outputs/task2/source-snapshots"/config["experiment"]/name)==sha
        records={r["condition"]["id"]:r for r in result["conditions"]}
        data,window_data,boot={},{},{}
        draws=np.random.default_rng(config["seed"]+2).integers(0,len(rows),size=(config["bootstrap_samples"],len(rows)))
        for name,r in records.items():
            directory=output/name
            preds=read_rows(directory/"predictions.jsonl");windows=read_rows(directory/"windows.jsonl")
            data[name]=preds;window_data[name]=windows
            assert len(preds)==len(rows)
            matrix=np.zeros((8,8),dtype=np.int64);h=hashlib.sha256();counts=[]
            for doc,p in zip(rows,preds,strict=True):
                assert doc["document_id"]==p["document_id"] and doc["labels"]==p["gold"]
                assert len(doc["text"])==len(p["predicted"])
                g=np.asarray(p["gold"]);v=np.asarray(p["predicted"])
                assert np.all((0<=v)&(v<8))
                matrix+=np.bincount(g*8+v,minlength=64).reshape(8,8)
                counts.append([int(np.sum((g==v)&(g!=0))),int(np.sum(v!=0)),int(np.sum(g!=0))])
                h.update(len(v).to_bytes(8,"little"));h.update(bytes(p["predicted"]))
            assert matrix.tolist()==r["quality"]["overall"]["confusion_gold_rows_predicted_columns"]
            assert counts==r["document_counts"] and matrix.sum()==result["characters"]
            assert h.hexdigest()==r["prediction_sha256"]
            assert r["repeat_predictions_identical"] and r["repeat_probabilities_identical"]
            assert len(r["measurements"])==config["repetitions"]
            assert all(s["prediction_sha256"]==h.hexdigest() and s["probability_sha256"]==r["probability_sha256"] for s in r["measurements"])
            assert statistics.median(s["seconds"] for s in r["measurements"])==r["median_seconds"]
            assert statistics.stdev(s["seconds"] for s in r["measurements"])==r["stdev_seconds"]
            cursors=[0]*len(rows)
            for w in windows:
                d=w["document"]
                assert w["core_start"]==cursors[d]
                assert 0<=w["start"]<=w["core_start"]<w["core_end"]<=w["end"]<=len(rows[d]["text"])
                assert w["input_tokens"]==w["end"]-w["start"]+2<=r["condition"]["max_length"]
                cursors[d]=w["core_end"]
            assert cursors==[len(doc["text"]) for doc in rows]
            assert len(windows)==r["windows"]
            assert r["forward_calls"]==(len(windows)+r["condition"]["batch_size"]-1)//r["condition"]["batch_size"]
            total=np.asarray(counts)[draws].sum(axis=1);scores=2*total[:,0]/(total[:,1]+total[:,2]);boot[name]=scores
            assert np.allclose(np.quantile(scores,[.025,.975]),r["micro_f1_bootstrap_95ci"],atol=1e-12)
            for p in directory.iterdir():artifacts[str(p.relative_to(ROOT))]=digest(p)
        for w in config["windows"]:
            family=[r for r in result["conditions"] if r["condition"]["window_id"]==w["id"]]
            core_name=f"{w['id']}-core";control_name=f"{w['id']}-prob-core"
            assert data[core_name]==data[control_name]
            # Verify old core predictions agree with the accepted previous phase.
            previous=ROOT/"outputs/task2/cclue-batch-scaling-v0.3"/phase/f"{w['id']}-b{w['batch_size']}"/"predictions.jsonl"
            assert read_rows(previous)==data[core_name]
            cache=np.load(output/control_name/"probabilities.npz",allow_pickle=False)
            probs=cache["probabilities"];intervals=cache["windows"];offsets=cache["offsets"]
            assert probs.dtype==np.float32 and probs.shape[1]==8
            assert np.all(np.isfinite(probs)) and np.all((0<=probs)&(probs<=1))
            assert np.allclose(probs.sum(axis=1),1,atol=1e-6)
            assert len(offsets)==len(intervals)+1 and offsets[-1]==len(probs)
            token_lengths=cache["character_token_lengths"];doc_offsets=np.cumsum([0,*[len(doc["text"]) for doc in rows]])
            prefixes=[np.r_[0,np.cumsum(token_lengths[a:b])] for a,b in zip(doc_offsets[:-1],doc_offsets[1:])]
            h=hashlib.sha256()
            for interval,a,b in zip(intervals,offsets[:-1],offsets[1:],strict=True):
                h.update(np.asarray(interval,dtype="<i8").tobytes());h.update(np.asarray(probs[a:b],dtype="<f4").tobytes())
                assert b-a==interval[2]-interval[1]
            assert h.hexdigest()==records[control_name]["probability_sha256"]
            for r in family:
                c=r["condition"];name=c["id"]
                assert window_data[name]==window_data[core_name]
                if c["mode"]=="core":continue
                assert r["probability_sha256"]==h.hexdigest()
                score=[np.zeros((len(doc["text"]),8)) for doc in rows]
                weights=[np.zeros(len(doc["text"])) for doc in rows]
                coverage=[np.zeros(len(doc["text"]),dtype=np.int32) for doc in rows]
                for interval,a,b in zip(intervals,offsets[:-1],offsets[1:],strict=True):
                    d,start,end,core_start,core_end=map(int,interval)
                    values=probs[a:b]
                    if c["mode"]=="prob-core":
                        values=values[core_start-start:core_end-start];left,right=core_start,core_end;mass=np.ones(right-left)
                    else:
                        left,right=start,end
                        if c["mode"]=="uniform":mass=np.ones(end-start)
                        else:
                            prefix=prefixes[d];length=prefix[end]-prefix[start]
                            dist=np.minimum(prefix[start:end]-prefix[start],prefix[end]-prefix[start+1:end+1])+1
                            mass=(dist/((length+1)//2))**c["power"]
                    score[d][left:right]+=values*mass[:,None];weights[d][left:right]+=mass;coverage[d][left:right]+=1
                assert all(np.all(total>0) for total in weights)
                predicted=[(s/t[:,None]).argmax(axis=1).tolist() for s,t in zip(score,weights)]
                assert predicted==[p["predicted"] for p in data[name]],name
                saved=np.load(output/name/"aggregation.npz",allow_pickle=False)
                assert np.array_equal(saved["coverage"],np.concatenate(coverage))
                assert np.allclose(saved["weight_sums"],np.concatenate(weights),atol=1e-12)
                differences=recovered=introduced=0
                for doc,p,old,cov in zip(rows,predicted,data[core_name],coverage,strict=True):
                    g=np.asarray(doc["labels"]);v=np.asarray(p);ref=np.asarray(old["predicted"])
                    assert np.array_equal(v[cov==1],ref[cov==1])
                    differences+=int(np.sum(v!=ref));recovered+=int(np.sum((v==g)&(ref!=g)));introduced+=int(np.sum((v!=g)&(ref==g)))
                changes[f"{phase}/{name}"]={"different_characters_vs_core":differences,"corrected_characters":recovered,
                                             "newly_incorrect_characters":introduced,"all_changes_in_overlap":True}
        for name,r in records.items():
            for comp in r.get("comparisons",{}).values():
                ref=records[comp["reference"]]
                assert np.allclose(100*np.quantile(boot[name]-boot[comp["reference"]],[.025,.975]),comp["paired_95ci_pp"],atol=1e-12)
                assert abs(comp["latency_ratio"]-r["median_seconds"]/ref["median_seconds"])<1e-12
        if phase=="development":assert select_weighting(result["conditions"],config)==selection["selected"]
        log=read_rows(output/"progress.jsonl")
        assert sum(e["event"]=="measurement" for e in log)==len(records)*config["repetitions"]
        phases[phase]={"conditions":len(records),"measured_runs":len(records)*config["repetitions"],"documents":len(rows),"characters":result["characters"],
                       "probability_cache_replay_exact":True,"coverage_weight_sums_verified":True,"window_positions_identical":True,
                       "core_control_identical":True,"previous_core_predictions_identical":True,"independent_confusion_recomputation":True,"bootstrap_recomputed":True}
    bundle=read_json(report/"report-bundle.json")
    for name,sha in bundle["generated_files_sha256"].items():assert digest(report/name)==sha
    for name,sha in bundle["input_files_sha256"].items():assert digest(report/name)==sha
    assert digest(ROOT/bundle["analysis_script"]["path"])==bundle["analysis_script"]["sha256"]
    if "pilot_experiment" in config:
        pilot=read_json(TASK/"reports"/config["pilot_experiment"]/"development.json")
        for name,sha in pilot["source_files_sha256"].items():
            assert digest(ROOT/"outputs/task2/source-snapshots"/config["pilot_experiment"]/name)==sha
        current=read_json(report/"development.json")
        old={r["condition"]["id"]:r for r in pilot["conditions"]}
        for r in current["conditions"]:
            assert r["prediction_sha256"]==old[r["condition"]["id"]]["prediction_sha256"]
            assert r["probability_sha256"]==old[r["condition"]["id"]]["probability_sha256"]
    tests=read_json(TASK/config.get("unit_tests_record","reports/unit-tests-v0.4.json"));assert digest(ROOT/tests["junit_xml"]["path"])==tests["junit_xml"]["sha256"]
    suites=ET.parse(ROOT/tests["junit_xml"]["path"]).getroot().findall("testsuite")
    assert sum(int(s.attrib["tests"]) for s in suites)==46
    assert all(int(s.attrib["failures"])+int(s.attrib["errors"])==0 for s in suites)
    write_json(report/"validation.json",{"passed":True,"phases":phases,"task1_unchanged":True,"task1_files_checked":len(receipt["verified_files"]),
               "previous_experiments_unchanged":True,"unit_tests_passed":46,
               "figure_files_checked":sum(Path(name).suffix in (".png",".svg",".pdf") for name in bundle["generated_files_sha256"]),
               "pilot_predictions_and_probabilities_identical":True,"changes":changes,"artifact_sha256":artifacts,
               "audit_script":{"path":str(Path(__file__).resolve().relative_to(ROOT)),"sha256":digest(Path(__file__))}})
    print({"passed":True,"phases":phases,"artifact_files":len(artifacts)})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--config",default="configs/fusion-study-optimized.toml")
    args=parser.parse_args();audit(load_config(args.config))
