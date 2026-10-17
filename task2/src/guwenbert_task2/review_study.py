from __future__ import annotations

from dataclasses import asdict
import csv
import hashlib
import json
import shutil
import statistics

import numpy as np

from .batch_study import checked_dataset, prediction_digest
from .common import ROOT,TASK,assert_protected,digest,read_json,source_hashes,write_json,write_rows
from .context_study import bootstrap_scores,document_counts
from .engine import infer
from .fusion import ProbabilityBackend
from .fusion_study import probability_digest,save_cache
from .metrics import quality
from .review import infer_review


def protect_previous(config):
    path=ROOT/config["protected_checkpoint"];saved=read_json(path)
    if not saved["passed"]:raise ValueError("Previous experiment is not verified")
    for name,sha in saved["files_sha256"].items():
        if digest(ROOT/name)!=sha:raise ValueError(f"Protected previous file changed: {name}")
    return digest(path)


def sources():
    return {**source_hashes(),**{str(p.relative_to(ROOT)):digest(p) for p in sorted((TASK/"tests").glob("*.py"))}}


def conditions(config,selection=None):
    result=[{"id":"baseline","mode":"baseline","selector":"uncertainty","fraction":0},
            {"id":"score-only","mode":"score-only","selector":"uncertainty","fraction":0}]
    if selection is None:
        for selector in config["selectors"]:
            for fraction in config["fractions"]:
                result.append({"id":f"{selector}-{int(fraction*100)}","mode":"review","selector":selector,"fraction":fraction})
    else:
        chosen=selection["selected"]["condition"]
        result.extend([chosen,{"id":f"random-{int(chosen['fraction']*100)}","mode":"review","selector":"random","fraction":chosen["fraction"]}])
    result.extend([{"id":"all-review","mode":"review","selector":"all","fraction":1},
                   {"id":"serial","mode":"serial","selector":"all","fraction":0}])
    return result


def select_policy(records,config):
    base=next(r for r in records if r["condition"]["id"]=="baseline")
    candidates=[r for r in records if r["condition"]["selector"] in ("uncertainty","disagreement")
                and r["condition"]["mode"]=="review" and r["repeat_predictions_identical"]]
    eligible=[r for r in candidates if r["median_seconds"]<=base["median_seconds"]*config["selection_slowdown_limit"]]
    within_cost=bool(eligible)
    if not eligible:eligible=candidates
    if not eligible:raise ValueError("No stable selection candidate")
    best=max(r["quality"]["overall"]["punctuation_micro"]["f1"] for r in eligible)
    near=[r for r in eligible if 100*(best-r["quality"]["overall"]["punctuation_micro"]["f1"])<=config["selection_quality_tolerance_pp"]]
    choice=min(near,key=lambda r:(r["median_seconds"],r["condition"]["fraction"],r["condition"]["selector"]))
    return {"condition":choice["condition"],"development_micro_f1":choice["quality"]["overall"]["punctuation_micro"]["f1"],
            "cost_limit_satisfied":within_cost,"eligible_conditions":[r["condition"]["id"] for r in eligible]}


def review_digest(out):
    h=hashlib.sha256()
    for c,p in zip(out["selected"],out["review_probabilities"],strict=True):
        h.update(np.asarray([c.document,c.start,c.end,c.band_start,c.band_end,c.cut],dtype="<i8").tobytes())
        h.update(np.asarray(p,dtype="<f4").tobytes())
    return h.hexdigest()


def save_review_cache(path,out):
    selected=out["selected"]
    np.savez_compressed(path,probabilities=np.concatenate(out["review_probabilities"]) if selected else np.empty((0,8),dtype=np.float32),
                        offsets=np.cumsum([0,*[c.end-c.start for c in selected]]),
                        windows=np.asarray([[c.document,c.start,c.end,c.band_start,c.band_end,c.cut] for c in selected],dtype=np.int64).reshape(-1,6),
                        owners=np.concatenate(out["owners"]))


def run_phase(config,receipt,phase):
    if phase not in ("development","test"):raise ValueError("Invalid phase")
    if not 2 < config["review_max_length"] <= 512 or (config["review_max_length"]+7)//8*8>512 or config["review_radius"]<1:
        raise ValueError("Invalid review window or radius")
    directory=ROOT/"outputs/task2"/config["experiment"]/phase;report=TASK/"reports"/config["experiment"]
    if directory.exists():raise FileExistsError("Phase exists; choose new experiment ID")
    previous=protect_previous(config);initial=sources();selection=None
    if phase=="test":
        selection=read_json(report/"selection.json")
        if selection["config"]!=config or selection["source_files_sha256"]!=initial:raise ValueError("Selection configuration/source changed")
        if digest(report/"development.json")!=selection["development_results_sha256"]:raise ValueError("Selection development file changed")
        selection_sha=digest(report/"selection.json")
    variants=conditions(config,selection);rows,dataset=checked_dataset(config,phase);texts=[r["text"] for r in rows]
    backend=ProbabilityBackend(config,receipt);directory.mkdir(parents=True)
    write_json(directory/"plan.json",{"phase":phase,"config":config,"conditions":variants,"dataset":dataset,"source_files_sha256":initial,
                                      "previous_checkpoint_sha256":previous,"selection_sha256":None if selection is None else selection_sha})
    samples,records={},{ };rng=np.random.default_rng(config["seed"]+(phase=="test"));warm=rng.permutation(len(variants)).tolist();orders=[]

    def once(c):
        backend.max_length=max(config["max_length"],(config["review_max_length"]+7)//8*8)
        if c["mode"]=="serial":return infer(texts,backend,config["max_length"],config["core_tokens"],"adaptive",1)
        return infer_review(texts,backend,config,c)

    with (directory/"progress.jsonl").open("x",encoding="utf-8",buffering=1) as progress:
        def log(v):
            line=json.dumps(v,ensure_ascii=False);progress.write(line+"\n");print(line,flush=True)
        log({"event":"plan","phase":phase,"conditions":len(variants),"documents":len(rows)})
        for i in warm:
            for _ in range(config["warmup_runs"]):once(variants[i])
            log({"event":"warmup","condition":variants[i]["id"]})
        for repeat in range(config["repetitions"]):
            order=rng.permutation(len(variants)).tolist();orders.append([variants[i]["id"] for i in order])
            for i in order:
                c=variants[i];name=c["id"];out=once(c);fingerprint=prediction_digest(out["labels"])
                first_sha=probability_digest(out) if c["mode"]!="serial" else None
                second_sha=review_digest(out) if c["mode"]!="serial" else None
                samples.setdefault(name,[]).append({"seconds":out["seconds"],**out["profile"],"prediction_sha256":fingerprint,
                                                   "first_probability_sha256":first_sha,"review_probability_sha256":second_sha})
                log({"event":"measurement","round":repeat+1,"condition":name,"seconds":round(out["seconds"],5),
                     "review_windows":out.get("review_windows",0),"prediction_sha256":fingerprint})
                if repeat!=config["repetitions"]-1:continue
                path=directory/name
                write_rows(path/"predictions.jsonl",[{"document_id":r["document_id"],"gold":r["labels"],"predicted":p}
                                                     for r,p in zip(rows,out["labels"],strict=True)])
                write_rows(path/"windows.jsonl",out["windows"])
                if c["mode"]!="serial":
                    write_rows(path/"candidates.jsonl",[asdict(c) for c in out["candidates"]])
                    write_rows(path/"selected.jsonl",[asdict(c) for c in out["selected"]])
                    save_review_cache(path/"review.npz",out)
                    if c["mode"]=="baseline":save_cache(path/"first-pass.npz",out)
                records[name]={"condition":c,"quality":quality(rows,out,config["boundary_margin"]),
                               "document_counts":document_counts([r["labels"] for r in rows],out["labels"]).tolist(),
                               "prediction_sha256":fingerprint,"first_probability_sha256":first_sha,"review_probability_sha256":second_sha,
                               "windows":len(out["windows"]),"forward_calls":out["forward_calls"],
                               "first_forward_calls":out.get("first_forward_calls",out["forward_calls"]),"review_forward_calls":out.get("review_forward_calls",0),
                               "candidate_count":out.get("candidate_count",0),"review_windows":out.get("review_windows",0),
                               "reviewed_characters":out.get("reviewed_characters",0),
                               "processed_content_tokens":out["processed_content_tokens"],"content_tokens":out["content_tokens"]}
    draws=np.random.default_rng(config["seed"]+2).integers(0,len(rows),size=(config["bootstrap_samples"],len(rows)))
    boot={name:bootstrap_scores(np.asarray(r["document_counts"]),draws) for name,r in records.items()}
    for name,r in records.items():
        ss=samples[name];seconds=[s["seconds"] for s in ss]
        r.update({"measurements":ss,"median_seconds":statistics.median(seconds),"stdev_seconds":statistics.stdev(seconds),
                  "characters_per_second":sum(map(len,texts))/statistics.median(seconds),"peak_allocated_bytes":max(s["peak_allocated_bytes"] for s in ss),
                  "repeat_predictions_identical":len({s["prediction_sha256"] for s in ss})==1,
                  "repeat_probabilities_identical":len({(s["first_probability_sha256"],s["review_probability_sha256"]) for s in ss})==1,
                  "micro_f1_bootstrap_95ci":np.quantile(boot[name],[.025,.975]).tolist()})
    for name,r in records.items():
        r["comparisons"]={}
        references=[("baseline","baseline"),("serial","serial")]
        c=r["condition"]
        if c["mode"]=="review" and c["selector"] in ("uncertainty","disagreement"):
            references.append(("random",f"random-{int(c['fraction']*100)}"))
        for key,ref_name in references:
            ref=records[ref_name]
            r["comparisons"][key]={"reference":ref_name,"latency_ratio":r["median_seconds"]/ref["median_seconds"],
                                   "speedup":ref["median_seconds"]/r["median_seconds"],
                                   "delta_micro_f1_pp":100*(r["quality"]["overall"]["punctuation_micro"]["f1"]-ref["quality"]["overall"]["punctuation_micro"]["f1"]),
                                   "paired_95ci_pp":(100*np.quantile(boot[name]-boot[ref_name],[.025,.975])).tolist()}
        if c["mode"]!="serial":
            if r["first_probability_sha256"]!=records["baseline"]["first_probability_sha256"]:raise AssertionError("First pass probabilities changed")
            if (directory/name/"windows.jsonl").read_bytes()!=(directory/"baseline/windows.jsonl").read_bytes():raise AssertionError("First pass windows changed")
    if records["score-only"]["prediction_sha256"]!=records["baseline"]["prediction_sha256"]:raise AssertionError("Scoring changed predictions")
    old=ROOT/"outputs/task2/cclue-window-fusion-v0.4-optimized"/phase/"original256-weighted1/predictions.jsonl"
    if (directory/"baseline/predictions.jsonl").read_bytes()!=old.read_bytes():raise AssertionError("Baseline differs from previous fusion")
    assert_protected(receipt)
    if protect_previous(config)!=previous or sources()!=initial:raise RuntimeError("Source/protected previous files changed")
    if selection is not None and digest(report/"selection.json")!=selection_sha:raise RuntimeError("Locked selection changed")
    result={"schema_version":1,"date":"2026-10-17","phase":phase,"config":config,"dataset":dataset,"documents":len(rows),"characters":sum(map(len,texts)),
            "conditions":[records[c["id"]] for c in variants],"environment":backend.environment,"warmup_order":[variants[i]["id"] for i in warm],
            "measurement_orders":orders,"source_files_sha256":initial,"previous_checkpoint_sha256":previous,"task1_unchanged":True,"previous_experiments_unchanged":True,
            "selection_sha256":None if selection is None else selection_sha,"test_labels_used_for_selection":False,
            "bootstrap":{"samples":len(draws),"seed":config["seed"]+2,"paired":True,"unit":"anonymous document group","multiple_comparison_adjustment":False},
            "timing":"warm model/tokenizer; encode, first pass, fusion, scoring/select, batch review, replace, render; exclude model/data load, provenance, scoring metrics, hashing, cache serialization and result I/O"}
    write_json(directory/"results.json",result);write_json(report/f"{phase}.json",result);export_table(result,report/f"{phase}.csv")
    snapshot=ROOT/"outputs/task2/source-snapshots"/config["experiment"]
    for name,sha in initial.items():
        p=snapshot/name
        if p.exists():
            if digest(p)!=sha:raise ValueError("Snapshot differs")
        else:p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,p)
    if phase=="development":
        write_json(report/"selection.json",{"config":config,"selected":select_policy(result["conditions"],config),"source_files_sha256":initial,
                   "development_results_sha256":digest(report/"development.json"),"test_labels_used":False})
    return {"experiment":config["experiment"],"phase":phase,"conditions":len(variants)}


def export_table(result,path):
    fields=["condition","micro_f1_percent","macro_f1_percent","sentence_f1_percent","edge_micro_f1_percent","median_seconds","stdev_seconds",
            "review_windows","candidate_count","reviewed_characters","forward_calls","selection_median_seconds","second_pass_median_seconds",
            "delta_vs_baseline_pp","latency_ratio_vs_baseline","speedup_vs_serial"]
    with path.open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in result["conditions"]:
            q=r["quality"]["overall"];comp=r["comparisons"]["baseline"]
            writer.writerow({"condition":r["condition"]["id"],"micro_f1_percent":q["punctuation_micro"]["f1"]*100,
                "macro_f1_percent":q["punctuation_macro_f1"]*100,"sentence_f1_percent":q["sentence_boundary"]["f1"]*100,
                "edge_micro_f1_percent":r["quality"]["window_edges"]["punctuation_micro"]["f1"]*100,
                **{k:r[k] for k in fields[5:11]},"selection_median_seconds":statistics.median(s.get("selection_seconds",0) for s in r["measurements"]),
                "second_pass_median_seconds":statistics.median(s.get("second_pass_seconds",0) for s in r["measurements"]),
                "delta_vs_baseline_pp":comp["delta_micro_f1_pp"],"latency_ratio_vs_baseline":comp["latency_ratio"],"speedup_vs_serial":r["comparisons"]["serial"]["speedup"]})
