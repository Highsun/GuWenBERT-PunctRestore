from __future__ import annotations

import csv
import hashlib
import json
import shutil
import statistics

import numpy as np

from .batch_study import checked_dataset, prediction_digest
from .common import ROOT, TASK, assert_protected, digest, read_json, source_hashes, write_json, write_rows
from .context_study import bootstrap_scores, document_counts
from .engine import infer
from .fusion import ProbabilityBackend, infer_probabilities
from .metrics import quality


def history_checks(config):
    receipts = {}
    for name in config["accepted_checkpoints"]:
        checkpoint = read_json(ROOT / name)
        if checkpoint["status"] != "accepted_by_user":
            raise ValueError("Previous experiment not accepted")
        for path, sha in checkpoint["files_sha256"].items():
            if digest(ROOT / path) != sha:
                raise ValueError(f"Accepted artifact changed: {path}")
        receipts[name] = digest(ROOT / name)
    return receipts


def study_sources():
    return {**source_hashes(), **{str(p.relative_to(ROOT)): digest(p) for p in sorted((TASK / "tests").glob("*.py"))}}


def conditions(config, selection=None):
    result = []
    for w in config["windows"]:
        powers = config["weighted_powers"] if selection is None else [selection["selected"][w["id"]]["power"]]
        modes = [("core",0),("prob-core",0),("uniform",0), *[("weighted",p) for p in powers]]
        for mode,power in modes:
            suffix = f"weighted{power}" if mode == "weighted" else mode
            result.append({**w,"window_id":w["id"],"id":f"{w['id']}-{suffix}","mode":mode,"power":power,
                           "padded_limit":(w["max_length"]+7)//8*8})
    result.append({"id":"serial256","window_id":"serial256","mode":"serial","power":0,
                   "max_length":256,"core_tokens":192,"batch_size":1,"padded_limit":256,"anchored":False})
    return result


def select_weighting(records,config):
    choices = {}
    for w in config["windows"]:
        rs = [r for r in records if r["condition"]["window_id"] == w["id"] and r["condition"]["mode"] == "weighted"]
        rs = [r for r in rs if r["repeat_predictions_identical"]]
        if not rs:
            raise ValueError("No stable weighted candidate")
        best = max(r["quality"]["overall"]["punctuation_micro"]["f1"] for r in rs)
        eligible = [r for r in rs if 100*(best-r["quality"]["overall"]["punctuation_micro"]["f1"]) <= config["selection_quality_tolerance_pp"]]
        choice = min(eligible,key=lambda r:(r["median_seconds"],r["condition"]["power"]))
        choices[w["id"]] = {"condition_id":choice["condition"]["id"],"power":choice["condition"]["power"],
                             "development_micro_f1":choice["quality"]["overall"]["punctuation_micro"]["f1"],
                             "eligible_powers":[r["condition"]["power"] for r in eligible]}
    return choices


def probability_digest(output):
    h=hashlib.sha256()
    for w,p in zip(output["probability_windows"],output["probabilities"],strict=True):
        h.update(np.asarray([w.document,w.start,w.end,w.core_start,w.core_end],dtype="<i8").tobytes())
        h.update(np.asarray(p,dtype="<f4").tobytes())
    return h.hexdigest()


def save_cache(path,output):
    windows=output["probability_windows"]
    np.savez_compressed(path, probabilities=np.concatenate(output["probabilities"]),
                        offsets=np.cumsum([0,*[w.end-w.start for w in windows]]),
                        windows=np.asarray([[w.document,w.start,w.end,w.core_start,w.core_end] for w in windows],dtype=np.int64),
                        character_token_lengths=np.concatenate([np.diff(e.prefix) for e in output["encoded"]]))


def run_phase(config,receipt,phase):
    if phase not in ("development","test"):
        raise ValueError("Unknown phase")
    output_dir=ROOT/"outputs/task2"/config["experiment"]/phase
    report=TASK/"reports"/config["experiment"]
    if output_dir.exists():
        raise FileExistsError("Phase exists; use a new experiment ID")
    protected=history_checks(config)
    sources=study_sources()
    selection=None
    if phase=="test":
        selection=read_json(report/"selection.json")
        if selection["source_files_sha256"]!=sources or selection["config"]!=config:
            raise ValueError("Source/config differs from locked development selection")
        if digest(report/"development.json")!=selection["development_results_sha256"]:
            raise ValueError("Development results changed")
        selection_sha=digest(report/"selection.json")
    variants=conditions(config,selection)
    rows,dataset=checked_dataset(config,phase)
    texts=[r["text"] for r in rows]
    backend=ProbabilityBackend(config,receipt)
    output_dir.mkdir(parents=True)
    write_json(output_dir/"plan.json",{"config":config,"phase":phase,"conditions":variants,"dataset":dataset,
                                       "source_files_sha256":sources,"history_checkpoints":protected,
                                       "selection_sha256":None if selection is None else selection_sha})
    times,records,fingerprints,prob_fingerprints={},{},{},{}
    rng=np.random.default_rng(config["seed"]+(phase=="test"))
    warm_order=rng.permutation(len(variants)).tolist()
    orders=[]

    def once(c):
        backend.max_length=c["padded_limit"]
        if c["mode"] in ("core","serial"):
            from .context_study import anchored_windows
            return infer(texts,backend,c["max_length"],c["core_tokens"],
                         "adaptive" if c["mode"]=="serial" else "fixed",c["batch_size"],
                         window_builder=anchored_windows if c["anchored"] else None)
        return infer_probabilities(texts,backend,c,c["mode"],c["power"])

    with (output_dir/"progress.jsonl").open("x",encoding="utf-8",buffering=1) as log_file:
        def log(value):
            line=json.dumps(value,ensure_ascii=False);log_file.write(line+"\n");print(line,flush=True)
        log({"event":"plan","phase":phase,"conditions":len(variants),"documents":len(rows)})
        for i in warm_order:
            for _ in range(config["warmup_runs"]):once(variants[i])
            log({"event":"warmup","condition":variants[i]["id"]})
        for repetition in range(config["repetitions"]):
            order=rng.permutation(len(variants)).tolist();orders.append([variants[i]["id"] for i in order])
            for i in order:
                c=variants[i];name=c["id"]
                out=once(c)
                fingerprint=prediction_digest(out["labels"])
                prob_sha=probability_digest(out) if "probabilities" in out else None
                fingerprints.setdefault(name,[]).append(fingerprint)
                if prob_sha is not None:prob_fingerprints.setdefault(name,[]).append(prob_sha)
                times.setdefault(name,[]).append({"seconds":out["seconds"],**out["profile"],
                                                "prediction_sha256":fingerprint,"probability_sha256":prob_sha})
                log({"event":"measurement","round":repetition+1,"condition":name,"seconds":round(out["seconds"],5),
                     "prediction_sha256":fingerprint})
                if repetition != config["repetitions"]-1:continue
                path=output_dir/name
                write_rows(path/"predictions.jsonl",[{"document_id":r["document_id"],"gold":r["labels"],"predicted":p}
                                                     for r,p in zip(rows,out["labels"],strict=True)])
                write_rows(path/"windows.jsonl",out["windows"])
                stats=None
                if "probabilities" in out:
                    coverage=np.concatenate(out["coverage_counts"]);sums=np.concatenate(out["weight_sums"])
                    np.savez_compressed(path/"aggregation.npz",coverage=coverage,weight_sums=sums)
                    stats={"minimum_weight_sum":float(sums.min()),"maximum_weight_sum":float(sums.max()),
                           "coverage_histogram":{str(k):int(v) for k,v in zip(*np.unique(coverage,return_counts=True))},
                           "multiply_covered_characters":int(np.sum(coverage>1))}
                    if c["mode"]=="prob-core":save_cache(path/"probabilities.npz",out)
                records[name]={"condition":c,"quality":quality(rows,out,config["boundary_margin"]),
                               "document_counts":document_counts([r["labels"] for r in rows],out["labels"]).tolist(),
                               "prediction_sha256":fingerprint,"probability_sha256":prob_sha,"aggregation":stats,
                               "windows":len(out["windows"]),"forward_calls":out["forward_calls"],
                               "processed_content_tokens":out["processed_content_tokens"],"content_tokens":out["content_tokens"],
                               "overlap_work_factor":out["processed_content_tokens"]/out["content_tokens"]}
    draws=np.random.default_rng(config["seed"]+2).integers(0,len(rows),size=(config["bootstrap_samples"],len(rows)))
    boot={name:bootstrap_scores(np.asarray(r["document_counts"]),draws) for name,r in records.items()}
    for name,r in records.items():
        samples=times[name];seconds=[s["seconds"] for s in samples]
        r.update({"measurements":samples,"median_seconds":statistics.median(seconds),"stdev_seconds":statistics.stdev(seconds),
                  "characters_per_second":sum(map(len,texts))/statistics.median(seconds),
                  "peak_allocated_bytes":max(s["peak_allocated_bytes"] for s in samples),
                  "repeat_predictions_identical":len(set(fingerprints[name]))==1,
                  "repeat_probabilities_identical":len(set(prob_fingerprints.get(name,[])))<=1,
                  "micro_f1_bootstrap_95ci":np.quantile(boot[name],[.025,.975]).tolist()})
    for name,r in records.items():
        c=r["condition"]
        if c["mode"]=="serial":continue
        core=records[f"{c['window_id']}-core"]
        if (output_dir/name/"windows.jsonl").read_bytes()!=(output_dir/core["condition"]["id"]/"windows.jsonl").read_bytes():
            raise AssertionError("Fusion changed fixed window positions")
        r["comparisons"]={}
        for key,ref_name in (("core",core["condition"]["id"]),("prob-core",f"{c['window_id']}-prob-core"),("serial","serial256")):
            ref=records[ref_name]
            r["comparisons"][key]={"reference":ref_name,"latency_ratio":r["median_seconds"]/ref["median_seconds"],
                                   "speedup":ref["median_seconds"]/r["median_seconds"],
                                   "delta_micro_f1_pp":100*(r["quality"]["overall"]["punctuation_micro"]["f1"]-ref["quality"]["overall"]["punctuation_micro"]["f1"]),
                                   "paired_95ci_pp":(100*np.quantile(boot[name]-boot[ref_name],[.025,.975])).tolist()}
        if c["mode"]=="prob-core" and r["prediction_sha256"]!=core["prediction_sha256"]:
            raise AssertionError("Probability retention changed core argmax predictions")
        if c["mode"] not in ("core","prob-core") and r["probability_sha256"]!=records[f"{c['window_id']}-prob-core"]["probability_sha256"]:
            raise AssertionError("Aggregation modes used different model probabilities")
    assert_protected(receipt)
    if history_checks(config)!=protected or study_sources()!=sources:raise RuntimeError("Protected source/artifacts changed")
    if selection is not None and digest(report/"selection.json")!=selection_sha:raise RuntimeError("Selection changed")
    result={"schema_version":1,"date":"2026-10-17","phase":phase,"config":config,"dataset":dataset,
            "documents":len(rows),"characters":sum(map(len,texts)),"environment":backend.environment,
            "conditions":[records[c["id"]] for c in variants],"warmup_order":[variants[i]["id"] for i in warm_order],
            "measurement_orders":orders,"source_files_sha256":sources,"history_checkpoints":protected,
            "task1_unchanged":True,"previous_experiments_unchanged":True,"task1_commit":receipt["task1_commit"],
            "model_files_sha256":receipt["model_files_sha256"],"test_labels_used_for_selection":False,
            "selection_sha256":None if selection is None else selection_sha,
            "bootstrap":{"samples":len(draws),"seed":config["seed"]+2,"paired":True,"unit":"anonymous document group","multiple_comparison_adjustment":False},
            "timing":"warm model/tokenizer; encode/window/sort/pad/transfer/model/probability/aggregate/argmax/render; exclude model/data load, provenance/scoring/hash/cache serialization and file writes"}
    write_json(output_dir/"results.json",result);write_json(report/f"{phase}.json",result)
    export_table(result,report/f"{phase}.csv")
    snapshot=ROOT/"outputs/task2/source-snapshots"/config["experiment"]
    for name,sha in sources.items():
        target=snapshot/name
        if target.exists():
            if digest(target)!=sha:raise ValueError("Source snapshot differs")
        else:
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    if phase=="development":
        write_json(report/"selection.json",{"config":config,"selected":select_weighting(result["conditions"],config),
                                           "source_files_sha256":sources,"development_results_sha256":digest(report/"development.json"),
                                           "rule":"per window: stable weighted candidate; within 0.02 pp of highest development micro-F1 prefer lowest median latency, then smaller power",
                                           "test_labels_used":False})
    return {"experiment":config["experiment"],"phase":phase,"conditions":len(variants)}


def export_table(result,path):
    fields=["condition","micro_f1_percent","macro_f1_percent","segmentation_f1_percent","sentence_f1_percent",
            "edge_micro_f1_percent","median_seconds","stdev_seconds","characters_per_second","windows","forward_calls",
            "aggregation_median_seconds","d2h_bytes","peak_allocated_gib","delta_vs_core_pp","slowdown_vs_core","speedup_vs_serial"]
    with path.open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in result["conditions"]:
            q=r["quality"]["overall"];core=r.get("comparisons",{}).get("core",{});serial=r.get("comparisons",{}).get("serial",{})
            writer.writerow({"condition":r["condition"]["id"],"micro_f1_percent":q["punctuation_micro"]["f1"]*100,
                             "macro_f1_percent":q["punctuation_macro_f1"]*100,"segmentation_f1_percent":q["segmentation_any_punctuation"]["f1"]*100,
                             "sentence_f1_percent":q["sentence_boundary"]["f1"]*100,"edge_micro_f1_percent":r["quality"]["window_edges"]["punctuation_micro"]["f1"]*100,
                             **{k:r[k] for k in ("median_seconds","stdev_seconds","characters_per_second","windows","forward_calls")},
                             "aggregation_median_seconds":statistics.median(s.get("aggregation_seconds",0) for s in r["measurements"]),
                             "d2h_bytes":r["measurements"][0].get("d2h_probability_bytes",r["measurements"][0].get("d2h_integer_bytes",0)),
                             "peak_allocated_gib":r["peak_allocated_bytes"]/2**30,"delta_vs_core_pp":core.get("delta_micro_f1_pp"),
                             "slowdown_vs_core":core.get("latency_ratio"),"speedup_vs_serial":serial.get("speedup")})
