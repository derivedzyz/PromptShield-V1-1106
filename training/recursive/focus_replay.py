"""Repair a rejected focus student before opening the standing regression sets."""
import argparse
import hashlib
from pathlib import Path
import random

from safetensors.torch import load_file
from focus_round import (HERE, SEED, Model, adapter_dir, audit, batches, decision, inputs,
                         metrics, prepare, read, reference, select, torch, update, write)


def mix(parent,current,keep):
    if keep==1:
        return current
    return {key:value.lerp(current[key],keep) for key,value in parent.items()}


def main():
    parser=argparse.ArgumentParser()
    for name in ("base","teacher","data","deepset-train","extra","pilot","out"):
        parser.add_argument("--"+name,required=True,type=Path)
    args=parser.parse_args()
    pilot=read(args.pilot/"audit.json")
    if not pilot["complete"] or pilot["passes"]:
        raise ValueError("Replay requires a completed failed pilot")
    args.out.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    rng=random.Random(SEED)
    data,manifest=inputs(args.data)
    if manifest!=pilot["original_manifest"]:
        raise ValueError("Data split changed")
    for f in (args.deepset_train,args.extra,HERE/"focus-train.json",HERE/"focus-dev.json"):
        if hashlib.sha256(f.read_bytes()).hexdigest()!=pilot["files_sha256"][f.name]:
            raise ValueError("Training input changed: "+f.name)
    domain,probe,focus,dev,anchors,pool=prepare(args,data,rng)
    model=Model(args.base,args.teacher)
    model.consistency_view="features"
    original=model.weights()
    parent=pilot["parent"]
    start=read(args.pilot/pilot["chosen"]/"report.json")
    if start["parent_sha256"]!=model.info["sha256"]:
        raise ValueError("Incorrect initial teacher")
    src=args.pilot/pilot["chosen"]/"adapter/adapter_model.safetensors"
    if hashlib.sha256(src.read_bytes()).hexdigest()!=pilot["chosen_sha256"]:
        raise ValueError("Pilot weight mismatch")
    ledger=read(args.teacher/"source-groups.json")
    used=set(ledger["groups"])
    pool=[r for r in pool if r["group"] not in used]
    rng.shuffle(pool)
    pool=pool[:2048]
    probs=[(model.scores(pool,v)/t).softmax(-1) for v,t in enumerate(parent["temperatures"])]
    selected,_=select(pool,*probs,parent["class_policy"],rng,64)
    if [r["group"] for r in selected]!=read(args.pilot/"pseudo-sources.json"):
        raise ValueError("The pilot selection could not be reproduced")
    old_batches=batches(model,domain,focus,anchors,parent["temperatures"][0],rng)
    precursor=next(r for r in pilot["checkpoints"] if r["name"]==pilot["chosen"])
    # Keep a matched pair from one process; past scores are never a stand-in for its weights.
    student0=control0=original
    config=adapter_dir(args.teacher)/"adapter_config.json"
    for stop in range(24,precursor["steps"]+1,24):
        begin=stop-24
        model.restore(control0)
        update(model,old_batches[begin:stop],precursor["lr"],.1)
        control0=model.weights()
        model.restore(student0)
        update(model,old_batches[begin:stop],precursor["lr"],.1,selected[2*begin:2*stop],True)
        student0=model.weights()
    pair={}
    for role,weights in (("control",control0),("student",student0)):
        model.restore(weights)
        path=args.out/("paired-"+role)
        sha=model.save(path,config,model.info["sha256"],"paired "+role)
        before=audit(model,data["calibration"],data["gate"])
        model.restore(load_file(str(path/"adapter/adapter_model.safetensors")))
        after=audit(model,data["calibration"],data["gate"])
        if before["raw"]["confusion"]!=after["raw"]["confusion"] or abs(before["scaled"]["brier"]-after["scaled"]["brier"])>1e-6:
            raise ValueError("Saved matched pair failed its own readback")
        pair[role]={"sha256":sha,"gate":after}
    print("Paired readback",pair["control"]["gate"]["raw"]["correct"],pair["student"]["gate"]["raw"]["correct"],flush=True)
    model.restore(original)
    base_rows=reference(model,anchors,parent["temperatures"][0])
    public=reference(model,domain,parent["temperatures"][0])
    intents=reference(model,focus,parent["temperatures"][0])
    model.restore(student0)
    current=(model.scores(anchors)/parent["temperatures"][0]).softmax(-1).tolist()
    buckets={}
    for c in (0,1):
        drift=[(r["reference"][c]-p[c],r) for r,p in zip(base_rows,current)
               if r["label"]==c and r["reference"][c]>=.70]
        drift.sort(key=lambda pair:pair[0],reverse=True)
        buckets[c]=[r for _,r in drift[:64]]
    intent={c:[r for r in intents if r["label"]==c] for c in range(3)}
    domain_by_class={c:[r for r in public if r["label"]==c] for c in (0,1)}
    rng=random.Random(SEED+1)
    schedule=[]
    for i in range(64):
        schedule.append(rng.sample(buckets[1][:24],1)+rng.sample(buckets[1][24:],1)+
                        [rng.choice(buckets[0]),rng.choice(domain_by_class[i%2]),
                         rng.choice(intent[2]),rng.choice(intent[(2,0,2,1)[i%4]])])
    report={k:pilot[k] for k in ("parent_sha256","parent","parent_dev","parent_deepset_probe",
                                 "original_manifest","files_sha256","n_domain_train","n_focus","n_focus_dev")}
    report.update(pilot_sha256=start["sha256"],matched_replication=pair,checkpoints=[],complete=False)
    for lr in (.000007,.000018):
        control,student=control0,student0
        previous=0
        for stop in (16,32,64):
            subset=schedule[previous:stop]
            previous=stop
            model.restore(control)
            update(model,subset,lr,.10)
            control=model.weights()
            model.restore(student)
            update(model,subset,lr,.10)
            student=model.weights()
            for keep in (.65,1.):
                model.restore(mix(original,control,keep))
                ctrl=audit(model,data["calibration"],data["gate"])
                model.restore(mix(original,student,keep))
                result=audit(model,data["calibration"],data["gate"])
                guard=decision(parent,result,ctrl)
                d=metrics(model.scores(dev),dev)
                safe=all(d[k]<=pilot["parent_dev"][k] for k in ("malicious_misses","risky_as_normal"))
                domain_dev=metrics(model.scores(probe),probe)
                name=f"lr-{round(lr*1e6):02d}-step-{stop:03d}-mix-{round(keep*100)}"
                sha=model.save(args.out/name,config,report["parent_sha256"],"targeted recursive student")
                row={"name":name,"sha256":sha,"steps":stop,"lr":lr,"keep":keep,"dev_guard":safe,
                     "passes":safe and all(guard.values()),"original_checks":guard,"gate":result,
                     "control":ctrl,"dev":d,"deepset_probe":domain_dev}
                report["checkpoints"].append(row)
                write(args.out/"audit.json",report)
                print(name,row["passes"],result["raw"]["correct"],d["confusion"][2],domain_dev["correct"],flush=True)
    def rank(r):
        review=r["dev"]["confusion"][2]
        return (r["passes"],review[2],-review[1],r["deepset_probe"]["correct"],r["gate"]["score"],
                -r["gate"]["scaled"]["brier"],-r["steps"])
    chosen=max(report["checkpoints"],key=rank)
    path=args.out/chosen["name"]
    source_ledger=args.pilot/pilot["chosen"]/"source-groups.json"
    if hashlib.sha256(source_ledger.read_bytes()).hexdigest()!=start["source_groups_sha256"]:
        raise ValueError("Pilot ledger mismatch")
    write(path/"source-groups.json",{"adapter_sha256":chosen["sha256"],"groups":read(source_ledger)["groups"]})
    info=read(path/"report.json")
    info["source_groups_sha256"]=hashlib.sha256((path/"source-groups.json").read_bytes()).hexdigest()
    write(path/"report.json",info)
    report.update(chosen=chosen["name"],chosen_sha256=chosen["sha256"],complete=True,passes=chosen["passes"])
    write(args.out/"audit.json",report)
    print("SEALED",chosen["name"],chosen["sha256"],chosen["passes"],flush=True)

if __name__=="__main__":
    main()
