"""Bounded teacher -> filtered soft labels -> student -> validation -> next teacher."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from model import Model, adapter_dir
from adaptive import fit_policy, inherited_groups, select
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"round3"))
from text_index import TextIndex

SEED=314159
MIN_PROB=.90
PER_CLASS=32
LR=1e-5
PSEUDO_WEIGHT=.35

def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def write(path,obj):
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8")
    tmp.replace(path)

def targets(rows):
    return torch.tensor([r["label"] for r in rows])

def fit_temperature(logits,labels):
    # Fit a single scalar on labelled calibration rows, never on the promotion set.
    values=torch.logspace(math.log10(.5),math.log10(8),33)
    losses=[F.cross_entropy(logits/t,labels).item() for t in values]
    return float(values[losses.index(min(losses))])

def metrics(logits,rows,temperature=1.):
    y=targets(rows)
    probs=(logits/temperature).softmax(-1)
    confidence,pred=probs.max(-1)
    ok=pred==y
    cm=torch.bincount(3*y+pred,minlength=9).reshape(3,3)
    supports=cm.sum(1)
    recalls=cm.diag()[supports>0]/supports[supports>0]
    ece=0.
    for i in range(10):
        inside=(confidence>i/10)&(confidence<=(i+1)/10)
        if inside.any():
            ece+=float(inside.float().mean()*abs(ok[inside].float().mean()-confidence[inside].mean()))
    return {"correct":int(ok.sum()),"n":len(rows),"confusion":cm.tolist(),
            "balanced_recall":float(recalls.mean()),"malicious_misses":int(((y==1)&~ok).sum()),
            "risky_as_normal":int(((y!=0)&(pred==0)).sum()),"high_conf_wrong":int((~ok&(confidence>=.9)).sum()),
            "brier":float(((probs-F.one_hot(y,3))**2).sum(1).mean()),"ece10":ece,
            "mean_confidence":float(confidence.mean())}

def audit(model,calibration,gate,adaptive=False):
    views=[model.scores(calibration,view) for view in (0,1)]
    temps=[fit_temperature(z,targets(calibration)) for z in views]
    logits=model.scores(gate)
    by_set={}
    for name in sorted({r["set"] for r in gate}):
        ids=[i for i,r in enumerate(gate) if r["set"]==name]
        by_set[name]=metrics(logits[ids],[gate[i] for i in ids])
    policy = {}
    if adaptive:
        probabilities = [(z/t).softmax(-1) for z,t in zip(views,temps)]
        policy["class_policy"] = fit_policy(*probabilities, targets(calibration))
    return {**policy,"temperatures":temps,"calibration_views":[metrics(z,calibration,t) for z,t in zip(views,temps)],
            "raw":metrics(logits,gate),"scaled":metrics(logits,gate,temps[0]),
            "by_set":by_set,"score":sum(v["balanced_recall"] for v in by_set.values())/len(by_set)}

def admit(model,pool,temps,rng):
    p,q=[(model.scores(pool,i)/temps[i]).softmax(-1) for i in (0,1)]
    choices=[]
    for row,a,b in zip(pool,p,q):
        if not (a.isfinite().all() and b.isfinite().all()):continue
        j=int(a.argmax())
        if j!=int(b.argmax()) or min(float(a[j]),float(b[j]))<MIN_PROB:
            continue
        choices.append({**row,"class":j,"soft":((a+b)/2).tolist(),"view_scores":[float(a[j]),float(b[j])]})
    rng.shuffle(choices)
    counts=Counter();groups=set();selected=[]
    for row in choices:
        label=row["class"]
        if counts[label]>=PER_CLASS or row["group"] in groups:continue
        selected.append(row);groups.add(row["group"]);counts[label]+=1
    return selected,len(choices)

def anchor_schedule(model,anchors,steps,rng,temperature):
    # Each step has six labelled examples. Half are difficult for the current teacher.
    by_class=[]
    for c in range(3):
        rows=[r for r in anchors if r["label"]==c]
        rows=rng.sample(rows,min(160,len(rows)))
        reference=(model.scores(rows)/temperature).softmax(-1)
        rows=[{**r,"reference":soft.tolist()} for r,soft in zip(rows,reference)]
        order=reference[:,c].argsort().tolist()
        by_class.append((rows,[rows[i] for i in order[:max(4,len(rows)//4)]]))
    return [[rng.choice(group) for both in by_class for group in both] for _ in range(steps)]

def contrast_schedule(model,batches,anchors,cases,rng,temperature):
    # Each case must consist entirely of existing labelled training examples.
    labels={r["text"]:r["label"] for r in anchors}
    rows=[]
    groups=set()
    seen=set()
    for case in cases:
        texts=case["texts"]
        if (case["group"] in groups or len(texts)!=3 or
                any(labels.get(t)!=i or t in seen for i,t in enumerate(texts))):
            raise ValueError("Contrast cases must be distinct labelled anchor triplets")
        groups.add(case["group"])
        seen.update(texts)
        rows.extend({"text":t,"label":i} for i,t in enumerate(texts))
    if not rows:
        raise ValueError("No contrast groups")
    scores=(model.scores(rows)/temperature).softmax(-1).tolist()
    rows=[{**r,"reference":s} for r,s in zip(rows,scores)]
    triplets=[rows[i:i+3] for i in range(0,len(rows),3)]
    index=0
    for step in range(0,len(batches),2):
        if index % len(triplets)==0:
            rng.shuffle(triplets)
        # Keep one hard example per class alongside the three examples from one case.
        batches[step]=batches[step][1::2]+triplets[index%len(triplets)]
        index+=1

def update(model,batches,lr,anchor_mix,selected=None,adaptive=False):
    opt=torch.optim.AdamW([p for p in model.lm.parameters() if p.requires_grad],lr=lr)
    model.lm.train()
    for i,batch in enumerate(batches):
        opt.zero_grad(set_to_none=True)
        for start in (0,3):
            micro=batch[start:start+3]
            logits=model.logits([r["text"] for r in micro])
            if anchor_mix:
                truth=F.one_hot(targets(micro),3).float()
                reference=torch.tensor([r["reference"] for r in micro])
                target=((1-anchor_mix)*truth+anchor_mix*reference).to("cuda")
                loss=-(target*logits.log_softmax(-1)).sum(-1).mean()
            else:
                loss=F.cross_entropy(logits,targets(micro).to("cuda"))
            (loss/2).backward()
        if selected:
            micro=selected[i*2:i*2+2]
            soft=torch.tensor([r["soft"] for r in micro],device="cuda")
            logp=model.logits([r["text"] for r in micro],view=int(adaptive)).log_softmax(-1)
            losses=-(soft*logp).sum(-1)
            if adaptive:
                losses=losses*torch.tensor([r["weight"] for r in micro],device="cuda")
            (PSEUDO_WEIGHT*losses.mean()).backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.lm.parameters() if p.requires_grad],1,error_if_nonfinite=True)
        opt.step()
    model.lm.eval()

def decision(parent,student,control):
    old,new=parent["raw"],student["raw"]
    checks={"balanced_recall_preserved":student["score"]>=parent["score"]-.001,
            "malicious_misses_preserved":new["malicious_misses"]<=old["malicious_misses"],
            "risky_as_normal_preserved":new["risky_as_normal"]<=old["risky_as_normal"],
            "high_confidence_errors_guard":new["high_conf_wrong"]<=old["high_conf_wrong"]+1,
            "brier_guard":student["scaled"]["brier"]<=parent["scaled"]["brier"]+.005,
            "vs_gold_only":student["score"]>=control["score"]-.005 and student["scaled"]["brier"]<=control["scaled"]["brier"]+.005,
            "measured_benefit":student["score"]>=parent["score"]+.002 or student["scaled"]["brier"]<=parent["scaled"]["brier"]-.003}
    return checks

def inputs(folder):
    manifest=json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    names=("anchors","pool","calibration","gate","exclusion-texts")
    if set(manifest["file_sha256"])!={name+".jsonl" for name in names}:
        raise ValueError("Incomplete split fingerprint manifest")
    for name in names:
        filename=name+".jsonl"
        if hashlib.sha256((folder/filename).read_bytes()).hexdigest()!=manifest["file_sha256"][filename]:
            raise ValueError("Changed data file: "+filename)
    data={name:read_rows(folder/(name+".jsonl")) for name in names}
    if any(not data[name] for name in names):
        raise ValueError("Empty training or validation partition")
    if any(not isinstance(r.get("text"),str) or not 0<len(r["text"].strip())<=20000 for rows in data.values() for r in rows):
        raise ValueError("Each row must contain 1 to 20000 characters")
    # Refuse labels in the pseudo-labelling path, even if a custom input file supplies one.
    if any(set(row)!={"text","group"} for row in data["pool"]):
        raise ValueError("Pool rows must contain only text and group")
    if any(not isinstance(row["group"],str) or not row["group"].strip() for row in data["pool"]):
        raise ValueError("Pool rows require a nonempty source group")
    blocked=TextIndex()
    for name in ("exclusion-texts","gate","calibration","anchors"):
        rows=data[name]
        if any(blocked.contains(r["text"]) for r in rows):
            raise ValueError(f"{name} overlaps a held-out or calibration split")
        if name!="exclusion-texts" and any(r["label"] not in (0,1,2) for r in rows):
            raise ValueError("Unexpected class label")
        for row in rows:blocked.add(row["text"])
    if any(blocked.contains(row["text"]) for row in data["pool"]):
        raise ValueError("Pool overlaps labelled or held-out material")
    return data,manifest

def main():
    parser=argparse.ArgumentParser()
    for name in ("base","teacher","data","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--rounds",type=int,choices=(1,2,3),default=2)
    parser.add_argument("--stabilized",action="store_true",help="Retain part of the teacher distribution on labelled examples; require three pseudo-label classes")
    parser.add_argument("--consistency-view",choices=("features","reword"),default="features")
    parser.add_argument("--adaptive",action="store_true",help="Class-specific admission, confidence weighting and source lineage")
    parser.add_argument("--prior-trial",type=Path,help="Import a legacy recursive teacher's accepted source groups")
    parser.add_argument("--seed",type=int,default=SEED)
    parser.add_argument("--contrast-anchors",type=Path,help="Grouped normal/execute/review triplets already present in anchors")
    args=parser.parse_args()
    if args.adaptive:
        args.stabilized=True
        if args.consistency_view!="features":
            parser.error("--adaptive requires the feature consistency view")
    elif args.prior_trial:
        parser.error("--prior-trial requires --adaptive")
    args.out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    data,manifest=inputs(args.data)
    contrast_data=args.contrast_anchors.read_bytes() if args.contrast_anchors else None
    model=Model(args.base,args.teacher)
    model.consistency_view=args.consistency_view
    config=adapter_dir(args.teacher)/"adapter_config.json"
    active_path=args.teacher.resolve()
    active_sha=model.info["sha256"]
    lr,anchor_mix,examined=(5e-6,.40,1024) if args.stabilized else (LR,0.,512)
    used=inherited_groups(args.teacher,model.info,args.prior_trial) if args.adaptive else set()
    protocol={"seed":args.seed,"round_limit":args.rounds,"soft_ce_weight":PSEUDO_WEIGHT,"adaptive":args.adaptive,
              "inherited_source_groups":len(used),"per_view_threshold":"per-class calibration" if args.adaptive else MIN_PROB,"class_cap":PER_CLASS,"lr":lr,"gold_per_step":6,
              "anchor_reference_mix":anchor_mix,"require_three_classes":args.stabilized,"consistency_view":args.consistency_view,
              "pseudo_per_step":2,"max_examined_per_round":examined,"input_manifest":manifest}
    report={"start":str(active_path),"start_sha256":active_sha,"active":str(active_path),
            "active_sha256":active_sha,"protocol":protocol,"rounds":[],"complete":False}
    if contrast_data is not None:
        protocol["contrast_anchor_sha256"]=hashlib.sha256(contrast_data).hexdigest()
    write(args.out/"trial.json",report)
    for ordinal in range(args.rounds):
        folder=args.out/f"round-{ordinal+1}"
        folder.mkdir()
        rng=random.Random(args.seed+ordinal)
        parent=audit(model,data["calibration"],data["gate"],args.adaptive)
        for view in parent["calibration_views"]:
            matrix=view["confusion"]
            if any(sum(c)<4 or c[j]/sum(c)<.5 for j,c in enumerate(matrix)):
                report["stop"]="unreliable_teacher_view"
                report["rounds"].append({"round":ordinal+1,"teacher_sha256":active_sha,"parent":parent,"promoted":False})
                break
        if "stop" in report:break
        pool=[r for r in data["pool"] if r["group"] not in used and int(hashlib.sha256(r["group"].encode()).hexdigest(),16)%args.rounds==ordinal]
        rng.shuffle(pool)
        pool=pool[:examined]
        if args.adaptive:
            if not pool:
                report["stop"]="source_pool_exhausted"
                break
            probs=[(model.scores(pool,v)/temp).softmax(-1) for v,temp in enumerate(parent["temperatures"])]
            selected,by_class=select(pool,*probs,parent["class_policy"],rng,PER_CLASS)
            qualified=sum(by_class.values())
        else:
            selected,qualified=admit(model,pool,parent["temperatures"],rng)
        record={"round":ordinal+1,"teacher_sha256":active_sha,"examined":len(pool),
                "qualified_views":qualified,"selected":len(selected),"class_counts":dict(Counter(r["class"] for r in selected)),
                "parent":parent}
        report["rounds"].append(record)
        if args.adaptive:
            record.update(qualified_by_class=by_class,prior_source_groups=len(used))
        if len(selected)<24 or len(record["class_counts"])<2 or (args.stabilized and min(record["class_counts"].get(c,0) for c in range(3))<8):
            report["stop"]="inadequate_pseudo_label_support"
            record["promoted"]=False
            break
        write(folder/"pseudo-labels.json",[{"text_sha256":hashlib.sha256(r["text"].encode()).hexdigest(),
                                            "source_group":r["group"],"soft":r["soft"],"view_scores":r["view_scores"],
                                            **{k:r[k] for k in ("weight","view_tv") if k in r}}
                                           for r in selected])
        initial=model.weights()
        batches=anchor_schedule(model,data["anchors"],math.ceil(len(selected)/2),rng,parent["temperatures"][0])
        if contrast_data is not None:
            contrast_schedule(model,batches,data["anchors"],json.loads(contrast_data),rng,parent["temperatures"][0])
        record["steps"]=len(batches)
        # Both runs start at the same parameters and receive the exact same labelled batches.
        update(model,batches,lr,anchor_mix)
        record["control"]=audit(model,data["calibration"],data["gate"])
        record["control_sha256"]=model.save(folder/"gold-only",config,active_sha,"matched gold-only control")
        model.restore(initial)
        update(model,batches,lr,anchor_mix,selected,adaptive=args.adaptive)
        candidate_path=folder/"student"
        student_sha=model.save(candidate_path,config,active_sha,"experimental student")
        model.restore(load_file(str(candidate_path/"adapter/adapter_model.safetensors")))
        record["student_sha256"]=student_sha
        record["student"]=audit(model,data["calibration"],data["gate"])
        record["checks"]=decision(parent,record["student"],record["control"])
        record["promoted"]=all(record["checks"].values())
        print("round",ordinal+1,"selected",len(selected),"promoted",record["promoted"],"checks",record["checks"],flush=True)
        if not record["promoted"]:
            report["stop"]="validation_guard"
            model.restore(initial)
            break
        active_path=candidate_path.resolve()
        active_sha=student_sha
        if args.adaptive:
            used.update(r["group"] for r in selected)
            ledger=active_path/"source-groups.json"
            write(ledger,{"adapter_sha256":active_sha,"groups":sorted(used)})
            info=json.loads((active_path/"report.json").read_bytes())
            info["source_groups_sha256"]=hashlib.sha256(ledger.read_bytes()).hexdigest()
            write(active_path/"report.json",info)
        report.update(active=str(active_path),active_sha256=active_sha)
        write(args.out/"trial.json",report)
    report.setdefault("stop","round_limit")
    report["complete"]=True
    write(args.out/"trial.json",report)
    print("Final accepted teacher",report["active_sha256"],flush=True)

if __name__=="__main__":
    main()
