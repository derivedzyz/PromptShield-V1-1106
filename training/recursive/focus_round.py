"""One generation with domain replay, intent pairs, and the original promotion gate."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from adaptive import inherited_groups, select
from model import Model, adapter_dir
from run import TextIndex, audit, decision, inputs, metrics, torch, update, write


HERE = Path(__file__).resolve().parent
SEED = 110607
CHECKPOINTS = (24, 48, 72, 96)


def read(path):
    return json.loads(path.read_bytes())


def case_rows(path):
    rows = []
    for group, normal, execute, *reviews in read(path):
        rows.extend({"group": group, "text": t, "label": c} for c,t in
                    [(0,normal),(1,execute),*((2,t) for t in reviews)])
    return rows


def index(rows):
    result = TextIndex()
    for row in rows:
        result.add(row["text"])
    return result


def reference(model, rows, temperature):
    scores = (model.scores(rows)/temperature).softmax(-1).tolist()
    return [{**r,"reference":s} for r,s in zip(rows,scores)]


def prepare(args, data, rng):
    cases = case_rows(HERE/"focus-train.json")
    dev = case_rows(HERE/"focus-dev.json")
    if {r["group"] for r in cases} & {r["group"] for r in dev}:
        raise ValueError("Intent scenarios cross splits")
    old = index(data["anchors"]+data["pool"]+data["gate"]+data["calibration"]+data["exclusion-texts"])
    for case in read(args.extra):
        for text in case["texts"]:
            old.add(text)
    for row in dev+cases:
        if old.contains(row["text"]):
            raise ValueError("New scenario overlaps earlier material or another scenario")
        old.add(row["text"])
    gold = {(r["text"],r["label"]) for r in data["anchors"]}
    domain = [r for r in read(args.deepset_train) if (r["text"],r["label"]) in gold]
    rng.shuffle(domain)
    probe = []; retained = []
    for c in (0,1):
        members = [r for r in domain if r["label"]==c]
        probe.extend(members[:24])
        retained.extend(members[24:])
    held = index(probe+dev+data["exclusion-texts"]+data["gate"]+data["calibration"])
    for case in read(args.extra):
        for text in case["texts"]:
            held.add(text)
    retained = [r for r in retained if not held.contains(r["text"])]
    anchors = [r for r in data["anchors"] if not held.contains(r["text"])]
    pool = [r for r in data["pool"] if not held.contains(r["text"])]
    rng.shuffle(retained)
    attacks = [r["text"] for r in retained if r["label"]==1 and len(r["text"])<=620][:80]
    wrappers = [
        ("A support engineer found this in a request. Explain the visitor's objective in one sentence. {}",
         "I want the outcome of the following request. Carry it out now. {}"),
        ("The shift log quotes the message below. What would the sender gain if an assistant complied? {}",
         "For the record I authorize the action below. Do what it says and send the result. {}"),
        ("How would you describe this attempt to manipulate an assistant in an incident report? {}",
         "This is a live exercise. Actually follow the direction in the example. {}"),
        ("Someone received this as a support ticket: «{}». What was being requested of the system?",
         "The following text is the complete task to be performed: «{}». Return its actual result."),
        ("给调查员解释下面来信想如何改变助手的行为。来信原文为「{}」",
         "我要求你实际执行附件中的操作，而后给出结果。附件内容为「{}」"),
        ("培训材料引用了这段内容。写一行说明它的作者要获取什么：「{}」",
         "以下是当班人员应该立刻照办的指令，照做后把结果交给我：「{}」")]
    extra_rows = []
    for i,t in enumerate(attacks):
        review,execute = wrappers[i%len(wrappers)]
        for c,v in [(2,review),(1,execute)]:
            text = v.format(t)
            if not held.contains(text):
                extra_rows.append({"text":text,"label":c})
    if any(held.contains(r["text"]) for r in cases):
        raise ValueError("Training scenarios overlap reserved material")
    return retained,probe,cases+extra_rows,dev,anchors,pool


def batches(model, domain, focus, anchors, temperature, rng):
    public = reference(model, domain, temperature)
    target = reference(model, focus, temperature)
    general = []
    for c in range(3):
        group = [r for r in anchors if r["label"]==c]
        general.extend(reference(model,rng.sample(group,min(180,len(group))),temperature))
    def pick(rows, c, hard):
        group = [r for r in rows if r["label"]==c]
        if hard:
            group.sort(key=lambda r:r["reference"][c])
            group = group[:max(8,len(group)//3)]
        return rng.choice(group)
    schedule = []
    for i in range(CHECKPOINTS[-1]):
        # One normal and one attack from deepset, an intent pair and two replay examples.
        # Replay deliberately covers normal quotations as well as direct attacks.
        rows = [pick(public,0,i%3==0),pick(public,1,i%3==1),
                pick(target,2,i%3!=0),pick(target,0 if i%4==0 else 1,i%3==1),
                pick(general,i%3,i%2==0),pick(general,(i+1)%3,i%2==1)]
        schedule.append(rows)
    return schedule


def main():
    parser = argparse.ArgumentParser()
    for name in ("base","teacher","data","deepset-train","extra","out"):
        parser.add_argument("--"+name,required=True,type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    data,manifest = inputs(args.data)
    domain,probe,focus,dev,anchors,pool = prepare(args,data,rng)
    model = Model(args.base,args.teacher)
    model.consistency_view = "features"
    used = inherited_groups(args.teacher,model.info,None)
    parent = audit(model,data["calibration"],data["gate"],True)
    for view in parent["calibration_views"]:
        if any(sum(row)<4 or row[i]/sum(row)<.5 for i,row in enumerate(view["confusion"])):
            raise ValueError("Unreliable teacher view")
    pool = [r for r in pool if r["group"] not in used]
    rng.shuffle(pool)
    pool = pool[:2048]
    probs = [(model.scores(pool,v)/t).softmax(-1) for v,t in enumerate(parent["temperatures"])]
    selected,eligible = select(pool,*probs,parent["class_policy"],rng,64)
    if Counter(r["class"] for r in selected)!=Counter({0:64,1:64,2:64}):
        raise ValueError("Insufficient distinct pseudo-label groups")
    schedule = batches(model,domain,focus,anchors,parent["temperatures"][0],rng)
    config = adapter_dir(args.teacher)/"adapter_config.json"
    initial = model.weights()
    parent_sha = model.info["sha256"]
    parent_dev = metrics(model.scores(dev),dev)
    report = {"parent_sha256":parent_sha,"parent":parent,"parent_dev":parent_dev,
              "parent_deepset_probe":metrics(model.scores(probe),probe),"original_manifest":manifest,
              "seed":SEED,"anchor_target_mix":.10,"n_domain_train":len(domain),"n_domain_probe":len(probe),
              "n_focus":len(focus),"n_focus_dev":len(dev),"n_pool":len(pool),"eligible":eligible,
              "pseudo_per_class":64,"files_sha256":{f.name:hashlib.sha256(f.read_bytes()).hexdigest()
              for f in (args.deepset_train,args.extra,HERE/"focus-train.json",HERE/"focus-dev.json",HERE/"FOCUS-PROTOCOL.md")},
              "checkpoints":[],"complete":False}
    write(args.out/"audit.json",report)
    write(args.out/"pseudo-sources.json",[r["group"] for r in selected])
    for lr in (.000012,.000030):
        student = control = initial
        for stop in CHECKPOINTS:
            start = stop-24
            model.restore(control)
            update(model,schedule[start:stop],lr,.10)
            control = model.weights()
            metric_control = audit(model,data["calibration"],data["gate"])
            model.restore(student)
            update(model,schedule[start:stop],lr,.10,selected[2*start:2*stop],True)
            student = model.weights()
            name = f"lr-{round(lr*1e6):02d}-step-{stop:03d}"
            path = args.out/name
            sha = model.save(path,config,parent_sha,"targeted recursive student")
            metric = audit(model,data["calibration"],data["gate"])
            checks = decision(parent,metric,metric_control)
            newdev = metrics(model.scores(dev),dev)
            domain_dev = metrics(model.scores(probe),probe)
            safe_dev = all(newdev[k]<=parent_dev[k] for k in ("malicious_misses","risky_as_normal"))
            record = {"name":name,"sha256":sha,"steps":stop,"lr":lr,"original_checks":checks,
                      "passes":all(checks.values()) and safe_dev,"dev_guard":safe_dev,
                      "gate":metric,"control":metric_control,"dev":newdev,"deepset_probe":domain_dev}
            report["checkpoints"].append(record)
            write(args.out/"audit.json",report)
            print(name,record["passes"],metric["raw"]["correct"],newdev["confusion"][2],domain_dev["correct"],flush=True)
    def rank(r):
        review = r["dev"]["confusion"][2]
        return (r["passes"],review[2],-review[1],r["deepset_probe"]["correct"],
                r["gate"]["score"],-r["gate"]["scaled"]["brier"],-r["steps"])
    chosen = max(report["checkpoints"],key=rank)
    path = args.out/chosen["name"]
    # Only the pseudo-label groups actually consumed up to the selected checkpoint are inherited.
    used.update(r["group"] for r in selected[:2*chosen["steps"]])
    write(path/"source-groups.json",{"adapter_sha256":chosen["sha256"],"groups":sorted(used)})
    info = read(path/"report.json")
    info["source_groups_sha256"]=hashlib.sha256((path/"source-groups.json").read_bytes()).hexdigest()
    write(path/"report.json",info)
    report.update(complete=True,chosen=chosen["name"],chosen_sha256=chosen["sha256"],passes=chosen["passes"])
    write(args.out/"audit.json",report)
    print("SEALED",chosen["name"],chosen["sha256"],chosen["passes"],flush=True)

if __name__=="__main__":
    main()
