"""Read test labels only after all supplied trials have stopped and written their decisions."""
import argparse
import hashlib
import json
from pathlib import Path
from safetensors.torch import load_file
from run import Model, metrics, write
from model import adapter_dir

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--base",type=Path,required=True)
    p.add_argument("--bundle",type=Path,required=True)
    p.add_argument("--trials",type=Path,nargs="+",required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    candidates={}
    roles={}
    blob=args.bundle.read_bytes()
    for path in args.trials:
        trial=json.loads((path/"trial.json").read_text(encoding="utf-8"))
        if not trial["complete"]:raise ValueError("Promotion decision is still open")
        assert hashlib.sha256(blob).hexdigest()==trial["protocol"]["input_manifest"]["round3_bundle_sha256"]
        last=trial["start_sha256"]
        candidates[last]=Path(trial["start"])
        roles[path.name+"/teacher"]=last
        for r in trial["rounds"]:
            assert r["teacher_sha256"]==last
            for key,name in (("control_sha256","gold-only"),("student_sha256","student")):
                if key in r:
                    candidates[r[key]]=path/f"round-{r['round']}"/name
                    roles[f"{path.name}/round-{r['round']}/{name}"]=r[key]
            if r["promoted"]:last=r["student_sha256"]
        assert last==trial["active_sha256"]
        roles[path.name+"/accepted"]=last
    tests=json.loads(blob)["test"]
    model=Model(args.base,next(iter(candidates.values())))
    results={}
    for sha,folder in candidates.items():
        f=adapter_dir(folder)/"adapter_model.safetensors"
        assert hashlib.sha256(f.read_bytes()).hexdigest()==sha
        model.restore(load_file(str(f)))
        results[sha]={}
        for name,rows in tests.items():
            logits=model.scores(rows)
            m=metrics(logits,rows)
            if name=="context":
                groups={}
                for r,pred in zip(rows,logits.argmax(-1).tolist()):
                    g=r["group"];groups[g]=groups.get(g,True) and pred==r["label"]
                m.update(complete=sum(groups.values()),groups=len(groups))
            results[sha][name]=m
        print(sha[:12],{k:(m["correct"],m["n"]) for k,m in results[sha].items()},flush=True)
    write(args.output,{"device":"CUDA BF16","batch_size":4,"test_sha256":hashlib.sha256(blob).hexdigest(),
                       "scope":"Post-hoc diagnostic; selection files are read-only.","roles":roles,"results":results})

if __name__=="__main__":
    main()
