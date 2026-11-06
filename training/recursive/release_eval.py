"""Evaluate a frozen public release on the inference entry point, one request at a time."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch


def main():
    parser=argparse.ArgumentParser()
    for name in ("release","base","bundle","extra","out"):
        parser.add_argument("--"+name,required=True,type=Path)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    parser.add_argument("--plain-base",action="store_true")
    args=parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(2)
    spec=importlib.util.spec_from_file_location("released_inference",args.release/"inference.py")
    runtime=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    if (args.base/"REVISION").read_text().strip()!=runtime.REPORT["revision"]:
        raise ValueError("Unexpected base revision")
    detector=runtime.Detector(str(args.base),device=args.device)
    if args.plain_base:
        for layer in detector.model.model.layers:
            for name in ("q_proj","v_proj"):
                setattr(layer.self_attn,name,getattr(layer.self_attn,name).original)
    blob=args.bundle.read_bytes()
    extra=args.extra.read_bytes()
    sets=json.loads(blob)["test"]
    sets["november"]=[{"text":txt,"label":i,"group":case["group"]} for case in json.loads(extra)
                      for i,txt in enumerate(case["texts"])]
    report={"adapter_sha256":None if args.plain_base else runtime.EXPECTED,
            "base_revision":runtime.REPORT["revision"],"device":args.device,"batch":1,"dtype":"bfloat16",
            "bundle_sha256":hashlib.sha256(blob).hexdigest(),"november_sha256":hashlib.sha256(extra).hexdigest(),
            "sets":{}}
    for name,rows in sets.items():
        matrix=[[0]*3 for _ in range(3)]
        groups={}
        for row in rows:
            pred=runtime.REPORT["labels"].index(detector.classify(row["text"])["model_label"])
            label=row["label"]
            matrix[label][pred]+=1
            if "group" in row:
                group=row["group"]
                groups[group]=groups.get(group,True) and pred==label
        correct=sum(matrix[i][i] for i in range(3))
        report["sets"][name]={"n":len(rows),"correct":correct,"confusion":matrix,
                              "groups":len(groups),"complete_groups":sum(groups.values())}
        print(name,correct,"/",len(rows),flush=True)
    with args.out.open("x",encoding="utf-8") as f:
        json.dump(report,f,indent=2)

if __name__=="__main__":
    main()
