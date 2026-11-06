"""Fingerprint five user-provided JSONL splits in the format accepted by run.py."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument("directory",type=Path)
    p.add_argument("--bundle",type=Path,help="Optional frozen JSON bundle used by evaluate_trial.py")
    args=p.parse_args()
    names=("anchors","calibration","gate","pool","exclusion-texts")
    files=[args.directory/(name+".jsonl") for name in names]
    result={"file_sha256":{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}
    if args.bundle:
        result["round3_bundle_sha256"]=hashlib.sha256(args.bundle.read_bytes()).hexdigest()
    with (args.directory/"manifest.json").open("x",encoding="utf-8") as f:
        json.dump(result,f,indent=2)

if __name__=="__main__":
    main()
