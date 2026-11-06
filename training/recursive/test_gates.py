import copy
import hashlib
import json
import random
import tempfile
import unittest
from pathlib import Path

import torch
from run import admit,decision,inputs


class Gates(unittest.TestCase):
    def test_better_average_cannot_hide_extra_malicious_misses(self):
        parent={"score":.90,"raw":{"malicious_misses":1,"risky_as_normal":0,"high_conf_wrong":1},"scaled":{"brier":.04}}
        student=copy.deepcopy(parent)
        student["score"]+=.01
        student["raw"]["malicious_misses"]+=1
        checks=decision(parent,student,parent)
        self.assertTrue(checks["measured_benefit"])
        self.assertFalse(checks["malicious_misses_preserved"])

    def test_disagreement_low_confidence_and_repeated_source_are_filtered(self):
        class TwoViews:
            def scores(self,rows,i):
                scores=[[9.,0.,0.],[9.,0.,0.],[0.,9.,0.],[0.,0.,.1],[0.,0.,9.]]
                if i:scores[2]=[0.,0.,9.]
                return torch.tensor(scores)
        pool=[{"text":str(i),"group":g} for i,g in enumerate(["a","a","b","c","d"])]
        accepted,qualified=admit(TwoViews(),pool,[1.,1.],random.Random(0))
        self.assertEqual(qualified,3)
        self.assertEqual({r["group"] for r in accepted},{"a","d"})
        self.assertEqual(len(accepted),2)

    def test_custom_pool_cannot_include_labels_or_eval_text(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            for variant,message in (({"text":"unlabelled","label":2,"group":"p"},"only text and group"),
                                    ({"text":"test question","group":"p"},"overlaps")):
                data={"anchors":{"text":"gold sample","label":1},"gate":{"text":"gate row","label":0},
                      "calibration":{"text":"calibration case","label":2},"exclusion-texts":{"text":"test question"},
                      "pool":variant}
                hashes={}
                for name,row in data.items():
                    f=folder/(name+".jsonl")
                    f.write_text(json.dumps(row),encoding="utf-8")
                    hashes[f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
                (folder/"manifest.json").write_text(json.dumps({"file_sha256":hashes}),encoding="utf-8")
                with self.assertRaisesRegex(ValueError,message):inputs(folder)

if __name__=="__main__":
    unittest.main()
