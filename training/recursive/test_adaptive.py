import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest

import torch

from adaptive import fit_policy, inherited_groups, select
from run import contrast_schedule


class AdaptiveTraining(unittest.TestCase):
    def test_counterfactual_triplets_require_training_labels(self):
        class FakeModel:
            def scores(self,rows):
                return torch.zeros((len(rows),3))
        anchors=[{"text":text,"label":i} for i,text in enumerate(["general","execute","review"])]
        cases=[{"texts":[r["text"] for r in anchors],"group":"one"}]
        batches=[[dict(r) for r in anchors for _ in range(2)] for _ in range(3)]
        contrast_schedule(FakeModel(),batches,anchors,cases,random.Random(1),1.)
        self.assertEqual([r["label"] for r in batches[0]],[0,1,2,0,1,2])
        self.assertEqual([r["label"] for r in batches[1]],[0,0,1,1,2,2])
        cases[0]["texts"][2]="held-out"
        with self.assertRaisesRegex(ValueError,"labelled anchor"):
            contrast_schedule(FakeModel(),batches,anchors,cases,random.Random(1),1.)

    def test_small_classes_need_clean_support_at_a_higher_floor(self):
        rows = torch.tensor([[.85,.1,.05]]*12 + [[.01,.98,.01]]*6 + [[.01,.01,.98]]*4)
        truth = torch.tensor([0]*12 + [1]*6 + [2]*4)
        rules = fit_policy(rows, rows, truth)
        self.assertEqual([r["threshold"] for r in rules], [.82,.94,None])
        truth[12] = 0
        self.assertIsNone(fit_policy(rows, rows, truth)[1]["threshold"])

    def test_agreement_distance_weight_and_source_cap(self):
        a = torch.tensor([[.99,.01,0.],[.84,.16,0.],[.99,.01,0.],[.98,.02,0.],[.99,.01,0.]])
        b = a.clone()
        b[3] = torch.tensor([.83,.17,0.])
        b[4] = torch.tensor([.01,.99,0.])
        rows = [{"group":g,"text":str(i)} for i,g in enumerate("abacd")]
        selected, eligible = select(rows,a,b,[{"threshold":.82}]*3,random.Random(4),32)
        self.assertEqual(eligible,{0:3})
        weights={r["group"]:r["weight"] for r in selected}
        self.assertEqual(set(weights),{"a","b"})
        self.assertGreater(weights["a"],weights["b"])
        self.assertGreater(weights["b"],0)

    def test_accepted_ledger_is_bound_to_weight_and_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            f=root/"source-groups.json"
            f.write_text(json.dumps({"adapter_sha256":"parent","groups":["doc-a","doc-b"]}))
            info={"sha256":"parent","source_groups_sha256":hashlib.sha256(f.read_bytes()).hexdigest()}
            self.assertEqual(inherited_groups(root,info,None),{"doc-a","doc-b"})
            with self.assertRaisesRegex(ValueError,"another teacher"):
                inherited_groups(root,{**info,"sha256":"other"},None)
            f.write_text("[]")
            with self.assertRaisesRegex(ValueError,"fingerprint"):
                inherited_groups(root,info,None)
            with self.assertRaisesRegex(ValueError,"requires"):
                inherited_groups(root,{"role":"experimental student"},None)

if __name__=="__main__":
    unittest.main()
