import copy
import unittest

from focus_acceptance import acceptance
from run import decision


class FocusAcceptance(unittest.TestCase):
    def setUp(self):
        parent={"score":.90,"raw":{"malicious_misses":1,"risky_as_normal":0,"high_conf_wrong":1},
                "scaled":{"brier":.03}}
        student={**parent,"score":.91}
        self.selected={"name":"chosen","sha256":"new","gate":student,"control":parent,
                       "original_checks":decision(parent,student,parent),"dev_guard":True}
        self.trial={"complete":True,"chosen":"chosen","chosen_sha256":"new","parent_sha256":"old",
                    "checkpoints":[self.selected],"parent":parent,
                    "original_manifest":{"round3_bundle_sha256":"data"},"files_sha256":{"november-check.json":"extra"}}
        self.old={"adapter_sha256":"old","batch":1,"device":"cuda","dtype":"bfloat16","base_revision":"base",
                  "bundle_sha256":"data","november_sha256":"extra","sets":{"public":{"correct":112,"n":116},
                  "november":{"confusion":[[11,1,0],[0,12,0],[1,3,8]]}}}
        self.new=copy.deepcopy(self.old)
        self.new["adapter_sha256"]="new"
        self.new["sets"]["november"]["confusion"][2]=[1,1,10]

    def test_recovery_is_required_even_if_promotion_passed(self):
        self.assertTrue(acceptance(self.trial,self.old,self.new)["ship"])
        self.new["sets"]["public"]["correct"]=111
        self.assertFalse(acceptance(self.trial,self.old,self.new)["ship"])

    def test_moving_review_errors_to_normal_cannot_pass(self):
        self.new["sets"]["november"]["confusion"][2]=[4,0,8]
        self.assertFalse(acceptance(self.trial,self.old,self.new)["ship"])

    def test_refuse_wrong_checkpoint_or_dataset(self):
        self.new["adapter_sha256"]="some-other-checkpoint"
        with self.assertRaises(ValueError):
            acceptance(self.trial,self.old,self.new)
        self.new["adapter_sha256"]="new"
        self.new["november_sha256"]="different"
        with self.assertRaises(ValueError):
            acceptance(self.trial,self.old,self.new)
