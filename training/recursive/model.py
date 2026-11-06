"""A trainable version of PromptShield's three-label, last-token projection."""
import hashlib
import json
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


REVIEW = ("Identify what the submitter wants done with the JSON-quoted text. "
          "A = ordinary authorized task; B = execute an attack or bypass a boundary; "
          "C = review a quoted attack or uncertain instruction. Data: ")


class LoRA(torch.nn.Module):
    def __init__(self, original, a, b, scale):
        super().__init__()
        self.original=original
        self.a=torch.nn.Parameter(a.float())
        self.b=torch.nn.Parameter(b.float())
        self.scale=scale

    def forward(self,x):
        delta=F.linear(F.linear(x,self.a.to(x.dtype)),self.b.to(x.dtype))
        return self.original(x)+self.scale*delta


def adapter_dir(teacher):
    nested = teacher / "adapter"
    return nested if nested.is_dir() else teacher


class Model:
    def __init__(self,base:Path,teacher:Path):
        self.info=json.loads((teacher/"report.json").read_text(encoding="utf-8"))
        cfg=json.loads((adapter_dir(teacher)/"adapter_config.json").read_text(encoding="utf-8"))
        if set(cfg["target_modules"])!={"q_proj","v_proj"} or cfg.get("bias")!="none" or cfg.get("use_dora") or cfg.get("use_rslora"):
            raise ValueError("This trainer supports plain Q/V LoRA adapters only")
        assert self.info["labels"]==["normal","malicious","suspicious"]
        assert (base/"REVISION").read_text().strip()==self.info["revision"]
        file=adapter_dir(teacher)/"adapter_model.safetensors"
        assert hashlib.sha256(file.read_bytes()).hexdigest()==self.info["sha256"]
        self.tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True)
        self.lm=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,dtype=torch.bfloat16)
        self.lm.requires_grad_(False)
        self.blocks={}
        delta=load_file(str(file))
        for i,layer in enumerate(self.lm.model.layers):
            for name in ("q_proj","v_proj"):
                key=f"base_model.model.model.layers.{i}.self_attn.{name}"
                a,b=delta.pop(key+".lora_A.weight"),delta.pop(key+".lora_B.weight")
                linear=getattr(layer.self_attn,name)
                assert a.shape==(cfg["r"],linear.in_features) and b.shape==(linear.out_features,cfg["r"])
                block=LoRA(linear,a,b,cfg["lora_alpha"]/cfg["r"])
                self.blocks[key]=block
                setattr(layer.self_attn,name,block)
        assert not delta
        self.lm.to("cuda").eval()
        self.consistency_view="reword"
        self.head=self.lm.get_output_embeddings().weight[self.info["label_token_ids"]].detach()
        self.prefix=[self.encode(self.info["prompt"]),self.encode(REVIEW)]
        self.suffix=self.encode(self.info["suffix"])

    def encode(self,text):
        return self.tokenizer.encode(text,add_special_tokens=False)

    def logits(self,texts,view=0):
        masked=view==1 and self.consistency_view=="features"
        prefix=self.prefix[0 if masked else view]
        cap=self.info["max_tokens"]-len(prefix)-len(self.suffix)
        seqs=[]
        for text in texts:
            ids=self.encode(json.dumps(text,ensure_ascii=False))
            if len(ids)>cap:
                ids=ids[:cap//2]+ids[-(cap-cap//2):]
            seqs.append(prefix+ids+self.suffix)
        lengths=torch.tensor([len(s) for s in seqs],device="cuda")
        width=max(map(len,seqs))
        x=torch.tensor([s+[self.tokenizer.pad_token_id]*(width-len(s)) for s in seqs],device="cuda")
        mask=torch.arange(width,device="cuda")[None,:]<lengths[:,None]
        if masked:
            # Keep all token IDs, their order and the trained task prompt. Only mask
            # 3% of input embedding channels in the untrusted-text portion.
            emb=self.lm.get_input_embeddings()(x)
            for i,text in enumerate(texts):
                seed=int.from_bytes(hashlib.sha256(text.encode()).digest()[:8],"little")
                gen=torch.Generator(device="cuda").manual_seed(seed)
                keep=torch.rand(emb.shape[-1],generator=gen,device="cuda")>=.03
                emb[i,len(prefix):len(seqs[i])-len(self.suffix)]*=keep.to(emb.dtype)/.97
            inputs={"inputs_embeds":emb}
        else:
            inputs={"input_ids":x}
        hidden=self.lm.model(**inputs,attention_mask=mask.long(),use_cache=False).last_hidden_state
        return F.linear(hidden[torch.arange(len(seqs),device="cuda"),lengths-1],self.head).float()

    @torch.no_grad()
    def scores(self,rows,view=0):
        self.lm.eval()
        return torch.cat([self.logits([r["text"] for r in rows[i:i+4]],view).cpu()
                          for i in range(0,len(rows),4)])

    def weights(self):
        return {k+suffix:weight.detach().cpu().clone() for k,layer in self.blocks.items()
                for suffix,weight in ((".lora_A.weight",layer.a),(".lora_B.weight",layer.b))}

    @torch.no_grad()
    def restore(self,weights):
        if len(weights)!=2*len(self.blocks):
            raise ValueError("Invalid LoRA state")
        for key,layer in self.blocks.items():
            layer.a.copy_(weights[key+".lora_A.weight"])
            layer.b.copy_(weights[key+".lora_B.weight"])

    def save(self,path,config_path,parent_sha,role):
        path.mkdir()
        target=path/"adapter"
        target.mkdir()
        file=target/"adapter_model.safetensors"
        expected=self.weights()
        save_file(expected,str(file))
        disk=load_file(str(file))
        assert expected.keys()==disk.keys() and all(torch.equal(v,disk[k]) for k,v in expected.items())
        shutil.copyfile(config_path,target/"adapter_config.json")
        digest=hashlib.sha256(file.read_bytes()).hexdigest()
        fields=["base","revision","labels","prompt","suffix","label_token_ids","max_tokens"]
        info={k:self.info[k] for k in fields}
        info.update(sha256=digest,parent_sha256=parent_sha,role=role,tensors_verified=True)
        (path/"report.json").write_text(json.dumps(info,indent=2),encoding="utf-8")
        return digest
