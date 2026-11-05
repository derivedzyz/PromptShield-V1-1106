import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
REPORT = json.loads((ROOT / "report.json").read_text(encoding="utf-8"))
EXPECTED = "6ce4ed9c6c8237a8e5cfe7c56e523a15c48d794f6cb1b611665c172a3b475559"


class Projection(torch.nn.Module):
    """Frozen rank-8 additive LoRA, alpha=16, for this run's Q/V projections."""
    def __init__(self, original, a, b):
        super().__init__()
        self.original = original
        assert a.shape == (8, original.in_features) and b.shape == (original.out_features, 8)
        self.register_buffer("a", a.to(original.weight.dtype))
        self.register_buffer("b", b.to(original.weight.dtype))

    def forward(self, x):
        return self.original(x) + 2 * F.linear(F.linear(x, self.a), self.b)


class Detector:
    def __init__(self, base=None, device="cuda"):
        weights = ROOT / "adapter_model.safetensors"
        if hashlib.sha256(weights.read_bytes()).hexdigest() != EXPECTED:
            raise RuntimeError("Incorrect adapter weights")
        source = base or REPORT["base"]
        options = {"local_files_only": True} if base else {"revision": REPORT["revision"]}
        self.tokenizer = AutoTokenizer.from_pretrained(source, **options)
        self.model = AutoModelForCausalLM.from_pretrained(source, dtype=torch.bfloat16, **options)
        delta = load_file(str(weights))
        for i, layer in enumerate(self.model.model.layers):
            for name in ("q_proj", "v_proj"):
                key = f"base_model.model.model.layers.{i}.self_attn.{name}"
                tuned = Projection(getattr(layer.self_attn, name), delta.pop(key+".lora_A.weight"),
                                   delta.pop(key+".lora_B.weight"))
                setattr(layer.self_attn, name, tuned)
        assert not delta
        self.device = device
        self.model.eval().to(device)
        self.prefix = self.encode(REPORT["prompt"])
        self.suffix = self.encode(REPORT["suffix"])

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    @torch.inference_mode()
    def classify(self, text):
        if not isinstance(text, str) or not 0 < len(text) <= 20000:
            raise ValueError("Expected 1 to 20000 characters")
        ids = self.encode(json.dumps(text, ensure_ascii=False))
        cap = REPORT["max_tokens"] - len(self.prefix) - len(self.suffix)
        if len(ids) > cap:
            ids = ids[:cap//2] + ids[-(cap-cap//2):]
        x = torch.tensor([self.prefix + ids + self.suffix], device=self.device)
        hidden = self.model.model(input_ids=x, use_cache=False).last_hidden_state[0,-1]
        head = self.model.get_output_embeddings().weight[REPORT["label_token_ids"]]
        probs = F.linear(hidden, head).float().softmax(-1)
        return {"model_version": "promptshield-qwen2.5-0.5b-lora-"+EXPECTED[:12],
                "model_label": REPORT["labels"][int(probs.argmax())],
                "probabilities": dict(zip(REPORT["labels"], probs.tolist())),
                "calibrated": False}
