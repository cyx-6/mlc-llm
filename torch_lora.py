from safetensors.torch import save_file, load_file
import torch
import glob
from collections import defaultdict
import os
import json


lora_path = "./dist/models/Llama2-Chinese-7b-Chat-LoRA/"
with open(os.path.join(lora_path, "adapter_config.json"), "r") as lora_config_file:
    lora_config = json.load(lora_config_file)
scale = 1.0 * lora_config["lora_alpha"] / lora_config["r"]
adapter_bin = torch.load(os.path.join(lora_path, "adapter_model.bin"))
lora_tensors = defaultdict(dict)
for k, v in adapter_bin.items():
    k = k.replace("base_model.model.", "")
    if k.count("lora_A"):
        lora_tensors[k.replace("lora_A.", "")]["lora_a"] = v
    elif k.count("lora_B"):
        lora_tensors[k.replace("lora_B.", "")]["lora_b"] = v * scale
    else:
        raise ValueError("Unexpected param in lora weight")
del adapter_bin

model_path = "./dist/models/llama-2-7b-chat-hf"
safetensors_files = glob.glob(model_path + r"/*.safetensors")

for sf in safetensors_files:
    base_model_tensors = load_file(sf, device="cuda:0")
    file_name = os.path.basename(sf)

    for k, v in lora_tensors.items():
        if k in base_model_tensors:
            base_model_tensors[k] = base_model_tensors[k] + torch.matmul(v["lora_b"], v["lora_a"])
    save_file(base_model_tensors, os.path.join(lora_path, file_name))
