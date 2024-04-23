import torch
import os
import json
import torch
import tqdm
from config import get_config

def main():

    config = get_config()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    PEFT_MODEL = f"{config.hf_account}/{config.model_hf_name}"

    lora_config = PeftConfig.from_pretrained(PEFT_MODEL)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        lora_config.base_model_name_or_path,
        return_dict=True,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    tokenizer=AutoTokenizer.from_pretrained(lora_config.base_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    merge_model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-math-7b-rl").to(DEVICE), f"{config.hf_account}/{config.model_hf_name}")

    merge_model = merge_model.merge_and_unload()

    MODEL = f"{config.hf_account}/deepseek-math-7b-rl-zaloai-vllm"

    merge_model.save_pretrained("deepseek-math-7b-rl-zaloai-vllm")
    merge_model.push_to_hub(MODEL, use_auth_token=True)

    tokenizer.save_pretrained("deepseek-math-7b-rl-zaloai-vllm")
    tokenizer.push_to_hub(MODEL, use_auth_token=True)

if __name__ == '__main__':
    main()
