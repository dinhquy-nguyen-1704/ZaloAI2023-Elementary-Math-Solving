import os
import json
import pandas as pd
import torch
import tqdm
import re
from config import get_config

from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training
)

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)


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

    model = PeftModel.from_pretrained(model, PEFT_MODEL).to(DEVICE)

    generation_config = model.generation_config
    generation_config.max_new_tokens = config.max_new_tokens
    generation_config.temperature = config.temperature
    generation_config.top_p = config.top_p
    generation_config.num_return_sequences = config.num_return_sequences
    generation_config.pad_token_id = tokenizer.eos_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id

    prompt = """
    <|im_start|>system
    You are an expert in math. You will receive multiple choice questions with options, solve step by step if available and choose the correct option.

    <|im_start|>user
    ### Question:
    Hương và Hồng hẹn nhau lúc 10 giờ 40 phút sáng. Hương đến chỗ hẹn lúc 10 giờ 20 phút còn Hồng lại đến muộn mất 15 phút. Hỏi Hương phải đợi Hồng trong bao nhiêu lâu?
    ### Choices:
    A. 20 phút
    B. 35 phút
    C. 55 phút
    D. 1 giờ 20 phút
    Please reason step by step, and put your final answer within \\boxed{}.
    ### Answer:

    <|im_start|>assistant
    """.strip()

    encoding = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=encoding.input_ids,
            attention_mask=encoding.attention_mask,
            generation_config=generation_config
        )

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == '__main__':
    main()
