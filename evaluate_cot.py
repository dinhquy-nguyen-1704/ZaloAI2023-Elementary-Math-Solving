import os
import json
import pandas as pd
import torch
import tqdm
import re

from process_data import process_data_cot, parse_json_test_to_lists
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

    test_samples = process_data_cot(config.dataset_test, tokenizer)

    results = []

    for problem in test_samples:

        encoding = tokenizer(problem, return_tensors="pt").to(DEVICE)
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=encoding.input_ids,
                attention_mask=encoding.attention_mask,
                generation_config=generation_config
            )

        solution = tokenizer.decode(outputs[0], skip_special_tokens=True)

        result = re.findall(r'\\boxed\{(.*)\}', solution)[-1]

        if result == "":
            result = 'E'

        results.append(result)

    dic = parse_json_test_to_lists(config.dataset_test)

    list_id = dic["list_id"]
    list_question = dic["list_question"]
    list_A = dic["list_A"]
    list_B = dic["list_B"]
    list_C = dic["list_C"]
    list_D = dic["list_D"]
    list_answer = dic["list_answer"]

    df_test = pd.DataFrame(list(zip(list_id, list_question, list_A, list_B, list_C, list_D, list_answer, results)),
                          columns=['id', 'question', 'A', 'B', 'C', 'D', 'answer', 'result'])

    correct = (df_test['answer'] == df_test['result']).sum()

    total = len(df_test)

    accuracy = correct / total

    print("Accuracy:", accuracy)

if __name__ == '__main__':
    main()
