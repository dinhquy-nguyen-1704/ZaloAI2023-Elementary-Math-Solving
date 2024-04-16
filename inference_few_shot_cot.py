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
    Below are some examples of tasks and their appropriate responses. After the examples, there is a new task. Write a response that appropriately completes the request.

    ### Example 1:

    <|im_start|>user
    ### Question:
    Chữ số 5 trong số 162,57 thuộc ……………………….
    ### Choices:
    A. Hàng đơn vị và có giá trị $\\\\frac{5}{10}$
    B. Hàng phần mười và có giá trị $\\\\frac{5}{10}$
    C. Hàng đơn vị và có giá trị 5
    D. D. 26 605 + 8 125
    Please reason step by step, and put your final answer within \\boxed{}.
    ### Answer:

    <|im_start|>assistant
    Trong số 162,57, chữ số 5 đứng sau dấu phẩy, nên nó thuộc hàng phần mười. \n\nVề giá trị của chữ số 5, chúng ta cần xem xét vị trí của nó trong số. Trong trường hợp này, chữ số 5 đứng ở vị trí sau dấu phẩy đầu tiên, nghĩa là nó đại diện cho $\\frac{5}{10}$ hoặc 0.5, chứ không phải là 5. \n\nVì vậy, câu trả lời chính xác là \"Hàng phần mười và có giá trị $\\frac{5}{10}$
    The answer is $\\boxed{B}$.

    ### Example 2:

    <|im_start|>user
    ### Question:
    Một hình bình hành có độ dày đáy bằng $\\frac{3}{2}$ dm và chiều cao bằng $\\frac{1}{2}$ độ dài đáy. Diện tích của hình bình hành là:
    ### Choices:
    A. $\\\\frac{9}{4}$ dm2
    B. $\\\\frac{9}{16}$ dm2
    C. $\\\\frac{9}{8}$ dm2
    D. $\\\\frac{3}{4}$ dm2
    Please reason step by step, and put your final answer within \\boxed{}.
    ### Answer:

    <|im_start|>assistant
    Diện tích hình bình hành bằng độ dài cạnh đáy nhân với chiều cao.\n Chiều cao của hình bình hành là: $\\frac{3}{2}$ ${\\times}$ $\\frac{1}{2}$ = $\\frac{3}{4}$ (dm)\n Diện tích hình bình hành đó là: $\\frac{3}{2}$ ${\\times}$ $\\frac{3}{4}$ = $\\frac{9}{8}$ (dm2)\n Đáp số: $\\frac{9}{8}$ dm2.
    The answer is $\\boxed{C}$.

    ### New task:

    <|im_start|>system
    You are an expert in math. You will receive multiple choice questions with options, solve step by step if available and choose the correct option.

    <|im_start|>user
    ### Question:
    Giá trị của chữ số 8 trong số thập phân 50,289:
    ### Choices:
    A. 8
    B. \\frac{8}{10}
    C. \\frac{8}{100}
    D. 80
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
