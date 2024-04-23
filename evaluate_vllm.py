from vllm import LLM, SamplingParams
from config import get_config
from process_data import process_data_cot, parse_json_test_to_lists
import pandas as pd
import re
import json

def main():

    config = get_config()

    llm = LLM(model=f"{config.hf_account}/deepseek-math-7b-rl-zaloai-vllm",
              dtype='auto',
              enforce_eager=True,
              gpu_memory_utilization=0.99,
              swap_space=4,
              max_model_len=2048,
              kv_cache_dtype="fp8_e5m2",
              tensor_parallel_size=1)
    
    test_samples = process_data_cot(config.dataset_test)

    sampling_params = SamplingParams(temperature=config.temperature, top_p=config.top_p, max_tokens=config.max_new_tokens)

    outputs = llm.generate(test_samples, sampling_params)

    results = []

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text

        try:
            result = re.findall(r'\\boxed\{(.*)\}', generated_text)[-1]

        except:
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
