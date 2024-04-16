import json
import tqdm
from utils.func import generate_and_tokenize_prompt
from datasets import load_dataset, Dataset
from utils.generate_prompt import generate_prompt_test, generate_few_shot_prompt

def process_data_train(file_path, tokenizer):

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    training_samples = []

    for sample in data["data"]:
        try:
            choices = sample['choices']
        except:
            break
        explanation = sample['explanation'].strip()
        question = sample['question']
        answer = sample['answer']

        choices = '\n'.join(choices)
        training_sample = generate_and_tokenize_prompt(
            tokenizer, question, choices, explanation, answer
        )

        training_samples.append(training_sample)

        choices_data = Dataset.from_list(training_samples)

    return choices_data


def process_data_cot(file_path, tokenizer):

    # Load the JSON file
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    test_samples = []

    for sample in data["data"]:
        try:
            choices = sample['choices']
        except:
            break
        question = sample['question']

        choices = '\n'.join(choices)
        test_sample = generate_prompt_test(
            question, choices
        )

        test_samples.append(test_sample)

    return test_samples


def process_data_few_shot_cot(file_path, tokenizer):

    # Load the JSON file
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    test_samples = []

    for sample in data["data"]:
        try:
            choices = sample['choices']
        except:
            break
        question = sample['question']

        choices = '\n'.join(choices)
        test_sample = generate_few_shot_prompt(
            question, choices
        )

        test_samples.append(test_sample)

    return test_samples

def parse_json_test_to_lists(file_name):

    with open(file_name) as json_file:
        json_test = json.load(json_file)

    list_id = []
    list_question = []
    list_A = []
    list_B = []
    list_C = []
    list_D = []
    list_answer = []

    for record in json_test['data']:

        id = record['id']
        question = record['question']
        choices = record['choices']
        answer = record['answer'][0]

        list_A.append(choices[0])
        list_B.append(choices[1])
        list_C.append(choices[2])
        try:
          list_D.append(choices[3])
        except IndexError:
          list_D.append("None")

        list_id.append(id)
        list_question.append(question)
        list_answer.append(answer)

    return {
      "list_id":list_id,
      "list_question":list_question,
      "list_A":list_A,
      "list_B":list_B,
      "list_C":list_C,
      "list_D":list_D,
      "list_answer":list_answer
    }
