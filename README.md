# ZaloAI2023-Elementary-Math-Solving
## 2. Getting Started
```
git clone https://github.com/dinhquy-nguyen-1704/ZaloAI2023-Elementary-Math-Solving.git
cd ZaloAI2023-Elementary-Math-Solving
```
```
pip install -r requirements.txt
```
```
huggingface-cli login
wandb login
```
## 3. Finetune
I only utilize a dataset of over 1000 samples from the competition to fine-tune the model.

To rerun the fine-tuning code, you can execute the following command line.
```
python main.py --hf_account <HuggingFace account> --model_hf_name <HuggingFace model's name>
```
You can also find the fine-tuned model I've trained at <a href="https://huggingface.co/quynguyen1704/deepseek-math-7b-rl-zaloai"><b>[🤗 Models]</b></a>.
## 4. Inference
To infer a fine-tuned model with any elementary math multiple-choice question, you can run the following commands.
> Chain of Thought:
```
python inference_cot.py --hf_account <HuggingFace account> --model_hf_name <HuggingFace model's name>
```
> Few-shot Chain of Thought:
```
python inference_few_shot_cot.py --hf_account <HuggingFace account> --model_hf_name <HuggingFace model's name>
```
You can absolutely use the model I've fine-tuned for inference as well.
> Chain of Thought:
```
python inference_cot.py --hf_account quynguyen1704 --model_hf_name deepseek-math-7b-rl-zaloai
```
```
> Few-shot Chain of Thought:
```
python inference_few_shot_cot.py --hf_account quynguyen1704 --model_hf_name deepseek-math-7b-rl-zaloai
## 5. Evaluate
To evaluate the accuracy of the model on the private test set, you can run the following command:
> Chain of Thought:
```
python evaluate_cot.py --hf_account <HuggingFace account> --model_hf_name <HuggingFace model's name> --max_new_tokens <max new tokens>
```
> Few-shot Chain of Thought:
```
python evaluate_few_shot_cot.py --hf_account <HuggingFace account> --model_hf_name <HuggingFace model's name> --max_new_tokens <max new tokens>
```
You can also completely replace [my model](https://huggingface.co/quynguyen1704/deepseek-math-7b-rl-zaloai) with yours and give it a try.
## 6. Result
The following table summarizes the results of the model after fine-tuning
|        Model        | Max_new_tokens | CoT | Few-shot | Accuracy |
|---------------------|----------------|-----|----------|----------|
| deepseek-math-7b-rl | 500            | [x] |          |   67%    |
| deepseek-math-7b-rl | 1024           | [x] |          |   81%    |
| deepseek-math-7b-rl | 1024           | [x] |    [x]   |   80%    |
