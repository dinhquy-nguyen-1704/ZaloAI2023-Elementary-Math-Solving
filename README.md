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
