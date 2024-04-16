import argparse

def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='deepseek-ai/deepseek-math-7b-rl')
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--num_return_sequences', type=int, default=1)
    parser.add_argument('--dataset_train', type=str, default='./datasets/train.json')
    parser.add_argument('--dataset_test', type=str, default='./datasets/test.json')
    parser.add_argument('--output_dir', type=str, default='experiments')
    parser.add_argument('--per_device_train_batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--num_train_epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--fp16', type=bool, default=True)
    parser.add_argument('--save_total_limit', type=int, default=3)
    parser.add_argument('--logging_steps', type=int, default=1)
    parser.add_argument('--optim', type=str, default='paged_adamw_8bit')
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine')
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--lora_r', type=int, default=4)
    parser.add_argument('--lora_alpha', type=int, default=4)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--push_to_hub', type=bool, default=True)
    parser.add_argument('--hf_account', type=str, default=None)
    parser.add_argument('--model_hf_name', type=str, default='deepseek-math-7b-rl-zaloai')

    args = parser.parse_args()

    return args
