import transformers
from config import get_config


def train(model, tokenizer, choices_data):

    config = get_config()

    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        fp16=config.fp16,
        save_total_limit=config.save_total_limit,
        logging_steps=config.logging_steps,
        output_dir=config.output_dir,
        optim=config.optim,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=choices_data,
        args=training_args,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    model.config.use_cache = False
    trainer.train()
