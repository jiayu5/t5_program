from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments
from dataloader import T5Dataset


tokenizer = AutoTokenizer.from_pretrained("langboat/mengzi-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("langboat/mengzi-t5-base")

train_data = T5Dataset('data/DuReaderQG/train.json', tokenizer=tokenizer)
eval_data = T5Dataset('data/DuReaderQG/test.json', tokenizer=tokenizer)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    return_tensors='pt'
)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=3,
    logging_dir="./logs",
    logging_steps=10,
    learning_rate=1e-5,
    max_steps=5000,
    eval_strategy="steps",
    eval_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="tensorboard",
    greater_is_better=False,
    save_total_limit=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model("./results/mengzi-t5-base-finetuned")
