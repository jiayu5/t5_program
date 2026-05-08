import numpy as np
import numpy as np
import torch
import evaluate
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments

from dataloader import T5Dataset

model_path = "./results/mengzi-t5-base-finetuned"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

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
    per_device_train_batch_size=2,
    logging_dir="./logs",
    logging_steps=10,
    eval_steps=100,
    learning_rate=1e-5,
    max_steps=1000,
    do_predict=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator
)

def print_predictions(predictions, labels):
    bleu = evaluate.load('bleu')
    for i in range(min(5, len(predictions))):
        print(f"Prediction: {predictions[i]}")
        print(f"Label: {labels[i]}")
        print("-" * 50)
    print(f"BLEU Score: {bleu.compute(predictions=predictions[i:i+1], references=[[label] for label in labels[i:i+1]], smooth=True)}")

def predict_with_trainer():
    predict_results = trainer.predict(test_dataset=eval_data)
    pred_array = predict_results.predictions[0] if isinstance(
        predict_results.predictions, tuple) else predict_results.predictions
    predictions = np.argmax(pred_array, axis=-1)
    labels = predict_results.label_ids
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_predictions = tokenizer.batch_decode(
        predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    print_predictions(decoded_predictions, decoded_labels)


def predict_with_dataloaders():
    predict_dataloaders = DataLoader(
        eval_data, batch_size=2, collate_fn=data_collator)

    model.eval()

    print("Starting prediction...")
    with torch.no_grad():
        for batch in predict_dataloaders:
            input_ids = batch['input_ids'].to(trainer.args.device)
            attention_mask = batch['attention_mask'].to(trainer.args.device)
            labels = batch['labels'].to(trainer.args.device)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128,
                num_beams=4
            )

            predicts = tokenizer.batch_decode(
                outputs, skip_special_tokens=True)
            labels[labels == -100] = tokenizer.pad_token_id
            decoded_labels = tokenizer.batch_decode(
                labels, skip_special_tokens=True)
            
            print_predictions(predicts, decoded_labels)


# predict_with_trainer()
predict_with_dataloaders()
