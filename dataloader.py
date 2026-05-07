import json
from torch.utils.data import IterableDataset


class T5Dataset(IterableDataset):
    def __init__(self, data_path, tokenizer=None):
        self.data_path = data_path
        self.tokenizer = tokenizer

    def __iter__(self):
        with open(self.data_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                prompt = f'阅读理解: 内容:{sample["context"]}, 问题:{sample["question"]}'
                label = f'{sample["answer"]}'
                inputs = self.tokenizer(
                    prompt, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
                labels = self.tokenizer(
                    label, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
                output = {
                    'input_ids': inputs['input_ids'].squeeze(),
                    'attention_mask': inputs['attention_mask'].squeeze(),
                    'labels': labels['input_ids'].squeeze()
                }
                yield output
