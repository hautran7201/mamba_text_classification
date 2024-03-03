import random

class Imdb_dataset:
    def __init__(self, imdb, tokenizer):
        self.imdb = imdb
        self.tokenizer = tokenizer
    
    def get_tokenized_dataset(self, split='Train', eval_ratio=0.1):
        if split in self.imdb:
            return self.imdb[split].map(
                self.preprocess_function,
                batched=True,
                remove_columns=['text']
            )

        elif split in ['eval'] and 'test' in self.imdb:
            test_dataset = self.imdb['test']
            total_samples = len(test_dataset)
            eval_samples = int(eval_ratio*total_samples)
            eval_indices = random.sample(range(total_samples), eval_samples)
            eval_dataset = test_dataset.select(eval_indices)
            return eval_dataset.map(
                self.preprocess_function,
                batched=True,
                remove_columns=['text']
            )
        else:
            print('Can not get data')
            return None

    def preprocess_function(self, examples):
        samples = self.tokenizer(
            examples['text'],
            max_length=128,
            truncation=True,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt"
        )

        samples.pop('attention_mask')

        return samples

    
    
