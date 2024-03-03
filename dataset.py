import random

class Imdb_dataset:
    def __init__(self, imdb, tokenizer):
        self.imdb = imdb
        self.tokenizer = tokenizer
    
    def get_tokenized_dataset(self, split='Train', eval_ratio=0.1):
        if split in self.imdb:
            return self.imdb[split].map(
                self.preprocess_function,
                batched=True
            )
        elif split in ['eval'] and 'test' in self.imdb:
            test_dataset = self.imdb[split]
            total_samples = len(test_dataset)
            eval_samples = int(eval_ratio*total_samples)
            eval_indices = random.sample(range(total_samples), eval_samples)
            eval_dataset = test_dataset.select(eval_indices)
            return eval_dataset.map(
                self.preprocess_function,
                batched=True
            )
        else:
            return None

    def preprocess_function(self, examples):
        samples = self.tokenizer(
            examples['text'],
            truncation=True
        )

        samples.pop('attention_mask')
        return samples

    
    
