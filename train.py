import evaluate
import os 
from huggingface_hub import login
from dataset import Imdb_dataset
from datasets import load_dataset
from utils import compute_metrics
from transformers import AutoTokenizer , TrainingArguments
from model import MambaTextClassification, MambaTrainer


# Huggingface token
token = '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'
login(token=token, write_permission=True)

# Model
model_path = 'state-spaces/mamba-130m'
model = MambaTextClassification.from_pretrained(model_path)

# Tokenizer
tokenizer_path = 'EleutherAI/gpt-neox-20b'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
tokenizer.pad_token_id = tokenizer.eos_token_id

# Metric 
metric = evaluate.load("accuracy")

# Data
imdb = load_dataset('imdb')
imdb = imdb.rename_column("label", "labels")
imdbDataset = Imdb_dataset(imdb, tokenizer)
train_dataset = imdbDataset.get_tokenized_dataset('train')
train_dataset.set_format("torch")
eval_dataset = imdbDataset.get_tokenized_dataset('eval')
eval_dataset.set_format("torch")

# Arguments
arguments = TrainingArguments(
    output_dir='mamba_text_classification',
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    warmup_steps=0.01,
    lr_scheduler_type='cosine',
    evaluation_strategy='steps',
    eval_steps=10,
    push_to_hub=True,
    load_best_model_at_end=True,
    report_to=None,
)


trainer = MambaTrainer(
    model=model,
    args=arguments,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.push_to_hub()


