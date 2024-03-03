import configargparse
import torch
from transformers import AutoTokenizer
from model import MambaTextClassification


# Config
parser = configargparse.ArgumentParser()
parser.add_argument("--infer_data", type=str)
args = parser.parse_args()

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
path = "hautran7201/mamba_text_classification"
model = MambaTextClassification.from_pretrained(path, device=device)
tokenizer_path = 'EleutherAI/gpt-neox-20b'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Predict
id2label = {0: " NEGATIVE ", 1: " POSITIVE "}
response = model.predict(
    args.infer_data,
    tokenizer,
    id2label
)
print()
print("Result:", response)
