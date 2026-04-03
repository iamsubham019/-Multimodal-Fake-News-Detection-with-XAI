from transformers import AutoTokenizer
from data.dataset import generate_synthetic_dataset, split_records, get_dataloaders

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
records = generate_synthetic_dataset(100)
train, val, test = split_records(records)
print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

train_dl, val_dl, test_dl = get_dataloaders(train, val, test, tokenizer)
batch = next(iter(train_dl))
print("input_ids shape:", batch["input_ids"].shape)
print("label:", batch["label"])
print("graph_x items:", len(batch["graph_x"]))
print("DATA LOADING OK")