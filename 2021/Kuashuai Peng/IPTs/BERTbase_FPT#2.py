from transformers import BertTokenizerFast, AutoTokenizer, AutoModelForMaskedLM
from transformers import default_data_collator
from transformers import AdamW, get_constant_schedule_with_warmup
from torch.utils.data import DataLoader
from datasets import Dataset
import collections
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
plt.style.use("ggplot")


# baseline model
checkpoint = 'bert-base-chinese'  # or other PLMs

if ('bert-base-chinese' in checkpoint) or ('albert-base-chinese' in checkpoint):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
else:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForMaskedLM.from_pretrained(checkpoint)

device = torch.device("cuda")
model.to(device)

data_file = 'ChineseFinancialComments_800K_seg.csv'
DATA = pd.read_csv(data_file)
# set training data sizes
split_index = int(DATA.shape[0] * 1.0)  # or 0.75, 0.50, 0.25
DATA = DATA.iloc[:split_index]
Comments = Dataset.from_pandas(DATA)
print('Comments', Comments)


def tokenize_function(examples):
    result = tokenizer(examples['Comment_seg'])
    if tokenizer.is_fast:
        result['word_ids'] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result


# Use batched=True to activate fast multithreading!
tokenized_dataset = Comments.map(
    tokenize_function, batched=True,
    remove_columns=['Datetime', 'Comment', 'ViewCount', 'URL', 'Comment_seg']
)
print('tokenized_dataset', tokenized_dataset)

print('model_max_length =', tokenizer.model_max_length)
chunk_size = (tokenizer.model_max_length // 4)
print('chunk_size =', chunk_size)


def group_texts(examples):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i: i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result


lm_dataset = tokenized_dataset.map(group_texts, batched=True)
print('lm_dataset', lm_dataset)

wwm_probability = 0.15


def whole_word_masking_data_collator(features):
    for feature in features:
        word_ids = feature.pop("word_ids")
        # Create a map between words and corresponding token indices
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)
        # Randomly mask words
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id
        feature["labels"] = new_labels
    return default_data_collator(features)


downsampled_dataset = lm_dataset.train_test_split(test_size=0.1, seed=42)
print('downsampled_dataset', downsampled_dataset)

# 定义数据加载器
train_dataloader = DataLoader(
    downsampled_dataset["train"], shuffle=True, batch_size=85,
    collate_fn=whole_word_masking_data_collator
)
validation_dataloader = DataLoader(
    downsampled_dataset["test"], batch_size=85,
    collate_fn=whole_word_masking_data_collator
)


num_epochs = 8
optimizer = AdamW(model.parameters(), lr=5e-5, no_deprecation_warning=True)
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_constant_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=int(num_training_steps * 0.1)
)
progress_bar = tqdm(range(num_training_steps))

df_loss = pd.DataFrame(columns=('epoch', 'iter', 'loss', 'perplexity'))
iter = 0
sum_loss = 0.0
interval = int(num_training_steps * 0.01)
for epoch in range(num_epochs):
    for batch in train_dataloader:
        model.train()
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        if not torch.isnan(loss):
            iter += 1
            sum_loss += loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        if iter % interval == 0:
            sum_loss /= interval
            train_ppl = 2 ** sum_loss.item()
            df_loss = pd.concat(
                [df_loss, pd.DataFrame.from_records(
                    [{'epoch': epoch + 1, 'iter': iter, 'loss': sum_loss.item(), 'perplexity': train_ppl}])],
                ignore_index=True)
            print(f'epoch: {epoch + 1}/{num_epochs}, iter: {iter}/{num_training_steps}, '
                  f'loss: {sum_loss:.4f}, perplexity: {train_ppl:.4f}')
            sum_loss = 0
        progress_bar.update(1)

    num_mask = num_correct = 0
    eval_loss = 0
    for batch in validation_dataloader:
        model.eval()
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        eval_loss += outputs.loss
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        row, col = batch["labels"].shape
        for i in range(row):
            for j in range(col):
                if batch["labels"][i][j] != -100:
                    num_mask += 1
                    if batch["labels"][i][j] == predictions[i][j]:
                        num_correct += 1
    eval_acc = num_correct / num_mask
    eval_loss = eval_loss / len(validation_dataloader)
    eval_ppl = 2 ** (eval_loss)
    print("----------------validation----------------")
    print(f"epoch: {epoch + 1}/{num_epochs}, accuracy: {eval_acc * 100:.2f}%, "
          f"loss: {eval_loss:.4f}, perplexity: {eval_ppl:.4f}")

TASK = checkpoint + '_FPT#2'
df_loss.to_csv('Loss_' + TASK + '.csv')
model.save_pretrained('./Model_' + TASK + '/')

x, y = df_loss['iter'], df_loss['loss']
title = 'Loss Curve: ' + TASK
plt.title(title)
plt.plot(x, y, "-o", markersize=6)
plt.xlabel("iter")
plt.ylabel("loss")
plt.show()