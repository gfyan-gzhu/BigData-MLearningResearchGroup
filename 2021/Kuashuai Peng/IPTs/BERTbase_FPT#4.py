from transformers import BertTokenizerFast, AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import AdamW, get_cosine_schedule_with_warmup
from datasets import Dataset
from torch.utils.data import DataLoader
import torch
import pandas as pd
from evaluate import load
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
plt.style.use("ggplot")


checkpoint = 'bert-base-chinese'  # or other PLMs

if ('bert-base-chinese' in checkpoint) or ('albert-base-chinese' in checkpoint):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
else:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification. \
    from_pretrained(checkpoint, num_labels=5, ignore_mismatched_sizes=True)

device = torch.device("cuda")
model.to(device)

data_file = 'DBReviews_100W.csv'
DATA = pd.read_csv(data_file, index_col=0)
DATA = DATA.sample(500_000)
# set training data sizes
split_index = int(DATA.shape[0] * 0.75)  # or 1, 0.50, 0.25
DATA = DATA.iloc[:split_index]
Comments = Dataset.from_pandas(DATA)
print('Comments', Comments)


def tokenize_function(example):
    return tokenizer(example['comment'], truncation=True)


# Use batched=True to activate fast multithreading!
tokenized_dataset = Comments.map(tokenize_function, batched=True, remove_columns=['comment'])
tokenized_dataset = tokenized_dataset.rename_column(original_column_name='label', new_column_name='labels')
tokenized_dataset.set_format("torch")
print('tokenized_dataset', tokenized_dataset)

downsampled_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
training_set = downsampled_dataset['train']
validation_set = downsampled_dataset['test']
print('training set', training_set)
print('validation set', validation_set)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
train_dataloader = DataLoader(training_set, shuffle=True, batch_size=100, collate_fn=data_collator)
validation_dataloader = DataLoader(validation_set, batch_size=100, collate_fn=data_collator)

num_epochs = 3
optimizer = AdamW(model.parameters(), lr=3e-5, no_deprecation_warning=True)
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=int(num_training_steps * 0.1),
    num_training_steps=num_training_steps
)
progress_bar = tqdm(range(num_training_steps))

metric1, metric2, metric3, metric4 = load('accuracy'), load('precision'), load('recall'), load('f1')
df_loss = pd.DataFrame(columns=('epoch', 'iter', 'loss'))
iter = 0
sum_loss = 0.0
interval = int(num_training_steps * 0.1)
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
            df_loss = pd.concat(
                [df_loss, pd.DataFrame.from_records([{'epoch': epoch + 1, 'iter': iter, 'loss': sum_loss.item()}])],
                ignore_index=True)
            print(f'epoch: {epoch + 1}/{num_epochs}, iter: {iter}/{num_training_steps}, loss: {sum_loss:.4f}')
            sum_loss = 0
        progress_bar.update(1)

    for batch in validation_dataloader:
        model.eval()
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        model_predictions = torch.argmax(logits, dim=-1)
        metric1.add_batch(predictions=model_predictions, references=batch["labels"])
        metric2.add_batch(predictions=model_predictions, references=batch["labels"])
        metric3.add_batch(predictions=model_predictions, references=batch["labels"])
        metric4.add_batch(predictions=model_predictions, references=batch["labels"])
    print("----------------validation----------------")
    score = (metric1.compute(), metric2.compute(average='macro'), metric3.compute(average='macro'), metric4.compute(average='macro'))
    print(f"accuracy: {score[0]['accuracy'] * 100:.2f}%, precision: {score[1]['precision'] * 100:.2f}%, "
          f"recall: {score[2]['recall'] * 100:.2f}%, f1 score: {score[3]['f1'] * 100:.2f}%")

TASK = checkpoint + '_FPT#4'
df_loss.to_csv('Loss_' + TASK + '.csv')
model.save_pretrained('./Model_' + TASK + '/')

x, y = df_loss['iter'], df_loss['loss']
title = 'Loss Curve: ' + TASK
plt.title(title)
plt.plot(x, y, "-o", markersize=6)
plt.xlabel("iter")
plt.ylabel("loss")
plt.show()
