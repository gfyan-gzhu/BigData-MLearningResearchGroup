import torch
from transformers import BertTokenizerFast, AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import AdamW, get_cosine_schedule_with_warmup
from transformers import DataCollatorWithPadding
from datasets import load_dataset
from torch.utils.data import DataLoader
from evaluate import load
from transformers import logging
logging.set_verbosity_error()


def getScore(checkpoint, seed):
    torch.manual_seed(seed)

    testModel = AutoModelForSequenceClassification. \
        from_pretrained(checkpoint, num_labels=2, ignore_mismatched_sizes=True)
    device = torch.device('cuda')
    testModel.to(device)

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    # tokenizer = AutoTokenizer.from_pretrained('')

    data_file = {'train': 'Fine-tuning dataset.csv'}
    FTdata = load_dataset('csv', data_files=data_file)

    def tokenize_function(example):
        return tokenizer(example['content'], truncation=True)

    # Use batched=True to activate fast multithreading!
    tokenized_dataset = FTdata.map(tokenize_function, batched=True, remove_columns=['content'])
    tokenized_dataset = tokenized_dataset.rename_column(original_column_name='label', new_column_name='labels')
    tokenized_dataset.set_format('torch')

    downsampled_dataset = tokenized_dataset['train'].train_test_split(test_size=0.4, seed=42)
    training_set = downsampled_dataset['train']
    downsampled_dataset = downsampled_dataset['test'].train_test_split(test_size=0.5, seed=42)
    testing_set = downsampled_dataset['test']

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataloader = DataLoader(training_set, batch_size=32, collate_fn=data_collator)
    test_dataloader = DataLoader(testing_set, batch_size=32, collate_fn=data_collator)

    num_epochs = 2
    optimizer = AdamW(testModel.parameters(), lr=2e-5, no_deprecation_warning=True)
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(num_training_steps * 0.1),
        num_training_steps=num_training_steps
    )

    for epoch in range(num_epochs):
        for batch in train_dataloader:
            testModel.train()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = testModel(**batch)
            loss = outputs.loss
            if not torch.isnan(loss):
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

    metric = load('accuracy')
    for batch in test_dataloader:
        testModel.eval()
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = testModel(**batch)
            logits = outputs.logits
            model_predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=model_predictions, references=batch["labels"])
    score = metric.compute()['accuracy']
    print(f'{checkpoint} with seed({seed}) has a score of {score * 100:.4f}%')
    return score
