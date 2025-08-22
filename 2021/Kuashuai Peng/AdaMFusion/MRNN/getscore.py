import torch
from transformers import BertTokenizerFast, AutoTokenizer
from transformers import AutoModelForMaskedLM, AutoModelForSequenceClassification
from transformers import AdamW, get_cosine_schedule_with_warmup
from transformers import DataCollatorWithPadding
from datasets import load_dataset
from torch.utils.data import DataLoader
from evaluate import load
from transformers import logging
logging.set_verbosity_error()
torch.manual_seed(3407)


def getScore(weights):
    C_0 = AutoModelForMaskedLM.from_pretrained('parentModel#0')
    C_1 = AutoModelForMaskedLM.from_pretrained('parentModel#1')
    C_2 = AutoModelForMaskedLM.from_pretrained('parentModel#2')
    C_3 = AutoModelForSequenceClassification.from_pretrained('parentModel#3', num_labels=10)
    C_4 = AutoModelForSequenceClassification.from_pretrained('parentModel#4', num_labels=5)
    C_Set = (C_0, C_1, C_2, C_3, C_4)

    MainStructure = set(C_Set[0].state_dict().keys())
    for C in C_Set:
        MainStructure = MainStructure & C.state_dict().keys()
    order_list = [x for x in C_Set[0].state_dict().keys()]
    MainStructure = sorted(MainStructure, key=lambda x: order_list.index(x))

    fusionModel = AutoModelForSequenceClassification.from_pretrained('parentModel#0', num_labels=2)
    fusionParameters = fusionModel.state_dict()
    for part in MainStructure:
        fusionParameters[part] = torch.zeros_like(fusionParameters[part])
        for i, C in enumerate(C_Set):
            fusionParameters[part] += weights[i] * C.state_dict()[part]

    fusionModel.load_state_dict(fusionParameters)
    device = torch.device('cuda')
    fusionModel.to(device)

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
    optimizer = AdamW(fusionModel.parameters(), lr=2e-5, no_deprecation_warning=True)
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(num_training_steps * 0.1),
        num_training_steps=num_training_steps
    )

    for epoch in range(num_epochs):
        for batch in train_dataloader:
            fusionModel.train()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = fusionModel(**batch)
            loss = outputs.loss
            if not torch.isnan(loss):
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

    accuracy = load('accuracy')
    precision = load('precision')
    recall = load('recall')
    f1_score = load('f1')
    for batch in test_dataloader:
        fusionModel.eval()
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = fusionModel(**batch)
            logits = outputs.logits
            model_predictions = torch.argmax(logits, dim=-1)
            accuracy.add_batch(predictions=model_predictions, references=batch["labels"])
            precision.add_batch(predictions=model_predictions, references=batch["labels"])
            recall.add_batch(predictions=model_predictions, references=batch["labels"])
            f1_score.add_batch(predictions=model_predictions, references=batch["labels"])
    accuracy = round(accuracy.compute()['accuracy'], 5)
    precision = round(precision.compute()['precision'], 5)
    recall = round(recall.compute()['recall'], 5)
    f1_score = round(f1_score.compute()['f1'], 5)
    score = (accuracy, precision, recall, f1_score)
    return score
