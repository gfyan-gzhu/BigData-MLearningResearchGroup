import torch
from transformers import BertTokenizerFast, AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import AdamW, get_cosine_schedule_with_warmup
from transformers import DataCollatorWithPadding
from datasets import load_dataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from evaluate import load
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  #


def fuseModels(w_Int, C_Set):
    MainStructure = set(C_Set[0].state_dict().keys())
    for C in C_Set:
        MainStructure = MainStructure & C.state_dict().keys()
    order_list = [x for x in C_Set[0].state_dict().keys()]
    MainStructure = sorted(MainStructure, key=lambda x: order_list.index(x))

    fusionModel = AutoModelForSequenceClassification.from_pretrained('model', num_labels=2)
    fusionParameters = fusionModel.state_dict()
    for part in MainStructure:
        fusionParameters[part] = torch.zeros_like(fusionParameters[part])
        for i, C in enumerate(C_Set):
            fusionParameters[part] += w_Int[i] * C.state_dict()[part]
    torch.save(fusionParameters, 'fusionModel.pt')


def getScore(w_Int, C_Set, scoresDB):
    if w_Int in scoresDB.keys():
        return scoresDB[w_Int]

    fuseModels(w_Int, C_Set)
    fusionModel = AutoModelForSequenceClassification.from_pretrained('model', num_labels=2)
    fusionModel.load_state_dict(torch.load('fusionModel.pt'))
    device = torch.device('cuda')
    fusionModel.to(device)

    tokenizer = AutoTokenizer.from_pretrained('model')

    data_file = {'train': 'fine-tuning.csv'}
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

    all_predictions = []
    all_labels = []

    for batch in test_dataloader:
        fusionModel.eval()
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = fusionModel(**batch)
            logits = outputs.logits
            model_predictions = torch.argmax(logits, dim=-1)
            all_predictions.extend(model_predictions.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
    score = accuracy_score(all_labels, all_predictions)

    print(f'<{w_Int[0]}, {w_Int[1]}, {w_Int[2]}, {w_Int[3]}, {w_Int[4]}> has a score of {score * 100:.4f}%')

    scoresDB[w_Int] = score
    with open('getScore_log_01_05__02.txt', 'a') as f:
        print(f'{w_Int[0]},{w_Int[1]},{w_Int[2]},{w_Int[3]},{w_Int[4]},{score:.8f}', file=f)
    return score
