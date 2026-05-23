import torch
import torch.utils.data
from transformers import BertTokenizer
import pandas as pd
import os
from sklearn.model_selection import train_test_split


data_dir = './'
total_csv_name = 'data.csv'

num_workers = 0
batch_size = 8
max_seq_len = 128
pretrained_model = 'model'
random_seed = 42

data_transform = {
    'train': {
        'max_length': max_seq_len,
        'padding': 'max_length',
        'truncation': True,
        'return_tensors': 'pt',
        'return_attention_mask': True,
        'return_token_type_ids': True
    },
    'val': {
        'max_length': max_seq_len,
        'padding': 'max_length',
        'truncation': True,
        'return_tensors': 'pt',
        'return_attention_mask': True,
        'return_token_type_ids': True
    },
    'test': {
        'max_length': max_seq_len,
        'padding': 'max_length',
        'truncation': True,
        'return_tensors': 'pt',
        'return_attention_mask': True,
        'return_token_type_ids': True
    }
}

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, transform_params):
        self.df = df
        required_columns = ['title', 'label']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"CSV缺少列：{missing_columns}")

        self.df = self.df[required_columns].copy()
        self.df = self.df.dropna(subset=['title', 'label']).reset_index(drop=True)
        self.df['label'] = self.df['label'].astype(int)

        self.tokenizer = tokenizer
        self.transform_params = transform_params

    def __getitem__(self, idx):
        text = str(self.df.loc[idx, 'title'])
        label = self.df.loc[idx, 'label']

        encoded = self.tokenizer(text, **self.transform_params)
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        token_type_ids = encoded['token_type_ids'].squeeze(0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
        }, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.df)

def load_and_split_in_memory():
    total_path = os.path.join(data_dir, total_csv_name)
    df = pd.read_csv(total_path)

    train_df, temp_df = train_test_split(
        df,
        test_size=0.4,
        random_state=random_seed,
        stratify=df['label']
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=random_seed,
        stratify=temp_df['label']
    )

    print(f"内存划分完成（train:val:test = 6:2:2）")
    print(f"训练集：{len(train_df)}")
    print(f"验证集：{len(val_df)}")
    print(f"测试集：{len(test_df)}")

    return train_df, val_df, test_df

train_df, val_df, test_df = load_and_split_in_memory()
tokenizer = BertTokenizer.from_pretrained(pretrained_model)

train_dataset = SentimentDataset(train_df, tokenizer, data_transform['train'])
val_dataset = SentimentDataset(val_df, tokenizer, data_transform['val'])
test_dataset = SentimentDataset(test_df, tokenizer, data_transform['test'])

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=num_workers, pin_memory=True
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=True
)