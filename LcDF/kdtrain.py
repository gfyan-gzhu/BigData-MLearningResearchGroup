import torch
import os
from dataload import train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, max_seq_len
from mumodel import model_base_12, model_small_6
from transformers import BertModel
from torch import nn, optim
from tqdm import tqdm
from torch.nn import functional as F
from tomethod import check, calibrate_target_weight, lr_cosine
from printlog import save_log
import logging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

GLOBAL_MAX_SEQ_LEN = max_seq_len
NUM_CLASSES = 10
VOCAB_SIZE = 21128

logger = logging.getLogger()
logger.setLevel(logging.INFO)
tips = 'model6_sentiment_fixed'
save_log(tips)

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备：{device}")

    teacher_model = model_base_12(
        num_classes=NUM_CLASSES,
        max_seq_len=GLOBAL_MAX_SEQ_LEN,
        vocab_size=VOCAB_SIZE  # 传入正确词表
    )

    studentNet = model_small_6(
        num_classes=NUM_CLASSES,
        max_seq_len=GLOBAL_MAX_SEQ_LEN,
        vocab_size=VOCAB_SIZE
    )

    print("正在加载本地预训练模型权重...")
    hf_model = BertModel.from_pretrained("./model")
    hf_state_dict = hf_model.state_dict()
    teacher_state_dict = teacher_model.state_dict()

    for k, v in hf_state_dict.items():
        if k in teacher_state_dict and teacher_state_dict[k].shape == v.shape:
            teacher_state_dict[k] = v

    teacher_model.load_state_dict(teacher_state_dict, strict=False)
    teacherNet = teacher_model.to(device).eval()
    studentNet = studentNet.to(device)

    optimizer = optim.AdamW(studentNet.parameters(), lr=2e-5, weight_decay=1e-4)
    lossCE = nn.CrossEntropyLoss()
    lossKD = nn.KLDivLoss(reduction='batchmean')

    val_num = len(val_dataset)
    best_acc = 0.0
    save_path = './pth/model6_best.pth'
    train_steps = len(train_loader)
    epochs = 100
    T = 2.0
    lambda_stu = 0.5

    for epoch in range(epochs):
        studentNet.train()
        running_loss = 0.0
        train_loader = tqdm(train_loader)

        for step, data in enumerate(train_loader):
            inputs, labels = data
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            inputs['input_ids'] = torch.clamp(inputs['input_ids'], 0, VOCAB_SIZE - 1)

            optimizer.zero_grad()

            logits_stu = studentNet(**inputs)
            loss_student = lossCE(logits_stu, labels)

            with torch.no_grad():
                logits_tea = teacherNet(**inputs)

            mask = check(logits_tea, labels)
            tea_easy = logits_tea[mask]
            stu_easy = logits_stu[mask]
            if len(tea_easy) > 0:
                loss_easy = lossKD(
                    F.log_softmax(stu_easy / T, 1),
                    F.softmax(tea_easy / T, 1)
                ) * T ** 2
            else:
                loss_easy = torch.tensor(0., device=device)

            tea_hard, stu_hard = calibrate_target_weight(mask, logits_tea, logits_stu, labels)
            if len(tea_hard) > 0:
                loss_hard = lossKD(
                    F.log_softmax(stu_hard / T, 1),
                    F.softmax(tea_hard / T, 1)
                ) * T ** 2
            else:
                loss_hard = torch.tensor(0., device=device)

            gamma = lr_cosine(epoch, epochs)
            loss = lambda_stu * loss_student + (1 - lambda_stu) * ((1 - gamma) * loss_easy + gamma * loss_hard)

            loss.backward()
            nn.utils.clip_grad_norm_(studentNet.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            train_loader.desc = f"epoch {epoch + 1} loss {loss.item():.4f}"

        studentNet.eval()
        acc = 0.0
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = val_data
                val_inputs = {k: v.to(device) for k, v in val_inputs.items()}
                val_labels = val_labels.to(device)

                val_inputs['input_ids'] = torch.clamp(val_inputs['input_ids'], 0, VOCAB_SIZE - 1)

                outputs = studentNet(**val_inputs)
                predict_y = torch.max(outputs, 1)[1]
                acc += torch.eq(predict_y, val_labels).sum().item()

        val_acc = acc / val_num
        print(f"epoch {epoch + 1} val_acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(studentNet.state_dict(), save_path)

    print("\n" + "=" * 60)
    print("训练完成，开始加载最优模型在测试集上评估...")

    studentNet.load_state_dict(torch.load(save_path, map_location=device))
    studentNet.eval()

    test_num = len(test_dataset)
    test_correct = 0.0

    with torch.no_grad():
        for test_data in tqdm(test_loader, desc="测试集评估"):
            test_inputs, test_labels = test_data
            test_inputs = {k: v.to(device) for k, v in test_inputs.items()}
            test_labels = test_labels.to(device)

            test_inputs['input_ids'] = torch.clamp(test_inputs['input_ids'], 0, VOCAB_SIZE - 1)

            outputs = studentNet(**test_inputs)
            predict_y = torch.max(outputs, 1)[1]
            test_correct += torch.eq(predict_y, test_labels).sum().item()

    test_acc = test_correct / test_num
    print("=" * 60)
    print(f"最终结果")
    print(f"最佳验证集准确率：{best_acc:.4f}")
    print(f"测试集准确率：{test_acc:.4f}")
    print("=" * 60)