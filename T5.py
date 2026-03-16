import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW, get_scheduler
from datasets import load_metric
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载评估指标
bleu_metric = load_metric("bleu")
rouge_metric = load_metric("rouge")

class CityPlanDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data(file_path)

    def load_data(self, file_path):
        data = []
        if file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if '|' in line:
                        src, tgt = line.strip().split('|', 1)
                        data.append({'source': src, 'target': tgt})
        elif file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        source = item['source']
        target = item['target']

        source_encoding = self.tokenizer(
            source,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        target_encoding = self.tokenizer(
            target,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': source_encoding['input_ids'].flatten(),
            'attention_mask': source_encoding['attention_mask'].flatten(),
            'labels': target_encoding['input_ids'].flatten()
        }

def load_model():
    tokenizer = T5Tokenizer.from_pretrained("mengzi-t5-base")
    model = T5ForConditionalGeneration.from_pretrained("mengzi-t5-base").to(device)
    return tokenizer, model

def train_model(model, tokenizer, train_path, epochs=10, batch_size=8, lr=1e-3):
    train_dataset = CityPlanDataset(train_path, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    scaler = GradScaler()
    loss_history = []
    best_loss = float('inf')
    early_stop_patience = 2
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                total_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            model.save_pretrained("./t5_correction_best")
            tokenizer.save_pretrained("./t5_correction_best")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print("Early stopping triggered")
                break

    plot_loss_curve(loss_history)
    return model

def plot_loss_curve(loss_history):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_history)+1), loss_history, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.close()

def inference(model, tokenizer, text, max_length=128):
    model.eval()
    inputs = tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            num_beams=5,
            top_k=50,
            top_p=0.9,
            early_stopping=True
        )
    
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

def evaluate(model, tokenizer, test_path):
    test_dataset = CityPlanDataset(test_path, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=8)
    
    model.eval()
    total_correct = 0
    total_samples = 0
    predictions = []
    references = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128,
                num_beams=5,
                top_k=50,
                top_p=0.9,
                early_stopping=True
            )

            pred_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs]
            ref_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in labels]

            predictions.extend(pred_texts)
            references.extend([[ref] for ref in ref_texts])

            for pred, ref in zip(pred_texts, ref_texts):
                if pred == ref:
                    total_correct += 1
                total_samples += 1

    exact_match = total_correct / total_samples
    bleu = bleu_metric.compute(predictions=predictions, references=references)['bleu']
    rouge = rouge_metric.compute(predictions=predictions, references=references)['rougeL'].mid.fmeasure

    print(f"Exact Match Accuracy: {exact_match:.4f}")
    print(f"BLEU Score: {bleu:.4f}")
    print(f"ROUGE-L Score: {rouge:.4f}")

    return exact_match, bleu, rouge

def demo_correction(model, tokenizer):
    test_cases = [
        "这个城市规化一点都不好，就这？",
        "道路设汁不合理，经常堵车！！",
        "公园建的太偏了，木有人去",
        "归划局的决策有问题，完全不考虑民生"
    ]
    
    print("\n=== 纠错效果展示 ===")
    for case in test_cases:
        corrected = inference(model, tokenizer, case)
        print(f"原始文本：{case}")
        print(f"纠错后：{corrected}\n")

if __name__ == "__main__":
    # 数据文件路径（AutoDL中直接替换为你的数据路径）
    train_data_path = "train_data.txt"
    test_data_path = "test_data.txt"

    # 加载模型和训练
    tokenizer, model = load_model()
    model = train_model(model, tokenizer, train_data_path)

    # 加载最优模型评估
    best_model = T5ForConditionalGeneration.from_pretrained("./t5_correction_best").to(device)
    best_tokenizer = T5Tokenizer.from_pretrained("./t5_correction_best")

    # 评估和演示
    evaluate(best_model, best_tokenizer, test_data_path)
    demo_correction(best_model, best_tokenizer)