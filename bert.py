import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_scheduler
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 标签映射
label2id = {'O': 0, 'B-MET': 1, 'I-MET': 2}
id2label = {0: 'O', 1: 'B-MET', 2: 'I-MET'}

class MetaphorDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data(file_path)

    def load_data(self, file_path):
        """加载序列标注数据"""
        data = []
        if file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.read().split('\n\n')
                for line in lines:
                    if line.strip():
                        tokens = []
                        labels = []
                        for item in line.strip().split('\n'):
                            if item.strip():
                                token, label = item.strip().split()
                                tokens.append(token)
                                labels.append(label2id[label])
                        data.append({'tokens': tokens, 'labels': labels})
        elif file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    item['labels'] = [label2id[label] for label in item['labels']]
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """数据预处理：处理tokenize后的标签对齐问题"""
        item = self.data[idx]
        tokens = item['tokens']
        labels = item['labels']
        
        # 使用BERT tokenizer处理
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 处理标签对齐（处理subword情况）
        word_ids = encoding.word_ids()
        aligned_labels = [-100] * len(word_ids)  # -100表示忽略的标签
        
        for i, word_id in enumerate(word_ids):
            if word_id is not None:
                aligned_labels[i] = labels[word_id]
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(aligned_labels)
        }

class BertForMetaphorDetection(nn.Module):
    def __init__(self, num_labels=3):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        # 获取BERT输出
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # 通过分类层得到每个token的标签概率
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)  # [batch_size, seq_len, num_labels]
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))
        
        return {'loss': loss, 'logits': logits, 'hidden_states': sequence_output}

def load_model():
    """加载预训练模型和tokenizer"""
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    model = BertForMetaphorDetection(num_labels=len(label2id)).to(device)
    return tokenizer, model

def train_model(model, tokenizer, train_path, epochs=10, batch_size=8, lr=2e-5):
    """模型训练函数"""
    train_dataset = MetaphorDataset(train_path, tokenizer)
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
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs['loss']
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
        
        # 保存最优模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            model.save_pretrained("./bert_metaphor_best")
            tokenizer.save_pretrained("./bert_metaphor_best")
    
    # 绘制损失曲线
    plot_loss_curve(loss_history)
    return model

def plot_loss_curve(loss_history):
    """绘制训练损失曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_history)+1), loss_history, marker='o', color='#2E86AB')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('BERT Metaphor Detection Training Loss Curve', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig('bert_training_loss.png', dpi=300, bbox_inches='tight')
    plt.close()

def inference(model, tokenizer, text, threshold=0.5):
    """隐喻识别推理函数"""
    model.eval()
    
    # 分词处理
    tokens = tokenizer.tokenize(text)
    encoding = tokenizer(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs['logits']
        hidden_states = outputs['hidden_states']
        
        # 获取每个token的预测概率
        probs = torch.softmax(logits, dim=-1)
        pred_probs, pred_ids = torch.max(probs, dim=-1)
        
        # 处理预测结果
        word_ids = encoding.word_ids()[0]
        results = []
        metaphor_words = []
        semantic_fingerprints = []
        
        current_word = None
        current_metaphor = False
        current_prob = 0.0
        
        for i, (word_id, pred_id, prob) in enumerate(zip(word_ids, pred_ids[0], pred_probs[0])):
            if word_id is not None and word_id != current_word:
                # 新单词开始
                if current_word is not None and current_metaphor:
                    # 保存上一个单词的结果
                    results.append({
                        'word': tokenizer.decode([encoding['input_ids'][0][current_word_start]]),
                        'is_metaphor': True,
                        'confidence': current_prob,
                        'position': (current_word_start, i-1)
                    })
                    metaphor_words.append(tokenizer.decode([encoding['input_ids'][0][current_word_start]]))
                    # 获取语义指纹（取第一个subword的隐藏状态）
                    semantic_fingerprints.append(hidden_states[0][current_word_start].cpu().numpy())
                
                current_word = word_id
                current_word_start = i
                current_metaphor = (id2label[pred_id.item()] in ['B-MET', 'I-MET']) and (prob.item() > threshold)
                current_prob = prob.item() if current_metaphor else 0.0
        
        # 处理最后一个单词
        if current_word is not None and current_metaphor:
            results.append({
                'word': tokenizer.decode([encoding['input_ids'][0][current_word_start]]),
                'is_metaphor': True,
                'confidence': current_prob,
                'position': (current_word_start, len(word_ids)-1)
            })
            metaphor_words.append(tokenizer.decode([encoding['input_ids'][0][current_word_start]]))
            semantic_fingerprints.append(hidden_states[0][current_word_start].cpu().numpy())
    
    return {
        'original_text': text,
        'metaphor_results': results,
        'metaphor_words': metaphor_words,
        'semantic_fingerprints': semantic_fingerprints
    }

def visualize_metaphor(text, metaphor_results):
    """可视化隐喻识别结果"""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.text(0.05, 0.5, text, fontsize=14, verticalalignment='center', fontfamily='SimHei')
    
    # 标记隐喻词
    for result in metaphor_results:
        if result['is_metaphor']:
            # 简单估算字符位置（实际应用中需要更精确的计算）
            start_char = result['position'][0]
            end_char = result['position'][1]
            rect = Rectangle((start_char*0.05, 0.1), (end_char-start_char)*0.05, 0.8, 
                           linewidth=2, edgecolor='red', facecolor='red', alpha=0.3)
            ax.add_patch(rect)
    
    ax.set_xlim(0, len(text)*0.05 + 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.title('Metaphor Detection Result Visualization', fontsize=16, fontfamily='SimHei')
    plt.tight_layout()
    plt.savefig('metaphor_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

def demo_metaphor_detection(model, tokenizer):
    """隐喻识别演示"""
    test_cases = [
        "这个项目就是个面子工程，完全不考虑实际需求",
        "最后一公里的问题一直没解决，居民出行很不方便",
        "政府部门之间互相踢皮球，导致问题迟迟得不到解决",
        "这个规划就是个大白象工程，浪费了大量资源"
    ]
    
    print("\n=== 隐喻识别效果展示 ===")
    for i, case in enumerate(test_cases):
        result = inference(model, tokenizer, case)
        print(f"\n示例 {i+1}:")
        print(f"原始文本：{case}")
        print(f"识别到的隐喻词：{result['metaphor_words']}")
        for item in result['metaphor_results']:
            print(f"  - {item['word']}: 置信度 {item['confidence']:.4f}")
        
        # 可视化结果
        visualize_metaphor(case, result['metaphor_results'])

if __name__ == "__main__":
    # 数据文件路径（AutoDL中直接替换为你的数据路径）
    train_data_path = "metaphor_train.txt"
    
    # 加载模型和训练
    tokenizer, model = load_model()
    model = train_model(model, tokenizer, train_data_path)
    
    # 加载最优模型进行演示
    best_model = BertForMetaphorDetection.from_pretrained("./bert_metaphor_best").to(device)
    best_tokenizer = BertTokenizer.from_pretrained("./bert_metaphor_best")
    
    # 演示隐喻识别效果
    demo_metaphor_detection(best_model, best_tokenizer)
