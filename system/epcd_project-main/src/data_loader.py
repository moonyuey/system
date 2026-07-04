import os
import pickle
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
from torch_geometric.data import Data
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import warnings
warnings.filterwarnings('ignore')

class SCCDDataLoader:
    def __init__(self, config):

        self.raw_path = config['dataset']['raw_path']
        self.processed_path = config['dataset']['processed_path']
        self.num_windows = config['graph']['num_time_windows']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.sentiment_model_name = 'uer/roberta-base-finetuned-dianping-chinese'  # 二分类（positive/negative）
        self.tokenizer = AutoTokenizer.from_pretrained(self.sentiment_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.sentiment_model_name).to(self.device)
        self.model.eval()
        # 定义情感标签映射
        self.label_map = {0: 'negative', 1: 'positive'}  # 该模型输出 0=neg, 1=pos



    def get_sentiment(self, text):
        """对单条文本返回三维情感概率 [positive, negative, neutral] (归一化)"""
        if not text or not isinstance(text, str):
            return [0.0, 0.0, 1.0]  # 中性
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]  # [neg_prob, pos_prob]
        pos_prob = probs[1]
        neg_prob = probs[0]
        # 设定中性：若两概率均小于0.4，则视为中性（保守）
        if pos_prob < 0.4 and neg_prob < 0.4:
            neu_prob = 1.0 - pos_prob - neg_prob
            return [pos_prob, neg_prob, neu_prob]  # 实际 pos+neg 可能小于1，剩余为中性
        else:

            total = pos_prob + neg_prob
            if total == 0:
                return [0.0, 0.0, 1.0]
            return [pos_prob/total, neg_prob/total, 0.0]

    def load_raw_data(self):

        posts_path = os.path.join(self.raw_path, 'posts.csv')
        comments_path = os.path.join(self.raw_path, 'comments.csv')
        posts = pd.read_csv(posts_path, encoding='utf-8')
        comments = pd.read_csv(comments_path, encoding='utf-8')
        if 'created_at' in posts.columns:
            posts.rename(columns={'created_at': 'timestamp'}, inplace=True)
        if 'created_at' in comments.columns:
            comments.rename(columns={'created_at': 'timestamp'}, inplace=True)
        if posts['timestamp'].dtype == object:
            posts['timestamp'] = pd.to_datetime(posts['timestamp']).astype('int64') // 10**9
        if comments['timestamp'].dtype == object:
            comments['timestamp'] = pd.to_datetime(comments['timestamp']).astype('int64') // 10**9
        return posts, comments

    def build_sessions(self, posts, comments):
        sessions = {}
        # 首先添加帖子作为第一条消息
        for _, row in posts.iterrows():
            pid = row['post_id']
            sessions[pid] = {
                'messages': [{
                    'user_id': row['user_id'],
                    'text': row['content'],
                    'timestamp': row['timestamp'],
                    'is_post': True,
                    'comment_id': None
                }],
                'label': 0  # 默认非暴力，后续从评论中判断
            }
        # 添加评论
        for _, row in comments.iterrows():
            pid = row['post_id']
            if pid not in sessions:
                continue
            sessions[pid]['messages'].append({
                'user_id': row['user_id'],
                'text': row['content'],
                'timestamp': row['timestamp'],
                'is_post': False,
                'comment_id': row.get('comment_id', None),
                'reply_to_comment_id': row.get('reply_to_comment_id', None),
                'reply_to_user_id': row.get('reply_to_user_id', None)  # 可能直接有被回复用户
            })
            if 'is_bullying' in row:
                sessions[pid]['label'] = max(sessions[pid]['label'], row['is_bullying'])
        for pid, sess in sessions.items():
            sess['messages'].sort(key=lambda x: x['timestamp'])
        session_list = list(sessions.values())
        return session_list

    def extract_sentiment_features(self, sessions):
        """对每个会话中所有消息的文本提取情感概率，并添加至消息字典"""
        for sess in tqdm(sessions, desc="Extracting sentiment"):
            for msg in sess['messages']:
                if msg['text'] and isinstance(msg['text'], str):
                    pos, neg, neu = self.get_sentiment(msg['text'])
                    msg['sentiment'] = [pos, neg, neu]
                else:
                    msg['sentiment'] = [0.0, 0.0, 1.0]
        return sessions

    def build_graph_sequence(self, sessions):
        all_graph_sequences = []
        all_labels = []
        for sess in tqdm(sessions, desc="Building graphs"):
            messages = sess['messages']
            if len(messages) < 2:  # 至少有一条帖子+一条评论才有交互
                continue
            label = sess['label']
            # 计算每个窗口的时间边界
            start_time = messages[0]['timestamp']
            end_time = messages[-1]['timestamp']
            duration = end_time - start_time
            if duration == 0:
                window_sec = 1
            else:
                window_sec = duration / self.num_windows
            # 按窗口分组
            window_messages = [[] for _ in range(self.num_windows)]
            for msg in messages:
                idx = min(int((msg['timestamp'] - start_time) / window_sec), self.num_windows - 1)
                window_messages[idx].append(msg)
            graph_seq = []
            for w_idx, msg_list in enumerate(window_messages):
                if not msg_list:
                    graph_seq.append(None)
                    continue
                # 提取该窗口内的节点（用户）
                users = list(set([m['user_id'] for m in msg_list]))
                user2idx = {u: i for i, u in enumerate(users)}
                # 统计交互边：回复关系
                edge_dict = defaultdict(int)
                for msg in msg_list:
                    src = msg['user_id']
                    dst = None
                    if 'reply_to_user_id' in msg and msg['reply_to_user_id'] is not None:
                        dst = msg['reply_to_user_id']
                    elif 'reply_to_comment_id' in msg and msg['reply_to_comment_id'] is not None:
                        for m in messages:
                            if m.get('comment_id') == msg['reply_to_comment_id']:
                                dst = m['user_id']
                                break
                    if dst is not None and dst in user2idx and src in user2idx:
                        edge_dict[(src, dst)] += 1
                # 如果没有任何交互，跳过该窗口
                if not edge_dict:
                    graph_seq.append(None)
                    continue
                # 构建边和权重
                edge_list = []
                edge_weight = []
                for (src, dst), w in edge_dict.items():
                    edge_list.append([user2idx[src], user2idx[dst]])
                    edge_weight.append(w)
                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
                edge_weight = torch.tensor(edge_weight, dtype=torch.float)
                num_nodes = len(users)
                out_deg = torch.zeros(num_nodes)
                in_deg = torch.zeros(num_nodes)
                for (src, dst), w in edge_dict.items():
                    out_deg[user2idx[src]] += w
                    in_deg[user2idx[dst]] += w
                norm_out = out_deg / (out_deg.sum() + 1e-8)
                user_sent = {u: [] for u in users}
                for msg in msg_list:
                    u = msg['user_id']
                    if u in user_sent and 'sentiment' in msg:
                        user_sent[u].append(msg['sentiment'])
                sent_feat = torch.zeros((num_nodes, 3))
                for u, sent_list in user_sent.items():
                    if sent_list:
                        avg_sent = np.mean(sent_list, axis=0)
                        sent_feat[user2idx[u]] = torch.tensor(avg_sent, dtype=torch.float)
                    else:
                        sent_feat[user2idx[u]] = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float)  # 中性
                # 拼接结构特征（3维）和情感特征（3维）
                x = torch.cat([out_deg.unsqueeze(1), in_deg.unsqueeze(1), norm_out.unsqueeze(1), sent_feat], dim=1)
                # 创建Data对象
                data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
                graph_seq.append(data)
            # 如果所有窗口都为空，跳过该会话
            if all(g is None for g in graph_seq):
                continue
            all_graph_sequences.append(graph_seq)
            all_labels.append(label)
        return all_graph_sequences, all_labels

    def preprocess(self):
        posts, comments = self.load_raw_data()
        sessions = self.build_sessions(posts, comments)
        sessions = self.extract_sentiment_features(sessions)
        graph_sequences, labels = self.build_graph_sequence(sessions)
        # 保存到 processed_path
        os.makedirs(self.processed_path, exist_ok=True)
        save_path = os.path.join(self.processed_path, 'graph_sequences.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump((graph_sequences, labels), f)
        print(f"Preprocessing done. Saved {len(graph_sequences)} sessions.")
        return graph_sequences, labels

    def load_processed(self):

        save_path = os.path.join(self.processed_path, 'graph_sequences.pkl')
        with open(save_path, 'rb') as f:
            graph_sequences, labels = pickle.load(f)
        return graph_sequences, labels