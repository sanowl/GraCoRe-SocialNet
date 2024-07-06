import os
from dotenv import load_dotenv
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from transformers import DistilBertModel, DistilBertTokenizer, PreTrainedModel, PretrainedConfig
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from datetime import datetime
from huggingface_hub import HfApi

load_dotenv()

class SocialNetConfig(PretrainedConfig):
    model_type = "social_net_predictor"

    def __init__(self, num_node_features=3, num_classes=10, **kwargs):
        super().__init__(**kwargs)
        self.num_node_features = num_node_features
        self.num_classes = num_classes

class SocialNetPredictor(PreTrainedModel):
    config_class = SocialNetConfig

    def __init__(self, config):
        super().__init__(config)
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.gat1 = GATv2Conv(config.num_node_features, 128, heads=4, dropout=0.4)
        self.gat2 = GATv2Conv(128*4, 64, heads=2, dropout=0.4)
        self.gat3 = GATv2Conv(64*2, 32, heads=1, dropout=0.4)
        self.fc1 = nn.Linear(768 + 32, 256)
        self.fc2 = nn.Linear(256, 128)
        self.output = nn.Linear(128, config.num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_ids, attention_mask, x, edge_index, batch=None):
        x = F.elu(self.gat1(x, edge_index))
        x = self.dropout(x)
        x = F.elu(self.gat2(x, edge_index))
        x = self.dropout(x)
        x = F.elu(self.gat3(x, edge_index))
        
        encoded_text = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0, :]
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = torch.cat([x, encoded_text[batch]], dim=1)
        x = F.elu(self.fc1(x))
        x = self.dropout(x)
        x = F.elu(self.fc2(x))
        x = self.dropout(x)
        out = self.output(x)
        return out

class SocialNetAnalyzer:
    def __init__(self, num_users=5000):
        self.graph = nx.barabasi_albert_graph(num_users, 10)
        self.user_data = self.generate_synthetic_data(num_users)
        self.model = None
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.graph_data = None
        self.text_data = None
        self.labels = None

    def generate_synthetic_data(self, num_users):
        interests = ['sports', 'technology', 'music', 'travel', 'food', 'fashion', 'art', 'literature', 'science', 'politics']
        locations = ['New York', 'London', 'Tokyo', 'Paris', 'Sydney', 'Berlin', 'Mumbai', 'São Paulo', 'Toronto', 'Dubai']
        
        data = {
            'user_id': [f'user{i}' for i in range(num_users)],
            'profile_description': [
                f"I'm a {np.random.choice(['passionate', 'dedicated', 'enthusiastic'])} {np.random.choice(interests)} lover from {np.random.choice(locations)}."
                for _ in range(num_users)
            ],
            'interests': [', '.join(np.random.choice(interests, size=3, replace=False)) for _ in range(num_users)],
            'location': [np.random.choice(locations) for _ in range(num_users)],
            'interest_category': [np.random.randint(0, len(interests)) for _ in range(num_users)]
        }
        return pd.DataFrame(data)

    def preprocess_data(self):
        degree_centrality = nx.degree_centrality(self.graph)
        eigenvector_centrality = nx.eigenvector_centrality(self.graph)
        pagerank = nx.pagerank(self.graph)
        node_features = torch.tensor([
            [degree_centrality[node], eigenvector_centrality[node], pagerank[node]]
            for node in self.graph.nodes()
        ], dtype=torch.float)

        edge_index = torch.tensor(list(self.graph.edges)).t().contiguous()

        self.graph_data = Data(x=node_features, edge_index=edge_index)

        self.text_data = self.user_data.apply(
            lambda row: f"Profile: {row['profile_description']} Interests: {row['interests']} Location: {row['location']}",
            axis=1
        ).tolist()
        self.labels = torch.tensor(self.user_data['interest_category'].tolist())

    def train_model(self, epochs=30, batch_size=128, learning_rate=1e-3):
        num_node_features = self.graph_data.num_node_features
        num_classes = len(self.labels.unique())

        config = SocialNetConfig(num_node_features=num_node_features, num_classes=num_classes)
        self.model = SocialNetPredictor(config)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        train_idx, val_idx = train_test_split(range(len(self.labels)), test_size=0.2, stratify=self.labels)
        train_loader = DataLoader([self.graph_data], batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                encoded_text = self.tokenizer(self.text_data, padding=True, truncation=True, return_tensors="pt")
                out = self.model(encoded_text['input_ids'], encoded_text['attention_mask'], batch.x, batch.edge_index, batch.batch)
                loss = criterion(out[train_idx], self.labels[train_idx])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            self.model.eval()
            with torch.no_grad():
                encoded_text = self.tokenizer(self.text_data, padding=True, truncation=True, return_tensors="pt")
                out = self.model(encoded_text['input_ids'], encoded_text['attention_mask'], self.graph_data.x, self.graph_data.edge_index)
                val_loss = criterion(out[val_idx], self.labels[val_idx])
                val_acc = (out[val_idx].argmax(dim=1) == self.labels[val_idx]).float().mean()
            
            scheduler.step(val_loss)
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    def predict_interests(self, user_ids):
        self.model.eval()
        with torch.no_grad():
            user_data = self.user_data.loc[self.user_data['user_id'].isin(user_ids)]
            text_data = user_data.apply(
                lambda row: f"Profile: {row['profile_description']} Interests: {row['interests']} Location: {row['location']}",
                axis=1
            ).tolist()
            encoded_text = self.tokenizer(text_data, padding=True, truncation=True, return_tensors="pt")
            out = self.model(encoded_text['input_ids'], encoded_text['attention_mask'], self.graph_data.x, self.graph_data.edge_index)
            predictions = out.argmax(dim=1)
        return predictions.tolist()

def main():
    try:
        analyzer = SocialNetAnalyzer(num_users=5000)
        analyzer.preprocess_data()
        analyzer.train_model(epochs=30, batch_size=128, learning_rate=1e-3)
        
        # Save the model locally
        local_save_directory = "./social_net_model"
        analyzer.model.save_pretrained(local_save_directory)
        analyzer.tokenizer.save_pretrained(local_save_directory)
        
        # Push to Hugging Face Hub
        hf_token = os.getenv('HF_TOKEN')
        if hf_token:
            api = HfApi()
            
            # Create a unique repository name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            repo_name = f"social-net-predictor-{timestamp}"
            repo_id = f"{api.whoami(token=hf_token)['name']}/{repo_name}"
            
            # Create the repository
            api.create_repo(repo_id, private=True, token=hf_token)
            
            # Upload the model
            api.upload_folder(
                folder_path=local_save_directory,
                repo_id=repo_id,
                repo_type="model",
                token=hf_token
            )
            print(f"Model pushed to Hugging Face Hub successfully! Repository: {repo_id}")
        else:
            print("HF_TOKEN not found in .env file. Model not pushed to Hugging Face Hub.")
        
        predictions = analyzer.predict_interests(['user1', 'user2', 'user3'])
        print("Predicted interest categories:", predictions)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
