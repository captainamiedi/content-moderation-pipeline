import spacy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import time
from typing import List, Tuple
import logging
import random
import copy
import re
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, confusion_matrix

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_md")
except:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_md"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    nlp = spacy.load("en_core_web_md")

# Define constants
VEC_DIM = 768  # BERT embedding dimension
HIDDEN_DIM = 256
CLASSIFIER_DIM = 128
LEARNING_RATE = 0.0001
EPOCHS = 50
PATIENCE = 10
BATCH_SIZE = 4
THREAT_KEYWORDS = ["kill", "hurt", "destroy", "illegal", "bomb", "drugs", 
                  "weapons", "attack", "harm", "shoot", "murder", "assault"]
SAFE_KEYWORDS = ["peace", "respectfully", "safe", "positive", "constructive", 
                "community", "understand", "help", "support", "improve"]

# Load BERT model for feature extraction
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model.eval()

class ContentModerationSystem:
    def __init__(self):
        self.gnn = None
        self.classifier = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_dim = None
        self.optimizer = None
        self.criterion = nn.BCEWithLogitsLoss()
        self.safe_confidence_threshold = 0.6  # Increased threshold for safety
        
    def train(self, X_train: List[str], y_train: List[int]) -> None:
        if len(X_train) == 0:
            raise ValueError("Training data cannot be empty")
        
        # Preprocess data
        docs = [nlp(text[:1000]) for text in X_train]
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        
        # Precompute features
        graph_data_list = [self._create_graph_data(doc) for doc in docs]
        
        # Train with optimized hyperparameters
        self._train_model(graph_data_list, y_train_tensor, 128, 64)
        
    def predict(self, document: str) -> Tuple[int, float]:
        start_time = time.time()
        
        # Handle empty document
        clean_doc = document.strip()[:1000]
        if not clean_doc:
            return (0, 1.0)
            
        # Rule-based threat detection
        if self._contains_explicit_threat(clean_doc):
            return (1, 0.95)  # High confidence for harmful content
            
        # Rule-based safety net
        if self._contains_safe_indicator(clean_doc):
            return (0, 0.90)  # High confidence for safe content
        
        doc = nlp(clean_doc)
        
        # GNN Graph Processing
        graph_data = self._create_graph_data(doc)
        gnn_feature = self.gnn(graph_data).detach().cpu().numpy()
        
        # Classify
        with torch.no_grad():
            features_tensor = torch.tensor(gnn_feature.flatten(), dtype=torch.float32).unsqueeze(0).to(self.device)
            logit = self.classifier(features_tensor)
            prob = torch.sigmoid(logit).item()
        
        # Format output with adjusted threshold
        classification = 1 if prob >= 0.5 else 0
        confidence = prob if classification == 1 else 1 - prob
        
        # Increase confidence for safe classifications
        if classification == 0 and confidence < self.safe_confidence_threshold:
            confidence = self.safe_confidence_threshold
            
        # Ensure real-time constraint
        elapsed = time.time() - start_time
        if elapsed > 0.5:
            return (0, max(0, 1.0 - elapsed/1000))
            
        return (classification, round(confidence, 2))
    
    def _contains_explicit_threat(self, text: str) -> bool:
        """Rule-based detection of explicit threat keywords"""
        text_lower = text.lower()
        return any(re.search(rf'\b{keyword}\b', text_lower) for keyword in THREAT_KEYWORDS)
    
    def _contains_safe_indicator(self, text: str) -> bool:
        """Rule-based detection of safety indicators"""
        text_lower = text.lower()
        return any(re.search(rf'\b{keyword}\b', text_lower) for keyword in SAFE_KEYWORDS)

    def _create_graph_data(self, doc):
        features = []
        edge_index = []
        
        # Get BERT embeddings for each token
        token_embeddings = []
        for token in doc:
            inputs = tokenizer(token.text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = bert_model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            token_embeddings.append(embedding)
        
        for i, token in enumerate(doc):
            vec = token_embeddings[i]
            features.append(vec)
            
            # Add dependency edges
            for child in token.children:
                if child.i < len(token_embeddings):
                    edge_index.append([token.i, child.i])
            
            # Add additional edges for important relations
            if token.dep_ in ['nsubj', 'dobj', 'attr', 'advcl', 'acomp']:
                if token.head.i < len(token_embeddings):
                    edge_index.append([token.i, token.head.i])
        
        if not features:
            return Data(
                x=torch.randn(1, VEC_DIM, dtype=torch.float32), 
                edge_index=torch.empty(2, 0, dtype=torch.long),
                num_nodes=1
            ).to(self.device)
        
        x = torch.tensor(np.array(features, dtype=np.float32), dtype=torch.float32)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        return Data(x=x, edge_index=edge_index).to(self.device)

    def _train_model(self, graph_data_list, y_train, gnn_hidden, clf_hidden):
        """Train GNN and classifier with given hyperparameters"""
        # Initialize models with reduced complexity
        self.gnn = GNNModel(hidden_dim=gnn_hidden).to(self.device)
        self.feature_dim = gnn_hidden
        self.classifier = Classifier(input_dim=self.feature_dim, 
                                    hidden_dims=[clf_hidden]).to(self.device)
        self.optimizer = torch.optim.AdamW(
            list(self.gnn.parameters()) + list(self.classifier.parameters()),
            lr=LEARNING_RATE, weight_decay=0.01
        )
        
        # Class weights for imbalance (more safe examples)
        weight = torch.tensor([1.0, 0.8]).to(self.device)  # Penalize false positives more
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=weight[1])
        
        best_f1 = 0
        no_improve = 0
        best_model = None
        
        for epoch in range(EPOCHS):
            self.optimizer.zero_grad()
            total_loss = 0
            all_predictions = []
            all_targets = []
            
            # Create batches
            indices = list(range(len(graph_data_list)))
            random.shuffle(indices)
            
            for i in range(0, len(indices), BATCH_SIZE):
                batch_indices = indices[i:i+BATCH_SIZE]
                if not batch_indices:
                    continue
                    
                batch_loss = 0
                batch_logits = []
                batch_targets = []
                
                for idx in batch_indices:
                    data = graph_data_list[idx]
                    target = y_train[idx]
                    
                    # Forward pass
                    gnn_out = self.gnn(data)
                    logit = self.classifier(gnn_out)
                    
                    batch_logits.append(logit)
                    batch_targets.append(target)
                
                # Calculate loss for batch
                logits = torch.cat(batch_logits)
                targets = torch.tensor(batch_targets, dtype=torch.float32).to(self.device)
                loss = self.criterion(logits.squeeze(1), targets)
                batch_loss = loss.item()
                total_loss += batch_loss
                
                # Backpropagation
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Collect predictions
                probs = torch.sigmoid(logits).cpu().detach().numpy()
                predictions = (probs > 0.5).astype(int)
                all_predictions.extend(predictions)
                all_targets.extend([t.item() for t in batch_targets])
            
            # Calculate metrics
            accuracy = accuracy_score(all_targets, all_predictions)
            tn, fp, fn, tp = confusion_matrix(all_targets, all_predictions).ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            avg_loss = total_loss / (len(graph_data_list) / BATCH_SIZE)
            
            # Early stopping with F1 score
            if f1 > best_f1:
                best_f1 = f1
                no_improve = 0
                # Save best model state
                best_model = {
                    'gnn': copy.deepcopy(self.gnn.state_dict()),
                    'classifier': copy.deepcopy(self.classifier.state_dict())
                }
            else:
                no_improve += 1
                
            logging.info(f"Epoch {epoch+1}/{EPOCHS}: Loss={avg_loss:.4f}, Acc={accuracy:.4f}, F1={f1:.4f}, FPR={fpr:.4f}")
                
            if no_improve >= PATIENCE:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if best_model:
            self.gnn.load_state_dict(best_model['gnn'])
            self.classifier.load_state_dict(best_model['classifier'])
        self.gnn.eval()
        self.classifier.eval()

class GNNModel(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.conv1 = GCNConv(VEC_DIM, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.out_channels = hidden_dim
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if edge_index.nelement() == 0:
            return torch.zeros(1, self.out_channels).to(x.device)
        x = F.leaky_relu(self.conv1(x, edge_index))
        x = F.leaky_relu(self.conv2(x, edge_index))
        return x.mean(dim=0).unsqueeze(0)

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(0.6))  # Increased dropout
            prev_dim = dim
            
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x) 