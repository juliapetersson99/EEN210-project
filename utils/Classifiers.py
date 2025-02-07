import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from  sklearn.svm import SVC



class LSTMClassifier(nn.Module):
    """
    LSTM-based classifier supporting:
      - __init__: constructor
      - forward : forward pass
      - fit     : training loop
      - predict : inference (class predictions)
      - score   : accuracy computation
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Define the LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        # Final classification layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass:
        x: (batch_size, seq_length, input_size)
        returns logits: (batch_size, output_size)
        """
        batch_size = x.size(0)
        
        # Initialize hidden state and cell state to zeros
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        
        out, (hn, cn) = self.lstm(x, (h0, c0))  # out shape: (batch_size, seq_length, hidden_size)
    
        out = out[:, -1, :]  # (batch_size, hidden_size)
        
        logits = self.fc(out)  # (batch_size, output_size)
        return logits
    
    def fit(
        self,
        X_train,
        y_train,
        epochs=10,
        batch_size=32,
        lr =1e-3,
        optimizer_type='adam', 
        betas=(0, 0), 
        momentum=0.0,
        validation_data=None,
        device=None
    ):
        """
        Args:
            X_train (ndarray): shape (num_samples, seq_len, input_size)
            y_train (ndarray): shape (num_samples,) (integer labels for cross-entropy)
            epochs (int)     : number of training epochs
            batch_size (int) : mini-batch size
            lr (float)       : learning rate
            validation_data  : tuple (X_val, y_val) if you have validation data
            device (str)     : 'cuda' or 'cpu'. If None, will use 'cuda' if available, else 'cpu'.

        Returns:
            history (dict): keys = ['train_loss', 'train_acc', 'val_loss', 'val_acc'],
                            each is a list of length = epochs
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device)

        # Create Datasets and DataLoaders
        train_dataset = SensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if validation_data is not None:
            X_val, y_val = validation_data
            val_dataset = SensorDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            val_loader = None
        
        # Define optimizer and loss function
        if optimizer_type.lower() == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=lr, betas=betas)
        elif optimizer_type.lower() == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum)
        else:
            raise ValueError("Unknown optimizer type!")
        
        loss_eval = nn.CrossEntropyLoss()
        
        # Store metrics each epoch in history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        for epoch in range(epochs):
            self.train()  # set the model to training mode
            running_loss = 0.0
            correct = 0
            total = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                # Forward pass
                logits = self.forward(batch_x)
                loss = loss_eval(logits, batch_y)

                # Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Accumulate metrics
                running_loss += loss.item() * batch_x.size(0)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_x.size(0)
            
            epoch_train_loss = running_loss / total
            epoch_train_acc = correct / total

            if val_loader is not None:
                self.eval()  # set the model to eval mode
                val_running_loss = 0.0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(device)
                        batch_y = batch_y.to(device)
                        logits = self.forward(batch_x)
                        loss = loss_eval(logits, batch_y)

                        val_running_loss += loss.item() * batch_x.size(0)
                        preds = torch.argmax(logits, dim=1)
                        val_correct += (preds == batch_y).sum().item()
                        val_total += batch_x.size(0)
                
                epoch_val_loss = val_running_loss / val_total
                epoch_val_acc = val_correct / val_total
            else:
                epoch_val_loss = None
                epoch_val_acc = None
            
            history['train_loss'].append(epoch_train_loss)
            history['train_acc'].append(epoch_train_acc)
            if epoch_val_loss is not None:
                history['val_loss'].append(epoch_val_loss)
                history['val_acc'].append(epoch_val_acc)
            else:
                history['val_loss'].append(None)
                history['val_acc'].append(None)
            
            print(f"Epoch [{epoch+1}/{epochs}]"
                  f" | Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}"
                  + (f" | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}"
                     if epoch_val_loss is not None else ""))

        return history

    @torch.no_grad()
    def predict(self, X, batch_size=32, device=None):
        """
        Predict class indices for the given data X.
        
        Args:
            X (ndarray): shape (num_samples, seq_len, input_size)
            batch_size (int): batch size for inference
            device (str): 'cuda' or 'cpu'. If None, will use 'cuda' if available, else 'cpu'.

        Returns:
            predictions (ndarray): shape (num_samples,)
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device)
        
        self.eval()  # set the model to eval mode
        
        dataset = SensorDataset(X, np.zeros(len(X)))  # dummy labels
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_preds = []
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            logits = self.forward(batch_x)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
        
        all_preds = np.concatenate(all_preds, axis=0)
        return all_preds

    @torch.no_grad()
    def score(self, X, y, batch_size=32, device=None):
        """
        Compute accuracy on the provided dataset.
        """
        preds = self.predict(X, batch_size=batch_size, device=device)
        accuracy = np.mean(preds == y)
        return accuracy


class SensorDataset(Dataset):
    """    
    PyTorch Dataset wrapping sensor data.
    Args:
        X (ndarray): shape (num_samples, seq_len, num_features)
        y (ndarray): shape (num_samples,) - integer class labels
    """
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Return single sample + label
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.long)
        )



class SVMClassifier:
    """
    SVM-based classifier supporting:
      - __init__: constructor
      - fit     : training loop
      - predict : inference (class predictions

      """
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale'):
        self.model = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        preds = self.predict(X)
        accuracy = np.mean(preds == y)
        return accuracy 
    

class RandomForestClassifier:
    """
    RandomForest-based classifier supporting:
      - __init__: constructor
      - fit     : training loop
      - predict : inference (class predictions
      - score   : accuracy computation
      """
    def __init__(self, n_estimators=100, criterion='gini', max_depth=None):
        self.model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        preds = self.predict(X)
        accuracy = np.mean(preds == y)
        return accuracy