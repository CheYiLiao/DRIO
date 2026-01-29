import time
import sys
import torch
import numpy as np


def train(model, X, S, Xval, Sval, batch_size, max_iter=10000, lr=1e-3,
          save_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Train a MIWAE or notMIWAE model.

    Args:
        model: MIWAE or notMIWAE model
        X: Training data with missing values filled with 0 [N, D]
        S: Training missingness mask [N, D]
        Xval: Validation data [N_val, D]
        Sval: Validation missingness mask [N_val, D]
        batch_size: Batch size for training
        max_iter: Maximum number of iterations
        lr: Learning rate
        save_path: Path to save best model (optional)
        device: Device to train on

    Returns:
        Dictionary with training history
    """
    # Convert to tensors
    X = torch.tensor(X, dtype=torch.float32, device=device)
    S = torch.tensor(S, dtype=torch.float32, device=device)
    Xval = torch.tensor(Xval, dtype=torch.float32, device=device)
    Sval = torch.tensor(Sval, dtype=torch.float32, device=device)

    N = X.shape[0]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    batch_pointer = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'iterations': []
    }

    # Save initial model if path provided
    if save_path is not None:
        torch.save(model.state_dict(), save_path)

    start = time.time()

    for i in range(max_iter):
        model.train()

        # Get batch
        x_batch = X[batch_pointer:batch_pointer + batch_size]
        s_batch = S[batch_pointer:batch_pointer + batch_size]

        # Forward pass
        optimizer.zero_grad()
        result = model(x_batch, s_batch)
        loss = result['loss']

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update batch pointer
        batch_pointer += batch_size
        if batch_pointer >= N - batch_size:
            batch_pointer = 0
            # Shuffle data
            perm = torch.randperm(N)
            X = X[perm]
            S = S[perm]

        # Validation every 100 iterations
        if i % 100 == 0:
            took = time.time() - start
            start = time.time()

            model.eval()
            with torch.no_grad():
                # Compute validation loss in batches
                val_loss = 0.0
                n_val_batches = max(1, len(Xval) // 100)
                for j in range(n_val_batches):
                    xv = Xval[j * 100:(j + 1) * 100]
                    sv = Sval[j * 100:(j + 1) * 100]
                    if len(xv) > 0:
                        val_result = model(xv, sv)
                        val_loss += val_result['loss'].item()
                val_loss /= n_val_batches

            # Save best model
            if val_loss < best_val_loss and save_path is not None:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)

            history['train_loss'].append(loss.item())
            history['val_loss'].append(val_loss)
            history['iterations'].append(i)

            print(f"{i}/{max_iter} updates, {took:.2f} s, {loss.item():.4f} train_loss, {val_loss:.4f} val_loss")
            sys.stdout.flush()

    return history


class MIWAEDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for MIWAE/notMIWAE training."""

    def __init__(self, X, S):
        """
        Args:
            X: Data with missing values filled with 0 [N, D]
            S: Missingness mask (1=observed, 0=missing) [N, D]
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.S = torch.tensor(S, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.S[idx]


def train_with_dataloader(model, train_dataset, val_dataset, batch_size, max_epochs=100,
                          lr=1e-3, save_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Train using PyTorch DataLoader for better memory efficiency.

    Args:
        model: MIWAE or notMIWAE model
        train_dataset: MIWAEDataset for training
        val_dataset: MIWAEDataset for validation
        batch_size: Batch size
        max_epochs: Maximum number of epochs
        lr: Learning rate
        save_path: Path to save best model
        device: Device to train on

    Returns:
        Dictionary with training history
    """
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=100, shuffle=False
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    history = {
        'train_loss': [],
        'val_loss': [],
        'epochs': []
    }

    for epoch in range(max_epochs):
        model.train()
        train_losses = []

        start = time.time()
        for x_batch, s_batch in train_loader:
            x_batch = x_batch.to(device)
            s_batch = s_batch.to(device)

            optimizer.zero_grad()
            result = model(x_batch, s_batch)
            loss = result['loss']
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_batch, s_batch in val_loader:
                x_batch = x_batch.to(device)
                s_batch = s_batch.to(device)
                result = model(x_batch, s_batch)
                val_losses.append(result['loss'].item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        took = time.time() - start

        if val_loss < best_val_loss and save_path is not None:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['epochs'].append(epoch)

        print(f"Epoch {epoch}/{max_epochs}, {took:.2f} s, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")
        sys.stdout.flush()

    return history
