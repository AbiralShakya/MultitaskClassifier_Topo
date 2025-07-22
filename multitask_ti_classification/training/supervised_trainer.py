import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np

def train_supervised(model, train_loader, val_loader, epochs, lr, class_weights=None):
    # CrossEntropyLoss for multi-class classification (3 classes: Trivial, TI, Semimetal)
    criterion = nn.CrossEntropyLoss(weight=class_weights) # class_weights should be tensor with 3 elements
    # Auxiliary loss for band gap (if applicable)
    aux_criterion = nn.MSELoss() 

    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Freeze encoders initially if you want to only train the fusion head
    # for param in model.real_space_encoder.parameters():
    #     param.requires_grad = False
    # ... then unfreeze after a few epochs. Or just train end-to-end.

    best_val_auc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Ensure data is on correct device
            batch.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

            logits = model(batch)
            main_loss = criterion(logits, batch.y)

            # If you have an auxiliary head (conceptual):
            # aux_predictions = model.aux_head(h_fused) # You'd need to modify FusionHead to output this
            # aux_loss = aux_criterion(aux_predictions, batch.band_gap)
            # total_loss = main_loss + 0.1 * aux_loss # Weighted sum

            total_loss = main_loss # For now, just main loss

            total_loss.backward()
            optimizer.step()
            total_loss += total_loss.item() * batch.num_graphs

        avg_train_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch} Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for batch in val_loader:
                batch.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                logits = model(batch)
                preds = torch.softmax(logits, dim=1).argmax(dim=1).cpu().numpy()
                targets = batch.y.cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(targets)

        # For multi-class, use accuracy instead of AUC
        val_acc = accuracy_score(val_targets, val_preds)
        
        print(f"Epoch {epoch} Val Acc: {val_acc:.4f}")

        if val_acc > best_val_auc:  # Using accuracy as best metric now
            best_val_auc = val_acc
            torch.save(model.state_dict(), "best_model.pt")
            print("Saved best model!")

# To run training:
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = TopologicalMaterialModel(...).to(device)
# # Calculate class weights for 3 classes: [trivial_weight, ti_weight, semimetal_weight]
# class_weights = torch.tensor([1.0, 1.0, 1.0]).to(device)  # Adjust based on class imbalance
# train_supervised(model, train_loader, val_loader, epochs=50, lr=1e-4, class_weights=class_weights)