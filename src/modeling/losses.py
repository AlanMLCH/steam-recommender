import torch, torch.nn.functional as F

def contrastive_loss(user_emb, item_emb, temperature=0.07):
    logits = (user_emb @ item_emb.T) / temperature
    labels = torch.arange(user_emb.size(0), device=user_emb.device)
    return F.cross_entropy(logits, labels)