def tokenize_texts(texts):
    tokenized_texts = [list(text) for text in texts]
    token2idx = {ch: idx for idx, ch in enumerate(sorted(set(''.join(texts))))}
    input_ids = [[token2idx[token] for token in tokens] for tokens in tokenized_texts]
    return input_ids, token2idx

def create_one_hot_encodings(input_ids, vocab_size):
    import torch
    import torch.nn.functional as F

    inputs_ids = torch.tensor(input_ids)
    one_hot_encodings = F.one_hot(inputs_ids, num_classes=vocab_size)
    return one_hot_encodings
