def generate_masked_input(input_seq, pad_token_idx=0, mask_token_idx=10001, mask_prob=0.15):
    """
    입력 시퀀스에서 마스크를 생성하고 레이블을 반환합니다.
    :param input_seq: [batch_size, max_len] - 입력 시퀀스
    :param pad_token_idx: 패딩 토큰 ID
    :param mask_token_idx: 마스크 토큰 ID
    :param mask_prob: 마스크 확률
    :return: 마스크된 입력 시퀀스, 레이블, 레이블 마스크
    """
    # NOTE: softmax 써야 함..
    input_seq = input_seq.clone()
    labels = input_seq.clone()
    mask = (torch.rand(input_seq.shape) < mask_prob) & (input_seq != pad_token_idx)

    # 입력에서 마스크 위치를 마스크 토큰으로 대체
    input_seq[mask] = mask_token_idx

    # 마스크된 위치만 레이블로 유지, 나머지는 -100으로 설정 (PyTorch의 ignore_index)
    labels[~mask] = -100

    return input_seq, labels


# if __name__ == "__main__":
#     torch.manual_seed(42)
#
#     num_items = 10
#     mask_token_idx = num_items + 0
#     pad_token_idx = num_items + 1
#     embed_dim = 4
#     num_heads = 2
#     num_layers = 2
#     max_len = 5
#     dropout = 0.1
#
#     model = BERT4Rec(
#         num_items=num_items,
#         embed_dim=embed_dim,
#         num_heads=num_heads,
#         num_layers=num_layers,
#         max_len=max_len,
#         dropout=dropout,
#     )
#
#     batch_size = 2
#     input_seqs = torch.randint(0, num_items, (batch_size, max_len))
#     input_seqs[0][-1] = pad_token_idx
#     input_seqs[0][-2] = pad_token_idx
#     input_seqs[1][-1] = pad_token_idx
#     padding_mask = input_seqs == pad_token_idx
#
#     batch_indices = torch.arange(input_seqs.size(0))
#     last_indices = (~padding_mask).sum(dim=1) - 1
#     input_seqs[batch_indices, last_indices] = mask_token_idx
#
#     positive_idx = torch.randint(0, num_items, (batch_size,))
#     negative_idx = torch.randint(0, num_items, (batch_size,))
#
#     logits = model(input_seqs, padding_mask)
#     output = logits[torch.arange(logits.size(0)), last_indices, :]
#
#     positive_embeddings = model.item_embedding(positive_idx)
#     negative_embeddings = model.item_embedding(negative_idx)
#     positive_scores = torch.sum(output * positive_embeddings, dim=-1)
#     negative_scores = torch.sum(output * negative_embeddings, dim=-1)
#
#     loss = bpr_loss(positive_scores, negative_scores)
#     print(f"BPR Loss: {loss.item()}")
