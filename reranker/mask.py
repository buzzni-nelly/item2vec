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
