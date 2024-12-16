import torch

import directories
import carca
from item2vec.volume import Volume
from carca.modules import CARCA

carca_config_path = directories.config("aboutpet", "carca", "v1")
carca_settings = carca.configs.Settings.load(carca_config_path)

volume = Volume(company_id="aboutpet", model="item2vec", version="v1")

model = CARCA.load_from_checkpoint(
    "/tmp/aboutpet/carca/v1/checkpoints/last.ckpt",
    num_items=volume.vocab_size(),
    embed_dim=carca_settings.embed_dim,
    num_heads=carca_settings.num_heads,
    num_layers=carca_settings.num_layers,
    max_len=carca_settings.max_len,
    dropout=carca_settings.dropout,
    lr=carca_settings.lr,
    weight_decay=carca_settings.weight_decay,
)
model.eval()
model.cpu()

# 더미 입력 텐서 생성
batch_size = 2
max_len = 10
input_seqs = torch.LongTensor(
    [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    ]
)
src_key_padding_mask = torch.BoolTensor(
    [
        [False, False, False, False, False, False, False, True, True, True],
        [False, False, False, False, False, False, False, False, False, False],
    ]
)
last_idxs = torch.LongTensor([[6],[9]])
candidate_idxs = torch.randint(0, model.num_items, (batch_size, 10))  # 후보군 인덱스 (선택 사항)

# ONNX로 저장
torch.onnx.export(
    model,
    (input_seqs, src_key_padding_mask, last_idxs, candidate_idxs),  # 모델 입력
    "CARCA.onnx",  # 저장할 파일 이름
    export_params=True,  # 학습된 가중치 포함
    opset_version=14,  # ONNX opset 버전
    do_constant_folding=True,  # 상수 폴딩 최적화
    input_names=["input_seqs", "src_key_padding_mask", "last_idxs", "candidate_idxs"],  # 입력 이름
    output_names=["output"],  # 출력 이름
    dynamic_axes={  # 동적 축 설정
        "input_seqs": {0: "batch_size", 1: "sequence_length"},
        "src_key_padding_mask": {0: "batch_size", 1: "sequence_length"},
        "masked_idxs": {0: "batch_size"},
        "candidate_pidxs": {0: "batch_size", 1: "num_candidates"},
        "output": {0: "batch_size", 1: "num_scores"},
    },
)
