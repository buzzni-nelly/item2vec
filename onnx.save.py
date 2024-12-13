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
batch_size = 32
input_seqs = torch.randint(0, model.num_items, (batch_size, model.max_len))  # 입력 시퀀스
src_key_padding_mask = torch.randint(0, 2, (batch_size, model.max_len)).bool()  # 패딩 마스크
last_idxs = torch.randint(0, model.max_len, (batch_size,))  # 마지막 인덱스
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
        "last_idxs": {0: "batch_size"},
        "candidate_idxs": {0: "batch_size", 1: "num_candidates"},
        "output": {0: "batch_size", 1: "num_candidates"},
    },
)
