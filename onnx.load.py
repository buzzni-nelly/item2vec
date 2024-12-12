import onnxruntime as ort
import numpy as np

# ONNX 모델 로드
onnx_model_path = "CARCA.onnx"
ort_session = ort.InferenceSession(onnx_model_path)

# 입력 데이터 생성
batch_size = 32
max_len = 50  # max_len은 모델에 따라 동적으로 설정해야 합니다.
num_candidates = 10
num_items = 1000  # num_items은 실제 모델에 따라 변경 필요.

# 더미 입력 데이터 (실제 데이터로 교체 가능)
input_seqs = np.random.randint(0, num_items, (batch_size, max_len)).astype(np.int64)  # 입력 시퀀스
src_key_padding_mask = np.random.randint(0, 2, (batch_size, max_len)).astype(bool)   # 패딩 마스크
last_idxs = np.random.randint(0, max_len, (batch_size,)).astype(np.int64)            # 마지막 인덱스
candidate_idxs = np.random.randint(0, num_items, (batch_size, num_candidates)).astype(np.int64)  # 후보군 인덱스

print("Model Inputs:")
for input in ort_session.get_inputs():
    print(f"Name: {input.name}, Shape: {input.shape}, Type: {input.type}")

# ONNX 모델 실행
outputs = ort_session.run(
    None,  # 출력 이름, None이면 모든 출력 반환
    {
        "input_seqs": input_seqs,
        "src_key_padding_mask": src_key_padding_mask,
        "last_idxs": last_idxs,
        "candidate_idxs": candidate_idxs,
    },
)

# 결과 출력
print("ONNX Model Output:", outputs[1])  # 첫 번째 출력
