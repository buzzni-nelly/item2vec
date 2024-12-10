## GPU 환경 변수 설정

PyTorch 모델 실행 시, 특정 GPU를 사용하거나 메모리 할당 전략을 지정하려면 아래의 환경 변수를 설정해야 합니다. 

### 환경 변수 설명

1. **`CUDA_VISIBLE_DEVICES`**  
   - 사용 가능한 GPU를 지정하는 변수입니다. 
   - 값으로 GPU 번호를 입력하여 특정 GPU만 사용하도록 제한할 수 있습니다.
   - 예를 들어, `"1"`을 설정하면 GPU 1번만 사용됩니다. 
   - 다중 GPU를 지정하려면 쉼표로 구분하여 설정합니다. (`"0,1"` 등)

2. **`PYTORCH_CUDA_ALLOC_CONF`**  
   - PyTorch의 GPU 메모리 할당 전략을 조정하는 변수입니다. 
   - `"expandable_segments:True"`는 PyTorch가 GPU 메모리를 필요한 만큼만 점진적으로 할당하도록 설정합니다.
   - 이 설정은 초기 메모리 사용량을 줄이고, 필요한 경우 추가로 메모리를 요청하여 사용하는 방식으로, 메모리 관리 효율성을 향상시킵니다.

### 설정 방법

터미널에서 아래 명령어를 실행하여 환경 변수를 설정합니다:

```bash
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```
