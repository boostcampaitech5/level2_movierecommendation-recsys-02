# Recbole 사용법

### Setup
```bash
pip install -r requirements.txt


```
### Files
'Recbole/general/'
* `run_recbole.py`: 학습코드입니다.
* `inference.py`: 추론 후 `submissions.csv` 파일을 만들어주는 소스코드입니다.
* `requirements.txt`: 모델 학습에 필요한 라이브러리들이 정리되어 있습니다.
* `args.py`: `argparse`를 통해 학습에 활용되는 여러 argument들을 받아줍니다.

### 실행방법(예시)


'train'
python run_recbole.py --model_name=MultiVAE --epochs=5

'inference' top_k = 뽑아낼려는 값 갯수
python inferency.py --inference_model=saved/MultiVAE~.pth --top_k=10 