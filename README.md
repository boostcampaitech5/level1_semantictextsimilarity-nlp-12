# 1. 프로젝트 개요
부스트캠프 5기 NLP Track의 Lv.1 project는 다양한 출처에서 얻어진 문장 쌍 데이터를 기반으로 STS task를 잘 수행할 수 있는 모델을 만드는 것이다.

관련하여, NLP-12 자연산활어가철이조는 STS를 잘 소화할 수 있는 Deep learning PLM을 찾아 fine-tuning 하는 것을 기본 전제로 하였으며 크게 모델적인 개선과 데이터적인 정제 두 파트로 나누어 문제를 해결하고자 하였다.

그렇기에, 기본적인 딥러닝 프레임워크 Pytorch, 자연어처리 PLM을 쉽게 사용할 수 있는 huggingface와 시각화를 위한 WandB를 사용하였고, V100이 주어지는 서버를 vscode를 SSH로 받아와 팀 전용 github와 연동하여 협업하였다.

# 2. 프로젝트 팀 구성 및 역할
- 변성훈(팀장) : 노션 자료 정리, EDA, 데이터 시각화, 모델별 output 분포 정리 및 분석
- 김지현 : label error correction 자료조사, R-drop, special token adding 가설 구현, ensemble 구현 및 실험
- 이예원 : 모델 리서치, 모델 성능 실험, 가설 하이퍼파라미터 대조군 비교 및 성능 실험, sheduler, k-fold 실험
- 이상민 : 데이터 분포 고려 데이터 증강 실험, check_point, padding_cutting
- 천재원 : 라벨 페널티, 스페셜 토큰 사용 아이디에이션 및 구현 / Rdrop, Sentence swap 구현 및 wandb 세팅 / 데이터 증강 아이디에이션

# 3. 프로젝트 수행 절차 및 방법

## 1) model 선정

### 후보
- BERT : RoBERTa 하위호환
- **RoBERTa** : 무난히 성능이 좋고, ELECTRA와의 앙상블을 고려했을 때, 좋다고 판단
- **ELECTRA** : 기본 성능이 보장되었고 generator와 discriminator로 학습해서 discriminator만 가져온다는 점에서 데이터셋이 크지 않은 우리의 환경에 적합.
- ALBERT : 좋은 한국어 모델이 없어서 배제
- DistillBERT : 경량화 모델로 배제
- T5 : 큰 모델이기도 하고 시간 상 배제

![image](https://user-images.githubusercontent.com/126854237/234342782-863f3dd5-3005-49f6-b232-93ab500bff5b.png)

## 2) baseline code 수정
- Early Stopping : 3번 연속 validation pearson score가 하락한다면, stop 하는 기능
- Attention mask : baseline code는 input id만 사용하고 attention mask를 model에 넣어주지 않음을 확인. > attention mask도 추가
- Wandb setting : project name은 사용 모델 이름, run name은 hyperparameter와 같은 특징을 사용해서 기록

## 3) EDA
dev.csv는 모든 label이 비교적 uniform
좋은 제출 score를 가진 모델의 csv 분포를 분석해본 결과 test.csv도 uniform한 분포라고 가정

여러가지 방법으로 데이터 증강을 하였지만 그 중에서 실제로 결과 제출에 사용한 방법은 아래와 같다.

우선 label 0의 샘플을 1000개로 줄이고 잘라낸 1000개에서 label 5를 다음과 같은 방식으로 증강하였다.
Label 5의 샘플들은 모두 공통적으로 sentence 2의 문장이 sentence 1의 문장에 대해 1. 띄어쓰기가 하나 빠져 있거나 2. 문장 끝에 마침표가 추가로 2~3개 연달아 온다는 특징이 있다. 그래서 Okt의 형태소 분석 기능을 사용하여 아래와 같은 조합의 단어 쌍 띄어쓰기가 나온다면 해당 띄어쓰기를 지워서 붙인 버전을 sentence 2로 증강했다.
| 명사 + 명사 | (단독 입찰 / 단독입찰) |
| 형용사 + 명사 | (상쾌한 아침 / 상쾌한아침) |
| 관형사 + 명사 | (이 것 / 이것) |
| 동사 + 명사 | (빛나는 우리 / 빛나는우리) |
| 부사 + 동사 | (밝게 빛나 / 밝게빛나) |

만약 label 5를 0과 똑같이 1000개로 증강할 경우 test.csv 를 통과한 output의 분포를 보았을 때, 전체적으로 5.0 쪽에 더 많은 예측이 지속적으로 나오는 것을 관찰할 수 있었다. 이는 5.0 샘플의 특징은 단순하고 적은 반면에, 0 샘플을 예측하기 위해서는 더 많은 것을 고려해야 하기 때문이라고 추측하였다. 그렇기에 5.0 label 증강을 조절하거나 5.0에 주는 penalty를 늘려서 이를 완화하였다.

또한 아래에서 언급될 special token 을 사용하기 위해 5.0 증강된 데이터이 source는 본래의 source를 그대로 가져오고 0에서의 sampled는 rtt로 바꾸어 주었다. 

## 4) 가설 구현
- Label Penalty
  - 주어진 train.csv는 label 0.0이 전체의 20%를 차지함. 모델이 아예 학습을 하지 않고 0.0으로만 예측해도 Pearson 0.2를 넘길 수 있을 정도.
  - Ground-Truth가 0.0이 아님에도, 모델이 0.0을 예측할 경우 해당 샘플의 기존 loss의 alpha배만큼 페널티 loss를 추가로 받게 함.
  - 추후 5.0에도 penalty를 부여

- Sentence Swap & R-Drop
  - Sentence Swap : 단순히 문장의 순서를 바꾼 경우에 대하여 학습
  - R-Drop : 그 때에 sentence1 [SEP] sentence2의 logit 값과 sentence2 [SEP] sentence1의 logit값의 차이 또한 Loss로 계산
  
- Special Token Leverage : [NSMC], [SAMPLED], [RTT]등, Sample id에 제공된 해당 샘플의 출처 및 속성값을 Special Token으로 추가 후 사용
  - vanila usage
    - 문장 쌍 앞에 토큰을 추가한 후(Ex. [CLS] [NSMC] [RTT] sentence1 [SEP] sentence2 [SEP] [PAD] [PAD]...),
    - 원래와 같이 [CLS] 토큰으로만 추론을 진행
  - special token w/ linear
    - 5개의 special token (Ex. [CLS], [NSMC], [RTT], [SEP], [SEP])을 concat한 후, linear layer block을 통과하여 1차원으로 projection
  - special token w/ voting
    - 5개의 special token을 각각 1차원으로 projection하여 5개의 후보 logits를 생성
    - 학습 가능한 5개의 weights로 logits를 weighted sum하여 최종 logit 생성

## 5) ensemble
### weighted average ensemble
- Ensemble 에 사용할 모델들의 실제 submit score 를 weight 로 사용하여 ensemble 모델의 예측 정확도를 높이려 노력
- 해당 score 들은 softmax를 통과하여 각 logit 과 곱해진 후 합해지는 weighted sum 의 형태
- 실험 결과 logit 들의 분포가 다를 경우(각 모델의 상관 관계가 낮을 경우) 성능 향상의 폭이 더 컸음
<br><br>

### 실험 결과
- 각 모델의 상관 관계가 낮을 수록 성능 향상의 폭이 크다.

  1. submit score: 0.9248

  |Model|submit score|
  |---|---|
  |snunlp/KR-ELECTRA-discriminator|0.9215|
  |snunlp/KR-ELECTRA-discriminator|0.9212|
  |snunlp/KR-ELECTRA-discriminator|0.9177|

  2. submit score: 0.9300 (+0.0052)

  |Model|submit score|
  |---|---|
  |snunlp/KR-ELECTRA-discriminator|0.9215|
  |snunlp/KR-ELECTRA-discriminator|0.9212|
  |klue/roberta-large|0.8999|

  `평균 제출 score 가 더 높음에도 불구하고, roberta-large 를 함께 앙상블한 결과가 더 높은 score 를 가진다.`
  
- 다양한 target 분포를 가진 모델을 앙상블 하면 성능이 높아진다.
   
   3. submit score: 0.9324 (+0.0024)

  |Model|submit score|
  |---|---|
  |snunlp/KR-ELECTRA-discriminator|0.9215|
  |snunlp/KR-ELECTRA-discriminator|0.9212|
  |snunlp/KR-ELECTRA-discriminator|임의의 값|
  |klue/roberta-large|0.9205|
  |klue/roberta-large|0.8999|
  |kykim/electra-kor-base|0.9165|

  `Target 값의 분포를 기준으로 각 모델을 보완 해 줄 수 있는 모델을 앙상블 한 결과가 더 높은 score 를 가진다. `
  
- 가장 높은 성능의 ensemble output

   4. submit score: 0.9352 (+0.0028) -> **최종 pearson : 0.9420**

  |Model|submit score|
  |---|---|
  |snunlp/KR-ELECTRA-discriminator|0.9238|
  |snunlp/KR-ELECTRA-discriminator|0.9232|
  |klue/roberta-large|0.9205|
  |kykim/electra-kor-base|0.9187|
  |beomi/KcELECTRA-base|0.9221|
  |monologg/koelectra-base-discriminator|0.9185|
  
# 4. 프로젝트 수행 결과

![image](https://user-images.githubusercontent.com/126854237/234345560-e7921bc6-d3ef-4ebd-ab24-d237b579ed33.png)

- 최종 pearson : 0.9420
