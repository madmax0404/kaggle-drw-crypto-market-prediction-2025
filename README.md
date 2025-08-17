# Kaggle
# DRW - Crypto Market Prediction

https://www.kaggle.com/competitions/drw-crypto-market-prediction/overview
---
## 기술 스택

* **Language**: Python
* **ML Models**: XGBoost, LightGBM, CatBoost
* **Hyperparameter Tuning**: Optuna
* **Feature Selection**: SHAP
* **Data Manipulation & Analysis:** Pandas, NumPy, itertools
* **Visualization**: Matplotlib, Seaborn, Optuna Dashboard
* **Cross-validation**: TimeSeriesSplit (Scikit-learn)
* **OS**: Linux (Ubuntu Desktop 24.04 LTS)
* **IDE**: VSCode, Jupyter Notebook
---
## 프로젝트 개요

Kaggle의 암호화폐 거래 데이터 예측 대회에 참가하여 **상위 11%** 성적을 거두었습니다.

익명화된 거래 내역 데이터(X1~X780, label)를 활용해 미래의 타겟 변수를 예측하였으며, 다양한 Feature Engineering (nonlinear transforms, row-wise aggregations, combination interactions)을 적용했습니다. LightGBM과 SHAP을 이용한 Feature Selection을 수행하였고, Optuna로 Hyperparameter Tuning을 자동화하여 성능을 극대화했습니다. 예측 모델로는 LightGBM, XGBoost, CatBoost를 Ensemble 하였습니다.

대회 평가지표로는 Pearson Correlation Coefficient가 사용되었으며, 저는 최종적으로 0.08230을 달성하였고, 상위 10개 팀의 평균 점수(0.11433)와 비교해 데이터 특성상 한계가 있었음을 확인할 수 있었습니다. 그러나 다양한 최신 ML 기법을 실험적으로 적용하여 성능을 끌어올린 경험은 실무적인 인사이트로 이어졌습니다.

## 문제

암호화폐 시장은 빠르게 변화하는 동시에 예측이 매우 어려운 시장으로, 가격 변동은 유동성, 주문 흐름, 투자자 심리, 구조적 비효율성 등 다양한 요인에 의해 좌우됩니다. 이 때문에 데이터에서 의미 있는 신호를 찾는 것이 쉽지 않습니다.

본 대회의 목표는 DRW의 실제 거래 전략에 활용되는 생산 환경(production) 특성과 공개된 시장 데이터를 함께 활용하여, 단기 암호화폐 선물 가격 변동을 예측하는 모델을 구축하는 것입니다. 참가자는 수백 개의 고차원 특성(X1~X780)과 거래량 관련 지표를 통합하여, 노이즈가 많은 환경에서도 가격 방향성을 효과적으로 포착할 수 있는 예측 신호를 생성해야 합니다.

## 데이터셋

Kaggle을 통해 DRW Trading가 제공한 데이터셋은 아래를 포함합니다 (https://www.kaggle.com/competitions/drw-crypto-market-prediction/data).

**train.parquet**

The training dataset containing all historical market data along with the corresponding labels.
학습 데이터셋으로, 모든 과거 시장 데이터와 해당하는 레이블이 포함되어 있습니다.

|컬럼명|설명|
|---|---|
|timestamp|각 행이 해당하는 분(minute)을 나타내는 타임스탬프 인덱스.|
|bid_qty|주어진 시점에서 매수자가 최고 매수 호가로 사고자 하는 총 수량.|
|ask_qty|주어진 시점에서 매도자가 최저 매도 호가로 팔고자 하는 총 수량.|
|buy_qty|해당 분 동안 최저 매도 호가에서 체결된 매수 거래 총량.|
|sell_qty|해당 분 동안 최고 매수 호가에서 체결된 매도 거래 총량.|
|volume|해당 분 동안 체결된 총 거래량.|
|X_{1,...,780}|독점 데이터 소스로부터 도출된 익명화된 시장 특징(feature) 집합.|
|label|예측해야 하는 익명화된 시장 가격 변동을 나타내는 타깃 변수.|

**test.parquet**

테스트 데이터셋은 train.parquet과 동일한 특징 구조를 가지지만, 다음과 같은 차이가 있습니다:

* **timestamp:** 미래 데이터를 미리 알 수 없도록 모든 타임스탬프가 마스킹되고, 셔플된 후 고유 ID로 대체됨.
* **label:** 테스트 세트의 모든 레이블은 0으로 설정됨.

**sample_submission.csv**

제출 형식을 보여주는 샘플 파일입니다. 제출물은 이 샘플 파일과 동일한 행(row) 수와 구조를 따라야 유효한 것으로 간주됩니다.

## 방법론 및 접근 방식

본 프로젝트는 포괄적인 머신러닝 파이프라인을 따랐습니다:

1.  **데이터 로드 및 초기 탐색:**
    * pandas 라이브러리를 사용하여 parquet 파일들을 로드하고, 초기 건전성 검사(sanity checks)와 EDA(Exploratory Data Analysis)를 통해 기본적인 통계 분석을 수행했습니다.

2.  **데이터 전처리, Feature Engineering 및 Selection 주요 내용:**
    * **전지 작업:** 800개에 근접하는 feature들을 축소시킬 필요가 있었습니다. .
    * **표 형식 데이터:** 결측값을 처리하고, 범주형 특성을 인코딩하며, 의미 있는 특성(features)을 엔지니어링했습니다.
    * **데이터 통합:** 활동량 측정 데이터에서 추출한 특징들과 인구 통계 및 행동 데이터를 모델 훈련을 위한 통합된 데이터셋으로 결합하기 위한 전략을 개발했습니다.

3.  **모델 선택 및 아키텍처:**
    * 분류에 적합한 다양한 머신러닝 모델을 탐색했습니다.
    * XGBoost, LightGBM, Catboost와 같은 그래디언트 부스팅 결정 트리 모델을 사용하고 앙상블했습니다.
    * 이 문제는 분류 문제이지만, 목표 변수가 순서형이기 때문에 회귀 모델을 사용했습니다. 예측값은 SciPy의 scipy.optimize.minimize 메서드를 사용하여 최적의 임계값을 찾아 클래스로 후처리되었습니다.

4.  **훈련 및 검증:**
    * StratifiedKFold를 활용하여 모델의 일반화를 보장했습니다.
    * RMSE 손실 및 **Quadratic Weighted Kappa(QWK)**와 같은 지표를 사용하여 훈련 진행 상황을 모니터링했습니다.
    * 과적합을 방지하기 위한 기술(예: 드롭아웃, 조기 종료, L1/L2 정규화)을 적용했습니다.
  
5.  **하이퍼파라미터 튜닝 및 특징 선택:**
    * 자동화된 하이퍼파라미터 탐색을 위한 최첨단 오픈 소스 라이브러리인 Optuna를 사용하여 하이퍼파라미터 튜닝을 진행했습니다. https://optuna.org/
    * 예시 이미지들:
        * ![Optuna Dashboard](<images/Screenshot from 2025-06-08 17-43-57.png>)
        * ![Optuna hyperparameter tuning 1](<images/learning_rate_optuna_visualization.png>)
        * ![Optuna hyperparameter tuning 2](<images/max_depth_optuna_visualization.png>)
        * ![Optuna hyperparameter tuning 3](<images/subsample_optuna_visualization.png>)
    * 머신러닝 모델 결과 설명을 위한 최첨단 오픈 소스 라이브러리인 **SHAP(SHapley Additive exPlanations)**을 특징 선택에 사용했습니다. https://shap.readthedocs.io/
        * ![SHAP feature importance](<images/feature_importance.png>)

6.  **모델 평가:**
    * 최종 모델의 성능을 주된 평가 지표인 **QWK(Quadratic Weighted Kappa)**를 사용하여 한 번도 보지 못한 테스트 세트로 평가했습니다.
    * "SHAP을 사용하여 어떤 특징들이 예측에 가장 많이 기여했는지 이해하기 위한 해석 가능성 분석을 수행했습니다.


















