# 공개 벤치마크 지표 해석과 비교 기준

작성일: 2026-06-16

## 요약

본 보고서의 공개 벤치마크 표는 모델의 대외 경쟁 성능을 주장하기 위한 표가 아니라, 같은 평가 프로토콜 안에서 사전학습 베이스라인, 추가학습(CPT), 지도학습(SFT), 선호학습(DPO)의 변화를 확인하기 위한 진단 결과이다.

특히 KMMLU, CLIcK, KoBEST처럼 이름이 같은 벤치마크라도 공개 모델 기술보고서의 점수와 본 보고서의 점수는 평가 방식이 다를 수 있다. 따라서 EXAONE, A.X, Qwen, Llama 등 공개 모델 보고서의 숫자와 본 보고서의 숫자를 같은 조건의 순위표처럼 직접 비교하기 어렵다.

## 본 보고서의 공개 지표 산출 방식

본 프로젝트의 공개 벤치마크 평가는 다음 방식으로 실행하였다.

| 항목 | 본 프로젝트의 처리 |
| --- | --- |
| 평가 프로토콜 | `kfri_dmpmaker_official_eval_v1` |
| 주 평가 방식 | `official_next_token_mcq` |
| 채점 방식 | 선택지 문자(A/B/C/D 등)의 다음 토큰 로그확률을 비교해 가장 높은 선택지를 예측값으로 사용 |
| 프롬프트 설정 | zero-shot |
| 보고 점수 | 정답 선택지와 예측 선택지가 일치한 비율(accuracy, %) |
| 표본 제한 | 없음. 로컬 normalized snapshot에 존재하는 벤치마크 전체 실행 |
| 사후학습 비교 | PT, CPT, SFT, DPO 네 단계를 같은 프로토콜로 비교 |
| 사용 목적 | 절대 성능 경쟁보다 단계별 변화와 토큰학습량 선택 근거 확인 |

실행 산출물 기준으로 전체 공개 벤치마크는 18개, 한국어 공개 지표는 11개를 사용하였다. 보고서 Table 6에는 한국어 관련 11개 지표를 중심으로 제시하였다.

## 사용한 한국어 공개 지표

| 지표 | 본 보고서의 점수 의미 | 비교할 때 확인할 점 |
| --- | --- | --- |
| KMMLU | 선택지 문자 next-token accuracy | 공개 보고서에서는 5-shot accuracy로 제시되는 경우가 많음 |
| KMMLU-hard | 선택지 문자 next-token accuracy | 본 프로젝트 내부 난도별 변화 확인용 |
| KMMLU-Pro | 선택지 문자 next-token accuracy | gated dataset의 project-local normalized snapshot 사용 |
| CLIcK | 선택지 문자 next-token accuracy | A.X 등 공개 보고서의 CLIcK 점수와 설정 차이가 있을 수 있음 |
| KoBALT-700 | 선택지 문자 next-token accuracy | project-local normalized public snapshot 사용 |
| HAERAE | 선택지 문자 next-token accuracy | 문항 구성과 평가 포맷 차이를 확인해야 함 |
| KoBEST-BoolQ | 선택지 문자 next-token accuracy | EXAONE 3.0 보고서는 5-shot F1로 제시 |
| KoBEST-COPA | 선택지 문자 next-token accuracy | EXAONE 3.0 보고서는 5-shot F1로 제시 |
| KoBEST-HellaSwag | 선택지 문자 next-token accuracy | 4지선다라 25% 근처는 무작위 기준선에 가까움 |
| KoBEST-SentiNeg | 선택지 문자 next-token accuracy | EXAONE 3.0 보고서는 5-shot F1로 제시 |
| KoBEST-WiC | 선택지 문자 next-token accuracy | EXAONE 3.0 보고서는 5-shot F1로 제시 |

## 공개 모델 보고서와 다른 점

공개 모델 기술보고서는 보통 각 모델이 정한 평가 설정을 사용한다. 예를 들어 EXAONE 3.0 기술보고서는 KMMLU를 5-shot accuracy로, KoBEST 계열을 5-shot F1로 보고한다. 반면 본 프로젝트는 같은 모델 계열의 학습 단계 변화를 빠르게 비교하기 위해 zero-shot 선택지 문자 next-token accuracy를 사용하였다.

| 비교 항목 | 본 프로젝트 | EXAONE 3.0 예시 | 해석 |
| --- | --- | --- | --- |
| KMMLU | zero-shot next-token accuracy | 5-shot accuracy | 같은 KMMLU라도 조건이 달라 직접 등호 비교 불가 |
| KoBEST-BoolQ/COPA/WiC/HellaSwag/SentiNeg | zero-shot next-token accuracy | 5-shot F1 | accuracy와 F1이 달라 직접 비교 불가 |
| 사후학습 비교 | PT/CPT/SFT/DPO를 동일 프로토콜로 비교 | 공개 모델별 최종 모델 점수 중심 | 본 보고서는 단계별 변화 해석에 더 적합 |
| 절대 성능 주장 | 하지 않음 | 모델 경쟁력 제시 | 본 프로젝트 점수는 경쟁 성능 근거로 쓰면 안 됨 |

즉, 본 보고서의 공개 지표는 "EXAONE보다 낮다/높다"를 엄밀히 말하기 위한 값이 아니라, 동일한 입력 정규화와 동일한 채점 코드 아래에서 우리 모델의 단계별 변화가 어떤 방향으로 움직였는지 보기 위한 값이다.

## 대표 공개 모델 점수와의 거리감

공개 모델의 대표 점수와 비교하면, 본 프로젝트의 공개 벤치마크 절대 점수는 낮은 편이다.

| 지표 | 본 프로젝트 DPO | 공개 모델 예시 | 비교할 때 확인할 점 |
| --- | ---: | ---: | --- |
| KMMLU | 27.99 | EXAONE 3.0 7.8B Inst. 44.5 | EXAONE은 5-shot accuracy |
| KoBEST-BoolQ | 50.00 | EXAONE 3.0 7.8B Inst. 91.5 | EXAONE은 5-shot F1 |
| KoBEST-COPA | 50.50 | EXAONE 3.0 7.8B Inst. 85.0 | EXAONE은 5-shot F1 |
| KoBEST-HellaSwag | 24.60 | EXAONE 3.0 7.8B Inst. 49.1 | EXAONE은 5-shot F1 |
| KoBEST-SentiNeg | 51.13 | EXAONE 3.0 7.8B Inst. 98.7 | EXAONE은 5-shot F1 |
| KoBEST-WiC | 51.75 | EXAONE 3.0 7.8B Inst. 71.2 | EXAONE은 5-shot F1 |
| CLIcK | 26.02 | A.X 4.0 72B 83.5 | 모델 규모와 평가 설정 모두 다름 |

이 표는 본 프로젝트가 공개 모델과 직접 경쟁 가능한 수준이라는 뜻이 아니다. 오히려 공개 벤치마크 절대 성능이 낮다는 점을 분명히 보여준다.

## 무작위 기준선과 해석

일부 점수는 무작위 선택 기준선 근처에 있다.

| 유형 | 무작위 기준선 | 본 프로젝트 해석 |
| --- | ---: | --- |
| 4지선다 객관식 | 약 25% | KMMLU, HellaSwag류에서 25% 근처면 실질 성능이 낮다고 봐야 함 |
| 2지선다 객관식 | 약 50% | BoolQ, COPA, WiC, SentiNeg류에서 50% 근처면 강한 성능 주장 불가 |

따라서 본 프로젝트의 KoBEST 일부 결과가 50% 안팎이라는 사실은 "상당한 성능"이 아니라, 해당 조건에서는 무작위 기준선에 가깝다는 뜻으로 해석해야 한다.

## 보고서에서 안전한 표현

다음 표현은 안전하다.

> 공개 한국어 벤치마크에서 절대 정답률은 공개 한국어 특화 LLM과 비교해 낮은 수준에 머물렀다. 특히 KoBEST 일부 항목은 무작위 선택 기준선에 가까워, 본 실험의 공개 지표 결과를 경쟁 성능으로 해석하기는 어렵다. 따라서 본 보고서에서는 공개 지표를 모델 완성도의 근거라기보다 사전학습 토큰학습량 선택과 사후학습 단계별 변화 확인을 위한 진단 지표로 사용하였다.

다음 표현은 피해야 한다.

- "공개 벤치마크에서 경쟁력을 확인하였다."
- "EXAONE, A.X 등 공개 한국어 모델과 유사한 수준에 도달하였다."
- "KoBEST에서 50점을 기록해 성능이 확보되었다."
- "DPO가 공개 지표 전반을 개선하였다."

## 본 보고서에서의 결론 위치

본 프로젝트의 공개 지표는 낮다. 다만 모든 단계에 같은 채점 방식을 적용했기 때문에, 다음 두 가지 용도로는 의미가 있다.

1. 사전학습 토큰학습량 선택  
   15B와 16B를 같은 프로토콜로 비교했을 때, 16B가 일부 지표에서는 나아졌지만 한국어 확장 가중 평균에서는 낮아졌다. 따라서 15B를 최종 사전학습 기준 베이스로 선택한 근거로 사용할 수 있다.

2. 사후학습 단계별 변화 확인  
   CPT, SFT, DPO가 공개 선택지형 정답률을 크게 끌어올리지는 못했다. 오히려 일부 지표는 하락했다. 따라서 사후학습의 성과는 공개 객관식 점수 상승이 아니라, 자체 생성형 평가와 사례 검토에서 확인한 반복 억제, 직접 답변, 근거 없는 단정 감소, 답변 형식 안정화로 설명하는 편이 맞다.

## 참고 자료

- 본 프로젝트 공개 지표 요약: `results/public_benchmarks/official_ko_metrics_11_posttraining_stage_comparison.csv`
- 본 프로젝트 평가 해석 문서: `docs/evaluation_summary.md`
- EXAONE 3.0 technical report: https://arxiv.org/html/2408.03541v1
- EXAONE 4.0 technical report: https://arxiv.org/html/2507.11407v2
- A.X 4.0 README: https://github.com/SKT-AI/A.X-4.0/blob/main/README.en.md
