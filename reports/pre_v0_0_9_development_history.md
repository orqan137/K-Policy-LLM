# Pre-v0.0.9 Development History

작성일: 2026-06-22

이 문서는 현재 공개 저장소의 v0.0.9 중심 설명을 보완하기 위해, make_llm에서 v0.0.9 이전에 진행한 버전별 실험과 운영 로그를 공개 가능한 수준으로 정리한 기록이다.

원시 로그, 내부 절대 경로, PID, 체크포인트 전체 경로, 데이터 원문은 공개 저장소에 포함하지 않는다. 여기에는 버전별 목적, 확인된 수치, 실패 양상, 그리고 이후 설계에 반영된 교훈만 남긴다.

## 1. 읽는 방법

현재 저장소의 최신 기준은 v0.0.9 계열이다. 이 문서의 v0.0.1-v0.0.8은 최신 모델 성능을 주장하기 위한 계보가 아니라, v0.0.9 설계가 왜 필요했는지를 설명하는 이전 실험 기록이다.

특히 다음 네 가지를 구분해야 한다.

| 구분 | 의미 |
| --- | --- |
| pipeline validation | 학습, 평가, export, serving 경로가 연결되는지 확인한 단계 |
| scratch baseline | 외부 pretrained weight 없이 학습한 비교 기준선 |
| adapted / ablation | 다른 버전 checkpoint나 운영상 편의가 섞인 비교 실험 |
| official candidate | 버전, tokenizer, 데이터팩, checkpoint lineage가 일관된 후보 |

v0.0.9 이전 버전들은 대부분 pipeline validation, scratch baseline, adapted/ablation 성격이 강하다. 따라서 아래 기록은 "어떤 실패를 확인했는가"에 초점을 둔다.

## 2. 버전별 요약

| 버전 | 주된 시기 | 역할 | 핵심 결과 | v0.0.9에 반영된 교훈 |
| --- | --- | --- | --- | --- |
| v0.0.1 | 2026-05-22 | end-to-end 기준선 | 학습, SFT/DPO, GGUF export, llama.cpp serving, public eval 연결 | 평가가 실행되는 것과 모델이 쓸 만한 것은 다르다 |
| v0.0.2 | 2026-05-22 - 05-26 | 선거·정책 자료 DAPT | 작은 선거 corpus 반복 학습은 초반 이후 public eval 하락 | 선거 표·통계 자료는 pretrain 반복보다 grounded numeric SFT/eval에 가깝다 |
| v0.0.3 | 2026-05-27 - 05-31 | 1.2B scratch one-pass baseline | 361.3M tokens 학습 완료, public score 일부 개선, 생성형 QA 실패 | public best와 generation health를 분리해야 한다 |
| v0.0.4 | 2026-05-28 | 3B undertrained probe | 3B는 충분히 학습되지 못했고 LR 재개 문제도 확인 | 큰 모델 실패가 아니라 token budget 부족과 운영 정책 문제로 해석해야 한다 |
| v0.0.5 | 2026-05-29 - 05-31 | 3B scratch one-pass baseline | public light macro가 좋아 보였지만 생성형 평가는 붕괴 | public light macro 단독 best는 위험하다 |
| v0.0.6 | 2026-05-31 | 0.46B clean policy baseline | 작은 clean-data 모델도 생성 안정성과 수치 QA를 해결하지 못함 | 모델 크기 축소만으로 도메인 QA가 해결되지 않는다 |
| v0.0.7 | 2026-06-01 - 06-08 | 대용량 pack, resume, posttrain 4계열 실험 | SFT/DPO/RL-like가 stop과 반복은 줄였지만 usable 답변은 회복하지 못함 | weak base 위의 사후학습은 정답성을 만들지 못한다 |
| v0.0.8 | 2026-06-05 - 06-12 | 64K tokenizer 계열 고속 장기학습 운영 | FlashAttention2/compile/batch 운영 개선, 그러나 64K 계보와 생성 품질 한계 확인 | v0.0.9는 80K tokenizer와 별도 lineage로 재정리해야 한다 |

## 3. v0.0.1: pipeline validation baseline

v0.0.1의 가장 중요한 성과는 모델 품질이 아니라 전체 경로가 연결되었다는 점이다. 1.2B급 모델을 학습하고, SFT와 heuristic DPO quick stage를 거친 뒤, GGUF로 변환해 llama.cpp HTTP serving까지 확인했다.

확인된 주요 수치는 다음과 같다.

| 항목 | 값 |
| --- | ---: |
| parameter count | 1,188,153,344 |
| pretrain tokens_seen | 349,999,104 |
| SFT train examples | 358 |
| DPO train examples | 358 |
| DPO final preference accuracy | 0.25 |
| public/common full eval examples | 49,577 |
| public/common macro | 27.8% |
| KMMLU-Pro | 19.38% |
| KoBALT-700 | 8.86% |
| CLIcK | 30.08% |
| GSM8K generation | 0.23% |
| MATH-500 generation | 0.20% |

정성 평가에서는 전화번호 형식, 회의록 템플릿, 기관명 반복, 의미 없는 영어 조각이 자주 생성되었다. 따라서 v0.0.1은 제품 후보가 아니라 baseline release, pipeline validation release로 기록했다.

## 4. v0.0.2: election/policy DAPT and repeat-risk check

v0.0.2에서는 OCR 복구된 선거 정책 자료를 DAPT 형태로 반영했다. 초기 mixed pack에는 election block을 약 5% 넣었고, election block effective repeat factor가 약 19.5배까지 올라갔다.

이후 더 보수적인 2% policy DAPT early-stop 실험도 수행했다.

| 항목 | 값 |
| --- | ---: |
| DAPT total steps | 12,500 |
| optimizer steps | 1,562 |
| tokens seen | 51,200,000 |
| finish status | early_stop_public_eval |
| best checkpoint source | step 2,500 |
| peak GPU memory | about 13.1GB |

public light eval은 초반 step 2,500에서 가장 나았고 이후 개선 없이 하락했다.

| step | kr_avg | overall_avg | CLIcK | KMMLU-Pro | KoBALT-700 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,500 | 25.33% | 27.38% | 48.0% | 18.0% | 10.0% |
| 5,000 | 25.33% | 25.54% | 48.0% | 20.0% | 8.0% |
| 12,500 | 24.00% | 25.23% | 46.0% | 20.0% | 6.0% |

이 단계의 결론은 선거자료 자체가 무의미하다는 것이 아니었다. 문제는 작은 선거 corpus를 pretraining에서 반복 노출하는 방식이었다. 투표율 XLSX와 표 자료는 closed-book pretraining보다 context-grounded numeric SFT/eval 후보로 전환하는 편이 낫다고 판단했다.

## 5. v0.0.3: 1.2B scratch one-pass baseline

v0.0.3은 1.2B급 모델을 random init scratch로 한 번 끝까지 돌린 비교 기준선이다.

| 항목 | 값 |
| --- | ---: |
| tokens_seen | 361,299,968 |
| training steps | 88,208 |
| first loss | 11.5029 |
| last loss | 4.4806 |
| best public light source | step 17,648 |
| matched200 macro, best checkpoint | 28.69% |
| matched200 macro, final checkpoint | 25.00% |

public score만 보면 best checkpoint가 final보다 좋아 보였다. 그러나 선택지 bias가 매우 강했다. 예를 들어 best checkpoint는 ARC-Challenge와 BoolQ에서 거의 전부 A를 예측했고, CommonsenseQA는 거의 전부 E를 예측했다.

SFT sweep도 수행했다. public matched50 macro 기준으로는 일부 개선이 있었다.

| SFT 후보 | public macro | kr_avg |
| --- | ---: | ---: |
| selected v0.0.3 SFT, lr 5e-6 step 30 | 31.38% | 26.67% |
| v0.0.3 pretrain macro-best base | 30.31% | 24.00% |

그러나 v0.0.1-v0.0.6 prepared generation eval에서는 v0.0.3 usable proxy가 0%였다. 즉 public MCQ 개선은 실제 사용자형 generation 품질을 보장하지 않았다.

## 6. v0.0.4: 3B undertraining and resume-policy correction

v0.0.4에서는 3B급 모델을 검토했다. 이 구간의 핵심은 "3B 모델이 실패했다"가 아니라, 모델 크기와 학습 토큰량의 균형이 맞지 않았다는 점이다.

3B extension 시도 중 기존 scheduler를 그대로 이어가면서 learning rate가 약 8.4e-5 수준으로 튀는 문제가 확인되었고, loss 상승을 보고 조기 중단했다. 이후 low-LR model-only continuation으로 바꾸었지만, 사용자가 전략 재검토를 요청하면서 장기 진행을 멈췄다.

운영상 얻은 교훈은 다음과 같다.

| 항목 | 교정 내용 |
| --- | --- |
| resume | optimizer/scheduler까지 exact resume할지, model-only continuation인지 명시 |
| version lineage | 다른 버전 checkpoint에서 이어가면 official version이 아니라 adapted/ablation으로 표시 |
| 3B 해석 | 큰 모델 자체의 실패가 아니라 token budget 부족과 undertraining으로 해석 |
| best alias | `checkpoint_best_public`만 쓰지 않고 기준과 run lineage를 함께 기록 |

## 7. v0.0.5: 3B scratch one-pass and public false positive

v0.0.5는 v0.0.4 continuation을 중단하고 3B급 모델을 random init scratch lineage로 다시 시작한 one-pass run이다. 사용 데이터는 361.3M token pack이었다.

| 항목 | 값 |
| --- | ---: |
| model scale | about 3B |
| packed tokens | 361,299,968 |
| planned steps | 88,208 |
| public light macro, best selected step | 32.2% |
| TruthfulQA MC1, best light | 100.0% |
| macro excluding TruthfulQA | 26.5% |
| Korean macro | 25.3% |

이 수치만 보면 v0.0.5가 좋아 보일 수 있다. 그러나 prepared grounded/numeric generation eval에서는 정반대 결론이 나왔다.

| metric | v0.0.5 |
| --- | ---: |
| nonempty rate | 29.2% |
| repeat flag rate | 16.8% |
| max token rate | 100.0% |
| numeric_all3_rate | 0.0% |
| usable proxy rate | 0.0% |

v0.0.5의 의미는 "좋은 3B 후보"가 아니라, public light macro와 open-ended generation 품질이 강하게 어긋날 수 있다는 반례였다.

## 8. v0.0.6: 0.46B clean policy baseline

v0.0.6은 데이터 양에 비해 모델이 너무 큰 문제를 확인하기 위해 0.46B급으로 줄인 clean-data 실험이다. 표, 투표율, 총람, XLSX, 숫자 평가용 자료는 pretraining에서 제외하고, 350M phase1 데이터에 clean policy 자료를 약 2% 섞었다.

| 항목 | 값 |
| --- | ---: |
| parameter count | 455,527,680 |
| training steps | 85,448 |
| public light macro, best selected step | 32.2% |
| TruthfulQA MC1, best light | 100.0% |
| macro excluding TruthfulQA | 26.5% |
| Korean macro | 25.3% |

그러나 generation eval에서는 반복과 숫자 실패가 계속 남았다.

| metric | v0.0.6 |
| --- | ---: |
| nonempty rate | 100.0% |
| repeat flag rate | 87.3% |
| max token rate | 91.4% |
| cue hit rate | 3.4% |
| numeric_all3_rate | 0.0% |
| usable proxy rate | 0.0% |

따라서 v0.0.6은 작은 모델이 빠르게 학습될 수 있다는 확인에는 의미가 있었지만, 목표 도메인 QA와 수치 해석을 해결한 후보는 아니었다.

## 9. v0.0.1-v0.0.6 prepared generation eval

2026-06-05에는 v0.0.1부터 v0.0.6까지 같은 prepared grounded/numeric prompt 291개를 직접 생성시켜 비교했다. 입력은 grounded eval 211개와 numeric eval 80개였고, 각 버전에 대해 동일한 prompt set을 사용했다.

| version | n | nonempty_rate | repeat_flag_rate | max_token_rate | cue_hit_rate | numeric_all3_rate | usable_proxy_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v0.0.1 | 291 | 100.0% | 95.9% | 100.0% | 3.4% | 0.0% | 0.7% |
| v0.0.2 | 291 | 100.0% | 79.7% | 100.0% | 10.7% | 0.0% | 3.4% |
| v0.0.3 | 291 | 100.0% | 93.5% | 95.2% | 6.2% | 0.0% | 0.0% |
| v0.0.4 | 291 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% |
| v0.0.5 | 291 | 29.2% | 16.8% | 100.0% | 0.0% | 0.0% | 0.0% |
| v0.0.6 | 291 | 100.0% | 87.3% | 91.4% | 3.4% | 0.0% | 0.0% |

가장 중요한 결과는 모든 버전에서 `numeric_all3_rate = 0.0%`였다는 점이다. prompt 안에 표와 숫자가 직접 들어 있어도 좌측 값, 우측 값, 차이를 동시에 맞히지 못했다.

이 평가 이후 public benchmark는 후보 선별의 보조 신호로 낮추고, generation health, grounded QA, numeric QA, refusal/uncertainty, manual sample review를 별도 gate로 두는 방향으로 바꾸었다.

## 10. v0.0.7: large pack, fast resume, and posttraining limits

v0.0.7은 대용량 all-source pack과 장기 학습 운영을 시도한 버전이다. 처음에는 source group 묶음 때문에 일부 legacy source가 pack에 충분히 들어가지 않는 문제가 있었고, 이후 source group을 세분화했다.

v0.0.7 full unique pack은 한국어와 영어 source를 넓게 포함했지만, early active run에서는 영어 external 비중과 resume 방식 문제가 남았다. fast resume은 모델, optimizer, scheduler, RNG는 이어받았지만 data iterator state가 없는 checkpoint에서 출발했기 때문에 exact data-cursor resume으로 볼 수 없었다.

속도 최적화도 진행했다.

| 항목 | 초기 v0.0.7 | fast resume |
| --- | ---: | ---: |
| micro batch | 1 | 3 |
| gradient checkpointing | on | off |
| optimizer | AdamW | fused AdamW |
| TF32 | off | on |
| public eval | rolling save마다 동기 실행 | milestone/final 비동기 실행 |
| 처리량 | about 2,864 tok/s | about 3,550 tok/s |

이 개선은 약 24% 속도 향상이었지만, 20B-24B 이상을 단일 A6000에서 빠르게 끝내기에는 여전히 부족했다.

### 10.1 v0.0.7 posttraining 4-way test

v0.0.7 마지막 checkpoint 계열에서 SFT, DPO, GRPO-like, PPO-like 실험을 수행했다. 이때 base checkpoint는 약 1.509B tokens_seen 지점이었다.

| alias | n | correct | usable | stopped | repeat | avg output tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base v0.0.7 | 35 | 1 | 0 | 0 | 17 | 64.0 |
| SFT | 35 | 1 | 0 | 17 | 5 | 45.9 |
| DPO | 35 | 0 | 0 | 19 | 8 | 42.7 |
| GRPO-like | 35 | 1 | 0 | 20 | 5 | 46.3 |
| PPO-like | 35 | 1 | 0 | 20 | 4 | 41.8 |

SFT는 eval loss를 낮추고 stop/EOS와 반복 억제에는 효과가 있었다. 그러나 usable 답변은 0/35로 유지되었다. DPO는 preference loss를 낮췄지만 qualitative 정답성은 회복하지 못했다. GRPO-like와 PPO-like는 정식 TRL GRPO/PPO가 아니라 local reward policy probe였고, expected numeric value rate는 0으로 남았다.

결론은 명확했다. 사후학습 방법을 늘리는 것만으로 weak base를 고칠 수 없다. base pretraining이 문장 생성, 지시 이해, 짧은 QA, 숫자 복사·계산의 최소 gate를 통과해야 사후학습이 의미가 있다.

## 11. v0.0.8: 64K tokenizer lineage and high-speed training operation

v0.0.8은 v0.0.7까지의 병목을 바탕으로 A6000에서 장기 학습을 더 빠르고 안전하게 돌리는 방법을 정리한 버전이다.

v0.0.8 official pack은 14개 FineWeb2 Korean shard와 기존 source를 사용해 구성했다.

| 항목 | 값 |
| --- | ---: |
| packed tokens | 33.976B |
| repeated tokens | 0 |
| Korean ratio | 91.08% |
| English ratio | 8.92% |
| domain-related ratio | 11.40% |
| FineWeb2 policy keyword route | 10.91% |
| official primary political/legal source ratio | 0.54% |

FineWeb2 policy keyword route는 공식 정치·법률 자료가 아니라, 키워드 기반으로 분리한 웹 보조 문서다. 따라서 보고서에서는 official primary source와 분리해서 설명하도록 정리했다.

### 11.1 speed recipe

v0.0.8에서 검증한 고속 학습 기본값은 다음과 같다.

| 설정 | 결정 |
| --- | --- |
| attention | FlashAttention2 |
| torch.compile | A6000 smoke 통과 후 max-autotune-no-cudagraphs 채택 |
| gradient checkpointing | false |
| optimizer | fused AdamW |
| TF32 | enabled |
| loss chunk tokens | 256 |
| micro batch / accumulation | b3 또는 b4 계열 probe 후 선택 |

A6000 smoke에서 FlashAttention2 + compile no-CUDAGraph 조합은 no-compile 대비 steady throughput이 약 23% 빨랐다. 이 운영 개선은 v0.0.9 장기 학습 설계에도 반영되었다.

### 11.2 early generation probe

v0.0.8 초기 checkpoint는 약 0.201B tokens_seen 지점에서 qualitative probe를 수행했다.

| alias | n | correct | usable | stopped | repeat | avg output tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| temp 0.2 | 35 | 0 | 0 | 7 | 21 | 57.9 |
| greedy | 35 | 1 | 0 | 5 | 21 | 60.5 |

이 시점의 모델은 아직 generation health gate를 통과할 단계가 아니었다. 날짜 조각, 숫자열, 웹 boilerplate 조각이 많았고, sampling 설정만의 문제가 아니었다.

### 11.3 v0.0.8 posttraining comparison

이후 step 190,464 계열에서 CPT, SFT, DPO 비교 실험을 수행했다.

| alias | raw usable | chat usable | 관찰 |
| --- | ---: | ---: | --- |
| base | 0/12 | 0/12 | 반복과 max-token 도달 |
| CPT | 0/12 | 0/12 | 법령·정책 문체는 강해졌지만 반복이 남음 |
| base -> SFT | 0/12 | 0/12 | stop과 짧은 형식은 개선, 정답성 부족 |
| CPT -> SFT | 0/12 | 0/12 | chat format에서는 매우 짧아졌지만 과도한 단답 발생 |
| CPT -> SFT -> DPO | 0/12 | 1/12 | 근거 부족 답변 1건만 usable |

이 결과는 loss 개선과 생성 품질 개선이 분리될 수 있음을 다시 보여주었다. v0.0.8은 운영과 속도 측면의 진전이 컸지만, 최종 모델 계보로는 64K tokenizer lineage에 묶여 있었다.

## 12. v0.0.9로 넘어간 이유

v0.0.9는 이전 버전들의 단순 연장이 아니라, 위 실패를 반영해 계보를 새로 정리한 버전이다.

| 이전 문제 | v0.0.9에서의 대응 |
| --- | --- |
| 64K tokenizer와 데이터 구성 불일치 | 80K SentencePiece tokenizer를 새로 학습 |
| 작은 corpus 반복과 token budget 부족 | 31.21B packed token pretrain pack 구성 |
| public light macro false positive | public benchmark와 generation health를 분리 |
| checkpoint alias 혼선 | version lineage, base checkpoint, resume_from, best 기준 명시 |
| weak base 위 SFT/DPO 실패 | 사후학습 전 base milestone 평가 강화 |
| official source 비율 낮음 | CPT와 SFT/DPO에서 법령·정책 문체와 구조화 답변 보강 |
| 숫자·표 QA 0% 문제 | numeric SFT, GRPO/RLVR 후보, 별도 numeric eval로 분리 |
| 최신 정치 사실 단정 위험 | current-fact/RAG/기준일 처리를 학습만으로 해결하지 않는 정책 채택 |

## 13. 공개 문서에서 피해야 할 해석

- v0.0.1-v0.0.8을 최종 성능 후보처럼 설명하지 않는다.
- v0.0.5와 v0.0.6의 public light macro 32.2%를 실제 질의응답 성능으로 해석하지 않는다.
- TruthfulQA MC1 100%를 truthfulness 확보로 해석하지 않는다.
- v0.0.7 SFT/DPO loss 감소를 usable 답변 개선으로 단정하지 않는다.
- v0.0.8의 speed recipe를 최종 품질 개선으로 혼동하지 않는다.
- FineWeb2 policy keyword route를 official political/legal source로 쓰지 않는다.
- local reward policy probe를 정식 GRPO/PPO 결과처럼 표현하지 않는다.

## 14. 한 줄 결론

v0.0.1-v0.0.8은 실패한 폐기물이 아니라, v0.0.9가 80K tokenizer, 더 큰 pretrain pack, clean CPT/SFT/DPO, multi-gate evaluation으로 재정리되어야 한다는 근거를 남긴 실험 로그다.
