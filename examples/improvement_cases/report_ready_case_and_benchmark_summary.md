# 보고서용 누적 사례 및 공식 벤치마크 요약

## 누적 전이 사례

- 같은 질문 6000개 후보를 기존 checkpoint 네 개에 동일 조건으로 생성했다.
- 새 학습은 하지 않았고, 기존 PT/CPT/SFT/DPO checkpoint만 평가했다.
- 보고서용 40개 코호트는 train split을 제외하고 골랐다.

| stage | 누적 상태 | 사례 수 |
|---|---|---:|
| PT | 시작점: 모두 fail | 40 |
| CPT | Q&A first-solve 없음 | 0 해결 / 40 잔여 |
| SFT | PT/CPT fail 중 처음 해결, DPO에서도 유지 | 30 해결 / 10 잔여 |
| DPO | PT/CPT/SFT fail 중 처음 해결 | 10 해결 / 0 잔여 |
| GRPO | 별도 numeric strict 기준 성공 | 0 |

원본 사례 보고서: `examples/improvement_cases/cumulative_transition_cases.md`

## 공식 벤치마크

- protocol: `kfri_dmpmaker_official_eval_v1`
- metric: `official_next_token_mcq`
- scoring: zero-shot next-token option-letter log probability
- sample cap: 없음
- 대상: 현재 로컬 normalized official snapshot에 존재하는 18개 benchmark 전체
- provenance: 2026-06-10에 만든 normalized snapshot 사용. 대부분은 HF `datasets.load_dataset` 캐시 기반이며, `KMMLU-Pro`와 `KoBALT-700`은 기존 project-local normalized snapshot을 사용
- 제외: manifest에는 `mmlu`, `mmlu_pro`가 있으나 현재 normalized root에 파일이 없어 제외
- 제외 처리: `public_light` 및 `report_light_200`은 공식 결과로 사용하지 않음

| stage | all_18 macro | all_18 weighted | korean_11 macro | korean_11 weighted | official_public_ko_3 macro | official_public_ko_3 weighted |
|---|---:|---:|---:|---:|---:|---:|
| PT | 31.13 | 28.70 | 31.99 | 28.48 | 17.92 | 20.30 |
| CPT | 31.18 | 28.59 | 31.93 | 28.30 | 17.82 | 20.41 |
| SFT | 32.31 | 29.19 | 32.18 | 28.05 | 18.17 | 20.30 |
| DPO | 32.38 | 29.25 | 32.31 | 28.14 | 18.05 | 20.23 |

주요 벤치마크:

| benchmark | PT | CPT | SFT | DPO | DPO Δ vs PT |
|---|---:|---:|---:|---:|---:|
| click | 24.61 | 24.61 | 26.02 | 26.02 | +1.41 |
| kmmlu | 28.70 | 28.34 | 27.91 | 27.99 | -0.71 |
| kmmlu_hard | 24.24 | 24.54 | 24.56 | 24.68 | +0.44 |
| kmmlu_pro | 20.02 | 20.41 | 18.92 | 18.85 | -1.17 |
| kobalt_700 | 9.14 | 8.43 | 9.57 | 9.29 | +0.15 |
| kobest_boolq | 48.01 | 48.29 | 49.72 | 50.00 | +1.99 |
| kobest_wic | 49.13 | 49.21 | 51.59 | 51.75 | +2.62 |
| boolq | 39.85 | 39.97 | 58.13 | 58.07 | +18.22 |

공식 벤치마크 요약: `results/public_benchmarks/official_ko_metrics_11_posttraining_stage_comparison.csv`

## 보고서 해석

- CPT는 추가 사전학습/도메인 적응 단계로 명분은 타당하지만, 같은 질문 Q&A first-solve로는 개선 사례가 잡히지 않았다.
- SFT와 DPO는 누적 질의응답 사례에서 보고서용 개선을 분명하게 보여준다.
- 공식 MCQ 전체 평균은 PT 대비 DPO가 소폭 상승하지만, 한국어 weighted는 하락한다. 따라서 공식 벤치마크를 주된 개선 근거로 밀기보다는 한계와 보조 지표로 제시하는 편이 안전하다.
- GRPO는 현재 strict 성공이 없어 결과표에서는 언급만 하고 성공 단계로 세우지 않는다.
