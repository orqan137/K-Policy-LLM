# 누적 전이 사례 검산 결과

## 검산 목적

- 기존 `누적_전이_사례_보고서.md`의 40개 사례가 실제로 같은 prompt를 `PT -> CPT -> SFT -> DPO`에 통과시켜 얻은 누적 개선 사례인지 재검산했다.
- 기계적 조건과 보고서용 품질을 분리해서 판단한다.

## 검산 입력

1. 대표 사례 목록: `examples/improvement_cases/cumulative_transition_cases.md`
2. 사례 검산 요약: 이 문서의 기계 검산 결과와 보수 표기 권장 표
3. 단계별 출력 원본: 공개 저장소에는 전체 JSONL을 포함하지 않고, 보고서용 대표 사례만 정리

## 기계 검산 결과

| 항목 | 결과 |
|---|---:|
| selected count | 40 |
| candidate pool count | 6000 |
| all score rows | 6000 |
| PT outputs | 6000 |
| CPT outputs | 6000 |
| SFT outputs | 6000 |
| DPO outputs | 6000 |
| SFT first-solve retained by DPO | 30 |
| DPO first-solve | 10 |
| train split 포함 | 0 |
| prompt/order mismatch | 0 |
| whitespace-normalized completion mismatch | 0 |

기계 조건 기준으로는 제대로 뽑혔다. 선택된 40개는 전체 score JSONL에 존재하고, 같은 ID/prompt가 네 단계 output과 정렬되어 있으며, pass vector도 선택 그룹과 일치한다.

## 해석할 때 확인할 점

- pass 기준은 자동 휴리스틱이다. `overall >= 0.70`, 반복/명백한 붕괴 패턴 없음, 길이 유효 조건으로 보았기 때문에, 의미적으로 완벽한 정답 보증은 아니다.
- score/report 저장 시 줄바꿈이 공백으로 정규화되어 원본 output과 문자열 hash가 다르게 보였지만, 공백 정규화 후 completion mismatch는 0개였다.
- `dpo_val` split은 train split은 아니지만, DPO 평가/개선용 데이터 계열이다. 외부 일반화 성능으로 말하기보다, 실패사례 개선 검증용 사례로 설명해야 한다.

## 보고서용 제외 또는 보수 표기 권장 사례

| case | id | stage | 이유 |
|---:|---|---|---|
| 5 | `dpo1000:failure_dpo:033658a8152784c9` | SFT | 자동 pass는 맞지만 `형식적`을 `형태적`으로 잘못 쓴 오탈자가 있다. 회의록 요약 사례로는 깔끔하지 않다. |
| 31 | `sft5000:v7_domain_sft:9465c18f89e04fdf73` | DPO | DPO first-solve는 맞지만 score `0.715789`로 threshold 근처이며, 답변 일부 표현이 다소 어색하다. 보수적으로는 후순위. |
| 34 | `dpo1000:failure_dpo:4370a552c8d271bd` | DPO | 자동 pass는 맞지만 회의록 주요 발언 표현이 원문과 완전히 매끈하게 대응하지 않는다. |
| 37 | `dpo1000:failure_dpo:0ade94ea7628131f` | DPO | 질문/기대답변의 `오늘`이 DPO 답변에서 `어제`로 바뀌었다. 보고서용 성공 사례로는 제외 권장. |
| 39 | `dpo1000:failure_dpo:28757d849bc5a3a9` | DPO | 기대답변의 `생계급여`가 DPO 답변에서 `주거급여`로 바뀌고 `긴급 복지은` 문법 오류가 있다. 제외 권장. |
| 40 | `dpo1000:failure_dpo:83f39d88d3472054` | DPO | 핵심 방향은 맞지만 `증액 규모`가 `증액 범위`로 바뀐다. 사용은 가능하나 최상위 사례로는 후순위. |

## 판단

- “누적단위로 개선된 사례를 뽑았는가?”라는 기계적 질문에는 `예`가 맞다.
- “보고서에 그대로 실어도 되는 깨끗한 성공 사례인가?”라는 질문에는 `일부는 교체/제외가 필요`하다.
- 특히 DPO 10개 중 37, 39는 제외하는 편이 안전하다.
- SFT 30개 중 5는 오탈자 때문에 제외하거나 후순위로 내리는 편이 좋다.
- 대체 DPO 후보는 충분하다. 전체 clean DPO first-solve 후보는 59개이고, 간단한 위험 휴리스틱을 통과한 후보도 56개였다.
