# Checkpoints

이 디렉터리는 공개 배포 후보의 tokenizer, config, generation config, export metadata를 제공합니다.

현재 대용량 모델 가중치는 포함하지 않았습니다. HF `model.safetensors`, llama.cpp F16 GGUF, Q4_K_M GGUF는 용량과 공개 범위를 확인한 뒤 Git LFS 또는 별도 릴리스 자산으로 배포합니다.

## 포함된 파일

| 파일 | 내용 |
| --- | --- |
| `config.json` | HF 형식 모델 설정 |
| `generation_config.json` | 생성 설정 |
| `tokenizer.model` | SentencePiece tokenizer |
| `tokenizer.vocab` | tokenizer vocabulary |
| `tokenizer.manifest.json` | tokenizer 생성 메타데이터 |
| `export_manifest.json` | export bundle 생성 메타데이터 |

## 추가할 파일

- HF `model.safetensors`
- llama.cpp F16 GGUF
- llama.cpp Q4_K_M GGUF
- 모델 파일 checksum
- 다운로드 위치
- 추론 예시
