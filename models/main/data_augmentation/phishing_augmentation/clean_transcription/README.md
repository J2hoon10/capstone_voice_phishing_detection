# Clean Transcription (Audio-First)

오디오를 OpenAI 전사 모델로 다시 전사해서 clean few-shot을 만들기 위한 폴더입니다.

## 파일

- `transcribe_one_sample.py`: 오디오 1개 샘플 전사 스모크 테스트
- `build_clean_fewshot_from_audio.py`: 카테고리별 clean few-shot CSV 생성

## 1샘플 테스트

```powershell
$env:OPENAI_API_KEY="sk-..."
C:\Users\myhom\anaconda3\Scripts\conda.exe run --no-capture-output -n capstone python models/classifier/preprocessing/augmented_pipeline/clean_transcription/transcribe_one_sample.py
```

`--dry-run` 옵션으로 API 호출 없이 샘플 파일 선택만 확인할 수 있습니다.

## clean few-shot 생성

```powershell
$env:OPENAI_API_KEY="sk-..."
C:\Users\myhom\anaconda3\Scripts\conda.exe run --no-capture-output -n capstone python models/classifier/preprocessing/augmented_pipeline/clean_transcription/build_clean_fewshot_from_audio.py --per-category 30
```
