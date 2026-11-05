# LangChain Examples

기존 강의 실습과 LangChain 문서 검색 봇을 보관합니다.
아래 명령은 저장소 루트에서 실행합니다.

## 문서 검색 봇

1. `uv sync`로 의존성을 설치합니다.
2. 루트의 `.env`에 `OPENAI_API_KEY`를 설정합니다.
3. `uv run python examples/ingestion.py`로 LangChain 웹 문서를 수집하고 색인합니다.
4. `uv run streamlit run examples/main.py`로 검색 UI를 실행합니다.

수집 및 검색 코드는 루트의 `chroma_db/`를 사용합니다.
색인 실행 시 웹 문서를 내려받으며 임베딩 API 비용이 발생할 수 있습니다.

## 기타 실습

- `2_agent_loop_raw_functino_calling.py`: Ollama 모델을 사용하는 도구 호출 실습
- `mediumblog1.txt`: 이전 텍스트 문서 실습용 샘플
- `src/`: 이전 프로젝트의 패키지 골격
