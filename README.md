# Org-roam RAG

Org 노트 검색 및 질의응답 기능을 개발하는 프로젝트입니다.
기존 LangChain 강의 실습과 문서 검색 봇 코드는 `examples/`에 보관합니다.

## Org 노트 색인

1. `uv sync`로 의존성을 설치합니다.
2. `.env`에 `OPENAI_API_KEY`와 `ORG_ROAM_DIR`을 설정합니다.
3. `uv run main` 또는 `uv run python __main__.py`로 `.org` 노트를 Chroma에 색인합니다.

색인은 `chroma_db/`의 `org_roam` 컬렉션에 저장됩니다. 재실행하면 변경된 청크만 임베딩하고 삭제된 노트의 청크는 제거합니다. 새 청크의 임베딩에는 OpenAI API 비용이 발생합니다.

- `examples/main.py`: 문서 검색 봇의 Streamlit UI
- `examples/backend/core.py`: 문서 검색 및 답변 생성
- `examples/ingestion.py`: LangChain 웹 문서 수집 및 색인
- `examples/logger.py`: 실습용 로그 출력
- `examples/2_agent_loop_raw_functino_calling.py`: 도구 호출 실습
- `examples/mediumblog1.txt`: 이전 실습용 샘플 데이터
- `examples/src/`: 기존 패키지 골격

실습 실행 방법은 [examples/README.md](examples/README.md)를 참고하세요.
