# Org-roam RAG

Org 노트 검색 및 질의응답 기능을 개발하는 프로젝트입니다.
기존 LangChain 강의 실습과 문서 검색 봇 코드는 `examples/`에 보관합니다.

- `examples/main.py`: 문서 검색 봇의 Streamlit UI
- `examples/backend/core.py`: 문서 검색 및 답변 생성
- `examples/ingestion.py`: LangChain 웹 문서 수집 및 색인
- `examples/logger.py`: 실습용 로그 출력
- `examples/2_agent_loop_raw_functino_calling.py`: 도구 호출 실습
- `examples/mediumblog1.txt`: 이전 실습용 샘플 데이터
- `examples/src/`: 기존 패키지 골격

실습 실행 방법은 [examples/README.md](examples/README.md)를 참고하세요.
