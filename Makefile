# Ultra-Simple Makefile for Financial Advisory System (Windows friendly)

.PHONY: help setup install run cli train docker-up docker-down clean

help:
	@echo Financial Advisory System
	@echo ========================
	@echo.
	@echo Commands:
	@echo   setup         - Complete setup
	@echo   install       - Install dependencies
	@echo   run           - Run Streamlit app
	@echo   cli           - Run CLI
	@echo   train         - Train model
	@echo   docker-up     - Start services
	@echo   docker-down   - Stop services
	@echo   clean         - Clean temp files

setup: install
	@echo Setting up...
	@if exist .env.example ( copy /Y .env.example .env >nul )
	@echo ✅ Setup complete! Edit .env if needed

install:
	@echo Installing dependencies...
	@pip install -r requirements.txt
	@python -c "import nltk; nltk.download('vader_lexicon')" 2>nul || true

run:
	@echo Starting Streamlit app...
	@streamlit run src/streamlit_app.py

cli:
	@echo Starting CLI...
	@python -m cli.main interactive

train:
	@echo Training model...
	@python -m src.llm_model --train

docker-up:
	@echo Starting services...
	@docker-compose up -d

docker-down:
	@echo Stopping services...
	@docker-compose down

clean:
	@echo Cleaning...
	@for /R %%f in (*.pyc) do del /Q %%f
	@for /R %%d in (__pycache__) do rmdir /S /Q %%d
	@if exist .pytest_cache rmdir /S /Q .pytest_cache
