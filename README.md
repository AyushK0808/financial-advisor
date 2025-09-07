# Financial LLM Assistant

A fine-tuned Large Language Model specialized in providing financial and stock market advice based on real-time market trends through a Retrieval-Augmented Generation (RAG) architecture. The system combines local AI processing with live market data to deliver personalized financial insights while maintaining complete privacy.

## Features

- **Fine-tuned Financial LLM**: Custom-trained DialoGPT-small optimized for financial advisory
- **Real-time Market Analysis**: Integration with live market data and trends
- **RAG Architecture**: Enhanced responses using relevant financial knowledge retrieval
- **Privacy-First**: All AI processing happens locally on your machine
- **Dual Interface**: Both command-line and web-based Streamlit interfaces
- **Containerized**: Docker support for easy deployment

## Tech Stack

-  **Local LLM:** DialoGPT-small with financial fine-tuning
-  **Web Search:** SearXNG for real-time market trends
-  **Database:** MongoDB for data persistence
-  **Knowledge:** ChromaDB vector database
-  **Market Data:** Yahoo Finance integration
-  **Privacy:** All processing happens locally

## Prerequisites

- Python 3.8+
- Docker & Docker Compose
- Make utility
- 8GB+ RAM recommended for optimal LLM performance

## Quick Start

### 1. Initial Setup
```bash
make setup
```
This will install all dependencies and prepare the environment.

### 2. Start Services
```bash
make docker-up
```
Launches all required services (MongoDB, ChromaDB, SearXNG) in Docker containers.

### 3. Train the Model (Optional)
```bash
make train
```
Fine-tune the LLM on financial data. This step is optional if you want to use pre-trained weights.

### 4. Choose Your Interface

#### Command Line Interface
```bash
make cli
```

#### Web Interface (Streamlit)
```bash
make run
```
Access the web interface at `http://localhost:8501`

## Available Commands

Run `make help` to see all available commands:

```bash
make help
```

### Core Commands

| Command | Description |
|---------|-------------|
| `make setup` | Install dependencies and initialize the environment |
| `make docker-up` | Start all Docker services |
| `make docker-down` | Stop all Docker services |
| `make train` | Fine-tune the LLM on financial data |
| `make cli` | Launch the command-line interface |
| `make run` | Start the Streamlit web application |
| `make clean` | Clean up temporary files and caches |

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Query    │───▶│   RAG System    │───▶│  Fine-tuned LLM │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ChromaDB      │◄───│  Knowledge Base │───▶│   Market Data   │
│ (Vector Store)  │    │   Retrieval     │    │ (Yahoo Finance) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐
│    MongoDB      │◄───│    SearXNG      │
│ (Persistence)   │    │ (Web Search)    │
└─────────────────┘    └─────────────────┘
```

## Usage Examples

### CLI Interface
```bash
$ make cli
> What are the current trends in the tech sector?
> Should I invest in renewable energy stocks?
> Analyze the performance of AAPL over the last quarter
```

### Streamlit Web Interface
1. Run `make run`
2. Open `http://localhost:8501` in your browser
3. Enter your financial questions in the chat interface
4. Get real-time analysis and recommendations

## Data Sources

The system pulls data from multiple reliable sources:

- **Yahoo Finance**: Real-time stock prices, historical data
- **SearXNG**: Market news and sentiment analysis  
- **Financial databases**: Economic indicators, company fundamentals
- **Custom datasets**: Fine-tuning data for domain adaptation

## Privacy & Security

- **Local Processing**: All LLM inference happens on your machine
- **No Data Sharing**: Your queries and data never leave your system
- **Secure APIs**: Encrypted connections to external data sources
- **Containerized**: Isolated execution environment

## Disclaimer

**This tool is for educational and informational purposes only. It does not constitute financial advice, investment recommendations, or professional counsel. Always consult with qualified financial advisors before making investment decisions. Past performance does not guarantee future results.**

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and commit: `git commit -am 'Add feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
