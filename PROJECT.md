# Whisper Chunk Transcribe

> **Note:** This is a project overview card. For technical documentation and setup instructions, see [README.md](README.md).

## Overview

A comprehensive experimental framework for audio transcription research using OpenAI's Whisper model, featuring automated experimentation, database-backed result tracking, and advanced audio preprocessing. Built to systematically evaluate transcription accuracy across different prompting strategies and audio processing techniques. Implements production-grade patterns including async task orchestration, connection pooling, and SSH-secured database access.

## Quick Reference

| Field | Value |
|-------|-------|
| **Status** | Active |
| **Deployed URL** | Not publicly deployed |
| **Build Time** | 3 months (Jul 2024 - Oct 2024) |

## Technical Stack

| Category | Technologies |
|----------|--------------|
| **Languages** | Python 3.10+, SQL (PostgreSQL) |
| **Frameworks** | PyTorch, Whisper (OpenAI), WhisperX, Faster-Whisper, PyAnnote Audio |
| **Infrastructure** | PostgreSQL, SSH Tunnel, Docker |
| **AI/ML** | Whisper ASR models, Voice Activity Detection (VAD), Audio segmentation |
| **Key Patterns** | Async/await, Connection pooling, Singleton pattern, Queue-based task management, ThreadPoolExecutor |

## Key Achievements

- Built modular experimentation framework supporting automated testing of multiple transcription strategies across large audio datasets
- Implemented sophisticated async task orchestration using asyncio.Queue and ThreadPoolExecutor for concurrent audio processing and transcription
- Designed comprehensive PostgreSQL schema with 12+ tables, soft-delete propagation, and automated triggers for experiment result tracking
- Developed dynamic prompt generation system that recursively builds context-aware prompts from previous transcriptions within token limits
- Created audio preprocessing pipeline with VAD-based segmentation, noise reduction (Wiener filter, non-stationary noise reduction), and audio normalization

## Technical Highlights

- **Async Task Management**: Multi-worker async architecture using asyncio.Queue for segment processing and ThreadPoolExecutor for offloading CPU-bound transcription tasks, preventing event loop blocking while maintaining concurrent processing efficiency
- **Database Architecture**: Production-ready PostgreSQL schema with cascading soft-delete functions, automatic timestamp triggers, comprehensive indexes, and foreign key constraints supporting complex experiment relationships and data integrity
- **Dynamic Prompt Engineering**: Recursive prompt-building algorithm that maximizes context usage by incrementally adding words from previous transcriptions while respecting Whisper's 255-token limit, using Hugging Face tokenizer for accurate token counting

## Code Metrics

| Metric | Value |
|--------|-------|
| **Lines of Code** | 5,333 (4,279 Python, 1,054 SQL) |
| **Primary Language** | Python 3.10+ (~80%) |
| **Test Coverage** | Unit + integration tests with pytest (5 test files) |
| **Key Dependencies** | whisperx, faster-whisper, pyannote-audio, torch, psycopg2-binary, asyncio, pandas, noisereduce, pydub |

---

*For detailed technical documentation, setup instructions, and contribution guidelines, please see [README.md](README.md).*
