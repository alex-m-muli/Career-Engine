# Career Engine: AI-Assisted Job Search Pipeline

Career Engine is a semi-autonomous, Python-based job application pipeline designed for senior software engineers. It treats the job search as a high-value pipeline, bypassing mass auto-apply bots in favor of precision tailoring.

## How it works
1. **Scrape:** Aggregates recent job postings across LinkedIn, Indeed, Bayt, and Glassdoor using `JobSpy`, with built-in anti-scam filtering.
2. **Score:** Uses Gemini 3.8 Flash and a custom 3-tier skill taxonomy to evaluate fit (0-10) and identify skill gaps.
3. **Tailor (DIS Architecture):** Uses a Multi-Pass Decoupled Synthesis architecture to generate tailored resume bullets. It strictly enforces an "Anti-AI Protocol" (no banned buzzwords, no rule-of-three triplets, bursty cadence) to ensure the output reads 100% human-written.
4. **Render:** Compiles the resume into a pixel-perfect, ATS-parseable vector PDF using `Typst`.
5. **Autofill:** Uses `Playwright` to attach to a real Chrome profile, pre-fills application forms (using a local `question_bank.json`), uploads the PDF, and pauses for human review before submission.
6. **Track:** Logs all applications to a local SQLite database to prevent duplicate submissions.

## Tech Stack
- **LLMs:** Gemini 3.8 Flash (primary), Groq `openai/gpt-oss-120b` (fast fallback).
- **PDF Engine:** Typst (vector PDF).
- **Browser Automation:** Playwright (headed mode, persistent profile).
- **Data:** Pydantic v2, SQLite, JSON.

## Setup & Usage
**