---
tags:
  - building-mcp-track-creative
---

# Math MCP Server

## The Ultimate Symbolic Mathematics Engine via MCP

**Track 1: Building MCP – Creative Category**  
**Tag:** `building-mcp-track-creative`

A full-featured, production-ready **Model Context Protocol (MCP)** server that turns any LLM into a powerful symbolic mathematics assistant.  
Solve equations · Integrate · Differentiate · Compute limits · Plot functions · Matrix algebra · LaTeX output · Batch processing — all with **zero external APIs**.

Powered by **SymPy · NumPy · Matplotlib**

---

## Features

| Feature                     | Description                                                                 | Returns                     |
|-----------------------------|-----------------------------------------------------------------------------|-----------------------------|
| Solve equations             | Symbolic + numeric roots of polynomials & systems                           | Exact + float solutions     |
| Integrate (definite/indefinite) | Full symbolic integration, including improper integrals                | Closed-form + numeric       |
| Differentiate               | Any order, partial derivatives                                              | Symbolic result             |
| Limits                      | One-sided, ∞, L’Hôpital automatically                                      | Exact limit                 |
| Plot functions              | High-quality PNG plots returned as base64 (ImageContent)                    | Beautiful graph             |
| Simplify & evaluate         | Full simplification + numerical evaluation with substitutions              | Clean expression + number   |
| Matrix operations           | Inverse, det, transpose, eigenvalues, multiplication                       | Matrix result               |
| LaTeX output                | Perfect typesetting-ready LaTeX                                             | Ready for papers            |
| Batch processing            | Run dozens of operations in a single call                                   | Bulk JSON results           |

**100 % offline · No API keys · CPU-only · Works with Claude, Cursor, Grok, etc.**

---

## Demo Video & Social Proof (REQUIRED)

- **Demo Video (60 sec):** https://youtu.be/9LHFtBjZYno
- **X/Twitter Post:** https://x.com/AdilAijaz16/status/1995075588156228020

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the beautiful Gradio UI
python app.py
# → opens at http://localhost:7860

# 3. OR run the pure MCP server (for Claude Desktop, Cursor, etc.)
python mcp_server.py
# → MCP endpoint: http://127.0.0.1:8000