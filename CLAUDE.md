# Project: Academic Paper Survey Generation Harness (学术论文综述生成 Harness)

## Context

Hackathon project for 上海人工智能实验室 2026 暑期夏令营. Task 4 (团队挑战题, 3人团队).

**Topic**: 世界模型 (World Models) — 生成一篇图文并茂的综述 (HTML/PDF).

**Core constraint**: Harness 的核心大模型调用必须使用 Intern-S2-Preview API. 评审时主办方会替换 API Key 重新运行.

## Key Requirements

- No hallucinated citations, no fact-less descriptions
- Output must be rich with figures and text (图文并茂)
- Must support `API_BASE_URL` and `API_KEY` via environment variables
- Code must be original (semantic plagiarism check will be performed)

## Available Resources

- **Intern-S2-Preview API**: https://chat.intern-ai.org.cn
- **MinerU 文档解析 API**: https://mineru.net/ (for parsing academic PDFs)
- **Sciverse 科学文献库 API**: https://sciverse.space/ (for paper search/retrieval)

## Architecture

```
User Query ("世界模型 综述")
    ↓
Agent Loop (Intern-S2-Preview as core LLM)
    ↓
Tool Calls:
  - Sciverse API → search & retrieve papers
  - MinerU API → parse PDF content
  - File I/O → read/write drafts, manage references
  - Web search → supplementary info (optional)
    ↓
Iterative: summarize → classify → organize → write → verify citations
    ↓
Output: HTML survey with figures
```

### Agent Loop Design

1. **Task Decomposition**: Break "world models survey" into sub-tasks (literature search, categorization, timeline, future directions)
2. **Tool Orchestration**: Call Sciverse to find papers → MinerU to parse → LLM to summarize/classify
3. **Citation Verification**: Cross-check every citation against actual paper metadata from Sciverse
4. **Iterative Refinement**: Review drafts, check for hallucinations, regenerate if needed
5. **Output Generation**: Render final survey as HTML with embedded figures

### Anti-Hallucination Strategy

- All paper metadata (title, authors, year, venue) must come from Sciverse API results, never LLM-generated
- Every claim must be traced back to a specific paper reference
- Verification pass: extract all citations from draft → validate each against source
- Figures must come from parsed papers (via MinerU) or be generated from verified content

## Deliverables Checklist

- [ ] Working Harness prototype (source code + run instructions)
- [ ] API config via environment variables (`INTERN_API_BASE_URL`, `INTERN_API_KEY`)
- [ ] Output: world models survey (HTML/PDF, 图文并茂)
- [ ] Intern-S2-Preview capability analysis table
- [ ] Presentation slides

## Bonus Features (prioritized)

1. **Memory system**: persist paper metadata across sessions to avoid re-searching
2. **Skill system**: packaged workflow for "generate survey from topic X"
3. **Sub-agent parallelism**: concurrent paper parsing and summarization
4. **Context compression**: handle long paper collections within model context limits

## Language

- Code and comments: English
- Comments and explanations to user: 中文
- Commit messages: English
- Survey output: 中文 (academic style)

## Preferences

- Fail fast, let exceptions bubble up
- Only add error handling at system boundaries (API calls, file I/O)
- Minimal abstraction — prefer flat, readable code over over-engineering
- Edit existing files over creating new ones
