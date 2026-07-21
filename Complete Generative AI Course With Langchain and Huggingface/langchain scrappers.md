# LangChain Web Scraping Tools

LangChain does not have its own built-in web scraping engine. Instead, it provides a unified **Document Loader** framework to fetch, convert, and parse web pages. These loaders can act as scrapers themselves or integrate with third-party web scraping APIs.

The primary web scraping utilities and integrations available in LangChain fall into three main categories.

---

## 1. Built-in Parsers & Loaders (Standard)

These are native LangChain modules that parse raw HTML directly from URLs.

- **`WebBaseLoader`** — The baseline standard. Uses Beautiful Soup under the hood to scrape text from static, non-JavaScript websites (often used for RAG pipelines).
- **`AsyncHtmlLoader`** — An asynchronous scraper for loading multiple pages concurrently. Commonly paired with the `Html2TextTransformer` to strip HTML tags and convert pages into clean, readable text.

---

## 2. Headless Browser Integrations

These modules simulate a real browser to scrape Single Page Applications (SPAs) and JavaScript-heavy websites.

- **`PlaywrightURLLoader` & `SeleniumURLLoader`** — Execute JavaScript in the background before extracting the text. Ideal for bypassing dynamic content-loading issues.
- **`UnstructuredURLLoader`** — Integrates with the Unstructured.io library to handle complex website layouts and parse out tables, lists, and headers.

---

## 3. Third-Party API Integrations

LangChain provides official toolkits and wrapper integrations for enterprise web scraping APIs. These tools handle proxy rotation, anti-bot/CAPTCHA bypassing, and JavaScript rendering at scale.

- **Bright Data (`langchain-brightdata`)** — Exposes classes like `BrightDataWebScraperAPI` and `BrightDataUnblocker`.
- **ScrapeGraph AI** — Provides AI-driven scraping tools (like `SmartCrawlerTool`) that extract data using natural language prompts rather than brittle CSS selectors.
- **Hyperbrowser (`langchain-hyperbrowser`)** — Cloud-based crawl, scrape, and extraction tools capable of autonomous, scalable web navigation.
- **Scrapeless** — A universal scraping API integration designed for modern dynamic websites and Google SERP/Trends extraction.
- **ScraperAPI (`langchain-scraperapi`)** — Wraps ScraperAPI so AI agents can scrape public websites without worrying about blocks or proxies.

---

## Quick Reference

| Category | Tool | Best For |
| --- | --- | --- |
| Built-in | `WebBaseLoader` | Static, non-JS sites; RAG pipelines |
| Built-in | `AsyncHtmlLoader` + `Html2TextTransformer` | Bulk async loading, clean text output |
| Headless | `PlaywrightURLLoader` / `SeleniumURLLoader` | JavaScript-heavy pages, SPAs |
| Headless | `UnstructuredURLLoader` | Complex layouts, tables, lists, headers |
| Third-party API | Bright Data, ScrapeGraph AI, Hyperbrowser, Scrapeless, ScraperAPI | Proxy rotation, anti-bot bypass, scale |
