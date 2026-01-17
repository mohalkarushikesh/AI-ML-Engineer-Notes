## 📌 Beautiful Soup Cheat Sheet

### 🔹 Installation
```bash
pip install beautifulsoup4
```

---

### 🔹 Import & Create Soup
```python
from bs4 import BeautifulSoup

html_doc = "<html><body><p>Hello World</p></body></html>"
soup = BeautifulSoup(html_doc, "html.parser")
```

---

### 🔹 Parsing from File or URL
```python
# From file
with open("index.html") as f:
    soup = BeautifulSoup(f, "html.parser")

# From requests
import requests
url = "https://example.com"
page = requests.get(url)
soup = BeautifulSoup(page.content, "html.parser")
```

---

### 🔹 Navigating the Tree
```python
soup.title          # <title> tag
soup.title.string   # text inside <title>
soup.p              # first <p> tag
soup.find_all("p")  # all <p> tags
```

---

### 🔹 Searching
```python
# Find by tag
soup.find("a")

# Find by attribute
soup.find("a", {"class": "link"})

# CSS selectors
soup.select("div.content > p")
```

---

### 🔹 Modifying
```python
tag = soup.p
tag['class'] = "new-class"   # add attribute
tag.string = "Updated text"  # change text
```

---

### 🔹 Extracting Data
```python
links = [a['href'] for a in soup.find_all("a", href=True)]
texts = [p.get_text() for p in soup.find_all("p")]
```

---

### 🔹 Common Use Cases
- **Web scraping**: Extract product info, articles, links.  
- **Data cleaning**: Parse messy HTML/XML.  
- **Automation**: Build crawlers or bots.  

---

## ⚠️ Risks & Considerations
- **Legal/Ethical**: Always check a site’s `robots.txt` and terms of service before scraping.  
- **Performance**: Beautiful Soup is easy but slower than `lxml` for huge documents.  
- **Blocking**: Websites may block scraping — use headers, delays, or proxies responsibly.  

---

## 📚 Sources
- [BeautifulSoup Cheat Sheet – DEV Community](https://dev.to/shahidkhans/beautifulsoup-cheat-sheet-python-5ep3)  
- [ScrapingAnt BeautifulSoup Cheatsheet](https://scrapingant.com/blog/beautifulsoup-cheatsheet)  
- [ProxiesAPI Complete BeautifulSoup Cheatsheet](https://proxiesapi.com/articles/the-complete-beautifulsoup-cheatsheet-with-examples)

---
