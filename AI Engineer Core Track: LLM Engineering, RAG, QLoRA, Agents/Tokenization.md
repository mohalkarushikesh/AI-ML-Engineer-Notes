## 🧩 What Tokenization Means
- **In Natural Language Processing (NLP):**
  - Splitting text into smaller units (tokens).
  - Tokens can be words, subwords, or characters depending on the tokenizer.
  - Example: `"I love AI!"` → `["I", "love", "AI", "!"]`.

- **In Programming/Compilers:**
  - Breaking source code into meaningful symbols (keywords, identifiers, operators).
  - Example: `int x = 10;` → `["int", "x", "=", "10", ";"]`.

- **In Blockchain/Crypto:**
  - Representing assets as digital tokens on a blockchain.
  - Example: Tokenizing real estate → each token represents fractional ownership.

---

## ✍️ Key Notes for NLP Tokenization
- **Whitespace tokenization**: Simple split by spaces; fast but crude.
- **Rule-based tokenization**: Uses regex/patterns for punctuation, contractions.
- **Subword tokenization (BPE, WordPiece, SentencePiece)**:
  - Handles rare words by splitting into smaller units.
  - Example: `"unhappiness"` → `["un", "happiness"]`.
- **Character-level tokenization**: Useful for languages with complex morphology.
- **Impact on models**: Vocabulary size, efficiency, and handling of unknown words depend on tokenizer choice.

---

## 📊 Tokenization Trade-offs
| Approach              | Pros | Cons |
|------------------------|------|------|
| Word-level             | Simple, intuitive | Large vocab, OOV issues |
| Subword-level          | Efficient, handles rare words | Less human-readable |
| Character-level        | No OOV, flexible | Long sequences, slower |
| Sentence-level         | Good for summarization | Too coarse for fine tasks |

---

## 🔑 Quick Reminders
- Tokenization is **language-dependent** (Chinese vs English vs Arabic).
- Punctuation, emojis, and special symbols need careful handling.
- In ML models, tokenization directly affects **embedding quality** and **training efficiency**.
- In crypto, tokenization requires **legal frameworks** and **smart contracts**.

---

## 🧩 Example Text
`"I can't believe it's already 2026!"`

---

## 🔠 Tokenization Methods & Examples

- **Whitespace Tokenization**  
  Splits only on spaces.  
  → `["I", "can't", "believe", "it's", "already", "2026!"]`

- **Rule-based / Regex Tokenization**  
  Handles punctuation and contractions more carefully.  
  → `["I", "ca", "n't", "believe", "it", "'s", "already", "2026", "!"]`

- **Word-level Tokenization**  
  Each word is a token, punctuation may be separate depending on rules.  
  → `["I", "can't", "believe", "it's", "already", "2026", "!"]`

- **Subword Tokenization (BPE, WordPiece, SentencePiece)**  
  Breaks rare or complex words into smaller units.  
  Example with WordPiece:  
  → `["I", "can", "'", "t", "believe", "it", "'", "s", "already", "202", "6", "!"]`

- **Character-level Tokenization**  
  Every character is a token.  
  → `["I", " ", "c", "a", "n", "'", "t", " ", "b", "e", "l", "i", "e", "v", "e", " ", "i", "t", "'", "s", " ", "a", "l", "r", "e", "a", "d", "y", " ", "2", "0", "2", "6", "!"]`

- **Sentence-level Tokenization**  
  Splits text into sentences (useful for longer passages).  
  Example with two sentences:  
  `"I can't believe it's already 2026! Time flies."`  
  → `["I can't believe it's already 2026!", "Time flies."]`

---

## 📊 Quick Comparison

| Method              | Example Output (shortened) |
|---------------------|-----------------------------|
| Whitespace          | ["I", "can't", "believe"] |
| Rule-based/Regex    | ["I", "ca", "n't", "believe"] |
| Word-level          | ["I", "can't", "believe"] |
| Subword (WordPiece) | ["I", "can", "'", "t", "believe"] |
| Character-level     | ["I", " ", "c", "a", "n", "'"] |
| Sentence-level      | ["I can't believe it's already 2026!"] |

---
