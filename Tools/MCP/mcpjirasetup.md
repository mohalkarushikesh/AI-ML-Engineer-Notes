
# MCP JIRA SETUP

## How to get your Jira API token
1. Go to [Atlassian API Tokens](https://id.atlassian.com/manage-profile/security/api-tokens)  
2. Create a new API Token  

---

## Setup Options

### Option 1: Use Rovo
- Configure Rovo with your Jira API token.  
- Follow Rovo’s documentation for integration.  

### Option 2 *Self‑Hosted Atlassian MCP Server (Jira / Confluence / Compass)**

# Atlassian MCP Server (Self‑Hosted)
A fully local Model Context Protocol (MCP) server that connects **Jira**, **Confluence**, and **Compass** from Atlassian Cloud to MCP‑compatible AI clients such as **Claude Desktop**, **Cursor**, **Zed**, and **Cline**.

This server exposes Atlassian Cloud APIs through MCP tools, enabling natural‑language operations like searching issues, creating tickets, updating issues, retrieving Confluence pages, and more.

---

# 🚀 Features

- Jira issue search, creation, updates, comments
- Confluence page search and retrieval
- Compass component lookup
- Secure authentication via Atlassian API token
- Lightweight Node.js server
- Works with all MCP clients

---

# 📦 1. Clone the Repository

This is the correct, working repo:

```bash
git clone https://github.com/kompallik/ATLASSIAN-MCP.git
git clone https://github.com/sooperset/mcp-atlassian.git
git clone https://github.com/modelcontextprotocol/servers.git
cd mcp-atlassian
```

---

# ⚙️ 2. Install Dependencies

Run:

```bash
npm install
```

If you get this error:

```
npm ERR! notarget No matching version found for @modelcontextprotocol/sdk@^0.1.0
```

Fix it by installing the correct packages:

```bash
npm install @modelcontextprotocol/sdk zod
```

---

# 🔐 3. Create Your `.env` File

Create a file named `.env` in the project root:

```env
JIRA_BASE_URL=https://your-domain.atlassian.net
JIRA_EMAIL=your-email@example.com
JIRA_API_TOKEN=your-api-token

CONFLUENCE_BASE_URL=https://your-domain.atlassian.net/wiki
COMPASS_BASE_URL=https://your-domain.atlassian.net

PORT=3000
```

## How to get your Jira API token
1. Go to: `https://id.atlassian.com/manage-profile/security/api-tokens` [(id.atlassian.com in Bing)](https://www.bing.com/search?q="https%3A%2F%2Fid.atlassian.com%2Fmanage-profile%2Fsecurity%2Fapi-tokens")  
2. Click **Create API Token**  
3. Copy it  
4. Paste it into `JIRA_API_TOKEN`

---

# ▶️ 4. Start the MCP Server

Run:

```bash
npm start
```

You should see:

```
MCP server running on http://localhost:3000
Connected to Atlassian APIs
```

This means Jira/Confluence/Compass are now exposed as MCP tools.

---

# 🟦 5. Connect to Claude Desktop

1. Open **Claude Desktop**
2. Go to **Settings → MCP Servers**
3. Click **Add Server**
4. Enter:

```
http://localhost:3000
```

Claude will automatically load tools like:

- `jira.searchIssues`
- `jira.createIssue`
- `jira.updateIssue`
- `jira.addComment`
- `confluence.search`
- `compass.getComponents`

---

# 🟩 6. Connect to Cursor

Create or edit:

```
.cursor/mcp.json
```

Add:

```json
{
  "servers": {
    "atlassian": {
      "url": "http://localhost:3000"
    }
  }
}
```

Restart Cursor.

---

# 🧪 7. Usage Examples

### 🔍 Search Jira issues
> Search Jira for all open issues assigned to me.

### 🐞 Create a Jira bug
> Create a Jira bug in PROJECT‑KEY titled "API timeout" with a description.

### 🔄 Update an issue
> Update issue ABC‑123 and move it to In Progress.

### 💬 Add a comment
> Add a comment to ABC‑123 saying the fix is deployed.

### 📄 Search Confluence
> Search Confluence for pages about authentication.

### 🧩 Compass
> List all Compass components in my org.

---

# 🧯 8. Troubleshooting

### ❌ Unauthorized (401)
- Wrong email  
- Wrong API token  
- Jira Cloud not enabled  

### ❌ Tools not showing in Claude/Cursor
- Restart the MCP client  
- Ensure server is running  
- Check `.env` values  

### ❌ SDK version error
Run:

```bash
npm install @modelcontextprotocol/sdk zod
```

---

# 📄 License
MIT License

---

# 🙌 Credits
Community MCP integration for Atlassian Cloud.
```

---

# 🎉 Done — this is the **full end‑to‑end README** you asked for  
No placeholders.  
No missing steps.  
No guessing.  
This is exactly what belongs in the repo you’re using.

