
# MCP JIRA SETUP

## How to get your Jira API token
1. Go to [Atlassian API Tokens](https://id.atlassian.com/manage-profile/security/api-tokens)  
2. Create a new API Token  

---

## Setup Options

### Option 1: Use Rovo
- Configure Rovo with your Jira API token.  
- Follow Rovo’s documentation for integration.  

---

### Option 2: Manual Setup

#### Clone Repositories
```bash
git clone https://github.com/kompallik/ATLASSIAN-MCP
git clone https://github.com/sooperset/mcp-atlassian
git clone https://github.com/modelcontextprotocol/servers
```

#### Install Dependencies
```bash
npm install @modelcontextprotocol/sdk zod
```

> Note: If you see  
> `No matching version found for @modelcontextprotocol/sdk@^0.1.0`  
> install the latest available version manually as shown above.

#### Start the Server
```bash
npm start
```

---

## Notes
- Ensure Node.js and npm are installed and updated.  
- Keep your Jira API token secure — treat it like a password.  
- If using Docker, you can containerize the MCP server for easier deployment.  
