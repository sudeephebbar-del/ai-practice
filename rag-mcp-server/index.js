#!/usr/bin/env node
// MCP Server: bridges VS Code / Cursor ↔ your FastAPI RAG service
// Cursor spawns this as a child process and communicates via stdin/stdout.

const { McpServer } = require("@modelcontextprotocol/sdk/server/mcp.js");
const { StdioServerTransport } = require("@modelcontextprotocol/sdk/server/stdio.js");
const { z } = require("zod");

const RAG_URL = process.env.RAG_BASE_URL || "http://localhost:8002";

const server = new McpServer(
  { name: "bss-rag", version: "1.0.0" },
  { capabilities: { tools: {} } },
);

server.registerTool(
  "search_bss_docs",
  {
    description:
      "Search the BSS platform knowledge base (architecture docs, runbooks, ADRs). " +
      "Use when asked about Oracle, Kafka, CDR processing, billing, API design, " +
      "PL/SQL patterns, or any platform-specific technical question.",
    inputSchema: {
      question: z.string().describe("The technical question to search for"),
    },
  },
  async ({ question }) => {
    try {
      const resp = await fetch(`${RAG_URL}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      if (!resp.ok) {
        throw new Error(`RAG service error: ${resp.status} ${resp.statusText}`);
      }

      const data = await resp.json();
      const sources = (data.sources || [])
        .map((s) => `- ${s.title} (page ${s.page})`)
        .join("\n");

      return {
        content: [
          {
            type: "text",
            text: `${data.answer}\n\n**Sources:**\n${sources}`,
          },
        ],
      };
    } catch (err) {
      return {
        isError: true,
        content: [
          { type: "text", text: `Error calling RAG service: ${err.message}` },
        ],
      };
    }
  },
);

(async () => {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  process.stderr.write(`BSS RAG MCP server ready (RAG_BASE_URL=${RAG_URL})\n`);
})().catch((err) => {
  process.stderr.write(`MCP server failed to start: ${err.message}\n`);
  process.exit(1);
});
