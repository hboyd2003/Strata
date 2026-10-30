# `serve/request-transcripts`

Read a recent API request as a chat in the Monitor. Own branch on upstream.

## What it does

With the API request monitor on (`"api_monitor": true`), each request's record also keeps its messages as the template
saw them (role, text, a count of pictures, reasoning, tool calls) and the reply the engine wrote, for every API,
streamed or not. A turn shares the messages it has in common with the turn before, so an agent's history is kept about
once, not once per request. Texts over 64K characters are cut, and past 2M characters the oldest messages are dropped.

Each row of the Monitor's request table opens a dialog: the chat on the left (drawn by the chat's own code: thinking,
tool calls, Markdown, a folded system prompt), and on the right the request's settings, prompt (tokens, reused, read,
prefill speed), reply (tokens, decode speed, MTP drafts, VRAM hit rate) and times.

**Privacy:** with `api_monitor` on, the server holds the content of recent requests in RAM, and anyone who can open the
Monitor can read them. It is off by default.
