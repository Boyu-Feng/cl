"""Run the unmodified GenericAgent task lifecycle inside its isolated workspace."""
import json
from pathlib import Path
import sys
import threading


def delivered_report(outputs, final, normal_exit):
    if not normal_exit:
        return ""
    for text in outputs:
        if '"transmitters"' in text:
            final = text
    return final


def main():
    protocol = sys.stdout
    sys.stdout = sys.stderr  # Keep upstream diagnostics off the JSON transport.
    sys.path.insert(0, "/agent")
    from agentmain import GenericAgent
    from plugins.hooks import register

    agent = GenericAgent()
    agent.verbose = False
    events = []
    endings = []
    assistant_replies = []
    report_outputs = []
    tool_results = []

    @register("llm_after")
    def trace_reply(ctx):
        response = ctx["response"]
        assistant_replies.append({"content": response.content,
                                  "has_tools": bool(response.tool_calls)})
        if not response.tool_calls:
            report_outputs.append(response.content)

    @register("tool_after")
    def trace_result(ctx):
        data = getattr(ctx.get("ret"), "data", None)
        tool_results.append({"tool": ctx["tool_name"], "data": data})
        if ctx["tool_name"] == "code_run" and isinstance(data, dict):
            if data.get("status") == "success" and data.get("exit_code") == 0:
                report_outputs.append(data.get("stdout", ""))
        if ctx["tool_name"] in ("file_write", "file_patch") and isinstance(data, dict):
            if data.get("status") == "success":
                path = Path(ctx["self"]._get_abs_path(ctx["args"].get("path", ""))).resolve()
                if path.is_relative_to("/agent") and path.suffix == ".json" and path.is_file():
                    report_outputs.append(path.read_text(encoding="utf-8"))

    @register("tool_before")
    def trace_tool(ctx):
        events.append({"tool": ctx["tool_name"], "args": dict(ctx["args"])})

    @register("agent_after")
    def trace_end(ctx):
        endings.append(ctx.get("exit_reason", {}))

    threading.Thread(target=agent.run, daemon=True).start()
    for line in sys.stdin:
        request = json.loads(line)
        if request.get("stop"):
            break
        backend = agent.llmclient.backend
        backend.request_seed = request["seed"]
        backend.request_calls = 0
        before = len(backend.usage)
        events.clear()
        endings.clear()
        assistant_replies.clear()
        report_outputs.clear()
        tool_results.clear()
        queue = agent.put_task(request["prompt"])
        while True:
            item = queue.get(timeout=request.get("timeout", 1800))
            if "done" in item:
                break
        # GenericAgent posts 'done' just before task_done; wait for clean state.
        agent.task_queue.join()
        end = endings[-1] if endings else {}
        response = end.get("data")
        final = getattr(response, "content", "") if end.get("result") == "CURRENT_TASK_DONE" else ""
        # GA can show the JSON then end with an acknowledgement after its native
        # code-block confirmation. Use its latest report, never score alternatives.
        final = delivered_report(report_outputs, final, end.get("result") == "CURRENT_TASK_DONE")
        result = {"response": final, "exit_reason": end.get("result", "no_final_response"),
                  "output": item["done"], "tools": list(events),
                  "assistant_replies": list(assistant_replies),
                  "tool_results": list(tool_results),
                  "delivered_outputs": list(report_outputs),
                  "usage": backend.usage[before:], "working": agent.handler.working,
                  "history_messages": len(backend.history)}
        with open("/agent/temp/session_history.json", "w") as handle:
            json.dump(backend.history, handle, ensure_ascii=False)
        protocol.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
        protocol.flush()


if __name__ == "__main__":
    main()
