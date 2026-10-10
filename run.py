import os
import sys

from agents.coding_agent import build_graph, run_coding_agent

category = sys.argv[1] if len(sys.argv) > 1 else "FinTech"

# Save the workflow diagram (useful for your documentation and video)
os.makedirs("outputs", exist_ok=True)
with open("outputs/coding_agent_graph.mmd", "w") as f:
    f.write(build_graph().get_graph().draw_mermaid())

result = run_coding_agent(category)

with open(f"outputs/{category}_report.md", "w", encoding="utf-8") as f:
    f.write(result["report"])

print("\n" + result["report"])