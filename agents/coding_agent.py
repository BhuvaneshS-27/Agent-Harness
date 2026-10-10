import json
import math
import re
import time
from typing import Dict, List, TypedDict

import ollama
from ddgs import DDGS
from langgraph.graph import END, START, StateGraph

import config

client = ollama.Client(host=config.OLLAMA_HOST)
MAX_ROUNDS = 3


# ---------- State shared by all nodes ----------
class AgentState(TypedDict):
    category: str
    queries: List[str]
    search_results: List[Dict]
    candidates: List[Dict]
    top10: List[Dict]
    rounds: int
    report: str


# ---------- Helpers ----------
def ask_llm_json(prompt: str) -> dict:
    """Ask the local LLM for JSON; never crash on bad output."""
    try:
        reply = client.chat(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0, "num_ctx": 8192},
        )
        text = reply["message"]["content"]
    except Exception as e:
        print(f"  [LLM error] {e}")
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------- Nodes ----------
def plan_queries(state: AgentState) -> dict:
    rounds = state["rounds"] + 1
    print(f"\n[plan_queries] round {rounds}")
    category = state["category"]
    previous = state["queries"]

    prompt = (
        f"Write 3 different web search queries to find the best-performing "
        f"{category} startups with their funding and valuation. "
        f"Do NOT reuse these queries: {previous}. "
        f'Return JSON only: {{"queries": ["...", "...", "..."]}}'
    )
    queries = ask_llm_json(prompt).get("queries", [])
    queries = [q for q in queries if isinstance(q, str) and q.strip()]

    if not queries:  # fallback if the small model fails
        queries = [
            f"top {category} startups 2026 funding valuation",
            f"best {category} startups latest funding round",
            f"fastest growing {category} companies",
        ][: 3]
        queries = [q + f" (round {rounds})" if q in previous else q for q in queries]

    print("  queries:", queries)
    return {"queries": previous + queries, "rounds": rounds}  


def search(state: AgentState) -> dict:
    print("[search]")
    new_queries = state["queries"][-3:]
    seen, results = set(), []
    for q in new_queries:
        try:
            for r in DDGS().text(q, max_results=8):
                if r["href"] not in seen:
                    seen.add(r["href"])
                    results.append(
                        {"title": r["title"], "url": r["href"], "body": r["body"]}
                    )
        except Exception as e:
            print(f"  [search error] {q}: {e}")
        time.sleep(1)  # be polite, avoid rate limits
    print(f"  found {len(results)} results")
    return {"search_results": results}


def extract(state: AgentState) -> dict:
    print("[extract]")
    snippets = "\n".join(
        f"- {r['title']} | {r['body']} | {r['url']}"
        for r in state["search_results"]
    )[:6000]

    prompt = (
        f"From the search results below, list real {state['category']} startups "
        f"mentioned. For each, give:\n"
        f'name, description (short), funding_usd_millions (number or null), '
        f'valuation_usd_billions (number or null), latest_news (short), '
        f"source_url.\n"
        f"Use ONLY facts in the results. Use null if unknown. Do not invent.\n"
        f'Return JSON only: {{"startups": [ ... ]}}\n\n'
        f"SEARCH RESULTS:\n{snippets}"
    )
    found = ask_llm_json(prompt).get("startups", [])

    merged = {c["name"].lower(): c for c in state["candidates"]}
    for s in found:
        if isinstance(s, dict) and isinstance(s.get("name"), str) and s["name"].strip():
            key = s["name"].strip().lower()
            if key not in merged:
                merged[key] = s
    print(f"  total candidates: {len(merged)}")
    return {"candidates": list(merged.values())}


def rank(state: AgentState) -> dict:
    """Scoring rubric (edit and explain in your report):
    score = 0.6 * log10(1 + funding in $M) + 0.4 * log10(1 + valuation in $M)
    """
    print("[rank]")
    scored = []
    for c in state["candidates"]:
        funding = to_float(c.get("funding_usd_millions")) or 0
        valuation_m = (to_float(c.get("valuation_usd_billions")) or 0) * 1000
        c["score"] = round(
            0.6 * math.log10(1 + funding) + 0.4 * math.log10(1 + valuation_m), 3
        )
        scored.append(c)
    scored.sort(key=lambda x: x["score"], reverse=True)
    return {"top10": scored[:10]}


def route_after_rank(state: AgentState) -> str:
    enough = len(state["top10"]) >= 10
    if enough or state["rounds"] >= MAX_ROUNDS:
        return "done"
    print(f"[route] only {len(state['top10'])} found, searching again")
    return "retry"


def report(state: AgentState) -> dict:
    print("[report]")
    lines = [f"# Top startups: {state['category']}\n"]
    lines.append(
        "Ranking rubric: 0.6*log10(1+funding $M) + 0.4*log10(1+valuation $M)\n"
    )
    for i, c in enumerate(state["top10"], 1):
        lines.append(f"## {i}. {c.get('name')}  (score {c.get('score')})")
        lines.append(f"- Description: {c.get('description')}")
        lines.append(f"- Funding ($M): {c.get('funding_usd_millions')}")
        lines.append(f"- Valuation ($B): {c.get('valuation_usd_billions')}")
        lines.append(f"- Latest news: {c.get('latest_news')}")
        lines.append(f"- Source: {c.get('source_url')}\n")
    return {"report": "\n".join(lines)}


# ---------- Build the graph ----------
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("plan_queries", plan_queries)
    g.add_node("search", search)
    g.add_node("extract", extract)
    g.add_node("rank", rank)
    g.add_node("report", report)

    g.add_edge(START, "plan_queries")
    g.add_edge("plan_queries", "search")
    g.add_edge("search", "extract")
    g.add_edge("extract", "rank")
    g.add_conditional_edges(
        "rank", route_after_rank, {"retry": "plan_queries", "done": "report"}
    )
    g.add_edge("report", END)
    return g.compile()


def run_coding_agent(category: str) -> dict:
    app = build_graph()
    return app.invoke(
        {
            "category": category,
            "queries": [],
            "search_results": [],
            "candidates": [],
            "top10": [],
            "rounds": 0,
            "report": "",
        }
    )