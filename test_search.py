from ddgs import DDGS

results = DDGS().text("top FinTech startups 2026", max_results=5)
for r in results:
    print(r["title"])
    print(r["href"])
    print(r["body"][:150])
    print("---")