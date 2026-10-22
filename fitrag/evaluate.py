"""Runs the 5-question test sheet and prints actual results. Record these in your README."""
import json
from rag import ask
for t in json.load(open("tests.json")):
    r = ask(t["question"])
    print(f"\n#{t['id']} [{t['designed_as'][:4]}] {t['question']}")
    print("  expected:", t["expected"])
    print("  score   :", round(r["top_score"], 2), "| sources:", [s["source"] for s in r["sources"]])
    print("  answer  :", r["answer"][:300].replace("\n", " "))
