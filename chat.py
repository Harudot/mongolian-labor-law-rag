# chat.py
import sys
import Rag as rag

def main():
    rag.DEBUG = "--debug" in sys.argv
    chunks, vectors = rag.build_index()
    print("Асуултаа бич (гарах: q)")
    while True:
        q = input("\n> ").strip()
        if q.lower() in ("q", "quit", "exit", ""):
            break
        reply, hits = rag.answer(q, chunks, vectors)
        print("\n" + reply)
        print("\n--- Эх сурвалж ---")
        for c, score in hits:
            print(f"{rag.describe(c)}  (score={score:.2f})")

if __name__ == "__main__":
    main()