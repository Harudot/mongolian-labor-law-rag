"""RAG-ийг eval_set.json дээр хэмжинэ. Үр дүн бүрийг eval_log.csv-д нэмнэ.

Жишээ:
  python eval_retrieval.py                               # зөвхөн хайлт (хурдан)
  python eval_retrieval.py --embed nomic-embed-text      # өөр embedding
  python eval_retrieval.py --chars 600                   # Туршилт A: жижиг хэсэг
  python eval_retrieval.py --norm 0                      # Туршилт B-г унтраах
  python eval_retrieval.py --hybrid 1 --wd 0.3           # embedding-ийн жинг бууруулах
  python eval_retrieval.py --answers --focus 0           # заалтаар төвлөрүүлэхгүй
  python eval_retrieval.py --answers                     # хариултыг бас шалгана (удаан)
  python eval_retrieval.py --set eval_holdout.json       # ШИНЭ асуултууд (тохируулалтад ашиглаагүй)
  python eval_retrieval.py --cdense 0                    # заалтын embedding-гүй (хуучин) хувилбар
  python eval_retrieval.py --set eval_dev2.json --answers    # 2-р хөгжүүлэлтийн багц (201-218)
  python eval_retrieval.py --set eval_final.json --answers   # ЭЦСИЙН шалгалт: хамгийн сүүлд НЭГ л удаа
"""
import csv
import json
import re
import time
import argparse
import datetime
import os
import Rag as rag

ap = argparse.ArgumentParser()
ap.add_argument("--embed", help="embedding модель, жишээ нь nomic-embed-text")
ap.add_argument("--chars", type=int, help="нэг хэсгийн дээд урт (MAX_CHARS)")
ap.add_argument("--norm", type=int, choices=[0, 1], help="тооны үгийг цифр болгох (1/0)")
ap.add_argument("--hybrid", type=int, choices=[0, 1], help="BM25-тай нийлүүлэх (1/0)")
ap.add_argument("--wd", type=float, help="RRF-д embedding-ийн жин (W_DENSE)")
ap.add_argument("--cdense", type=int, choices=[0, 1], help="заалтыг embedding-ээр ч эрэмбэлэх (1/0)")
ap.add_argument("--focus", type=int, help="LLM-д өгөх заалтын тоо (0 = унтраах)")
ap.add_argument("--answers", action="store_true", help="LLM хариултыг бас шалгах")
ap.add_argument("--set", default="eval_set.json", help="асуултын багц, жишээ нь eval_holdout.json")
ap.add_argument("--k", type=int, default=5, help="хайлтын хэдэн хэсгийг харах вэ")
args = ap.parse_args()

if args.embed:
    rag.EMBED_MODEL = args.embed
if args.chars:
    rag.MAX_CHARS = args.chars
if args.hybrid is not None:
    rag.HYBRID = bool(args.hybrid)
if args.wd is not None:
    rag.W_DENSE = args.wd
if args.cdense is not None:
    rag.CLAUSE_DENSE = bool(args.cdense)
if args.focus is not None:
    rag.FOCUS_TOP = args.focus
if args.norm is not None:
    rag.NORMALIZE_EMBED = bool(args.norm)

with open(args.set, encoding="utf-8") as f:
    tests = json.load(f)

# Файлын нэр ба доторх id таарч байгаа эсэхийг шалгана (багцууд андуурагдаж дарагдахаас сэргийлнэ)
EXPECT_IDS = {"eval_set.json": (1, 99), "eval_holdout.json": (101, 199),
              "eval_dev2.json": (201, 299), "eval_final.json": (301, 399)}
base = os.path.basename(args.set)
if base in EXPECT_IDS:
    lo, hi = EXPECT_IDS[base]
    wrong = [t["id"] for t in tests if not lo <= t["id"] <= hi]
    if wrong:
        print(f"\nАНХААР: {base} дотор id {wrong[0]} гэх мэт байна ({lo}-{hi} байх ёстой). "
              f"Өөр багцын файлаар дарагдсан байж магадгүй. Үр дүнг нэмэхээсээ өмнө шалга!\n")

chunks, vectors = rag.build_index()
print(f"\nEmbedding: {rag.EMBED_MODEL} | MAX_CHARS: {rag.MAX_CHARS} | "
      f"norm: {int(rag.NORMALIZE_EMBED)} | hybrid: {int(rag.HYBRID)} | "
      f"W_DENSE: {rag.W_DENSE} | focus: {rag.FOCUS_TOP} | cdense: {int(rag.CLAUSE_DENSE)} | LLM: {rag.LLM_MODEL} | хэсэг: {len(chunks)} | багц: {args.set}\n")

# ---------- 1. Хайлтын чанар ----------
scored = [t for t in tests if t["article"] is not None]
ranks = []
top1_in = []      # хуульд байгаа асуултын эхний хэсгийн cosine
for t in scored:
    hits = rag.retrieve(t["question"], chunks, vectors, k=args.k)
    top1_in.append(hits[0][1])
    arts = [c["article"] for c, _ in hits]
    rank = arts.index(t["article"]) + 1 if t["article"] in arts else None
    ranks.append(rank)
    if rank != 1:
        shown = ", ".join(f"Зүйл {a}" for a in arts[:args.k])
        status = f"{rank}-р байранд" if rank else f"top-{args.k}-д олдсонгүй"
        print(f"  [{t['id']:>2}] {status} | хүлээсэн: Зүйл {t['article']} | олдсон: {shown}")
        print(f"       {t['question']}")


def hit(n):
    return sum(1 for r in ranks if r is not None and r <= n) / len(ranks)


mrr = sum(1 / r for r in ranks if r) / len(ranks)
print(f"\nХАЙЛТ ({len(ranks)} асуулт):  hit@1 = {hit(1):.0%}   hit@3 = {hit(3):.0%}   "
      f"hit@5 = {hit(5):.0%}   MRR = {mrr:.2f}")

# ---------- 1б. Хуульд байхгүй асуултыг cosine-оор ялгаж болох эсэх ----------
unscored = [t for t in tests if t["article"] is None]
if unscored:
    top1_out = [rag.retrieve(t["question"], chunks, vectors, k=1)[0][1] for t in unscored]
    srt = sorted(top1_in)
    print(f"\nCOSINE (эхний хэсэг): хуульд БАЙГАА асуулт  min={srt[0]:.2f}  медиан={srt[len(srt)//2]:.2f} | "
          f"хуульд БАЙХГҮЙ асуулт  max={max(top1_out):.2f}")
    print("  (байхгүй асуултын max нь байгаа асуултын min-ээс эрс бага бол босгоор ялгаж болно)")

# ---------- 2. Хариултын чанар (заавал биш) ----------
answers_ok = ""
cite_total = ""
if args.answers:
    print("\n--- Хариулт шалгаж байна (удаан) ---\n")
    ok = ok_cite = 0
    for t in tests:
        t0 = time.time()
        reply, _ = rag.answer(t["question"], chunks, vectors)
        dt = time.time() - t0
        low = reply.lower()
        if t["article"] is None:
            good = "олдсонгүй" in low
        else:
            good = any(a.lower() in low for a in t["answers"])
        ok += good
        cited = [int(x) for x in re.findall(r"Зүйл\s*(\d+)", reply)]
        cite_ok = good and (t["article"] is None or t["article"] in cited)
        ok_cite += cite_ok
        note = "" if cite_ok or not good else "  [түлхүүр үг таарсан ч Зүйл зөрсөн]"
        soft = t["article"] is None and not good and any(
            p in low for p in ("заагдаагүй", "тусгагдаагүй", "заагаагүй", "байхгүй"))
        if soft:
            note += "  [зөөлөн: 'заагдаагүй' гэсэн, гараар унш]"
        print(f"{'ЗӨВ ' if good else 'АЛДАА'} {dt:5.1f}s | [{t['id']:>2}] {t['question']}{note}")
        print(f"      -> {reply}\n")
    answers_ok = f"{ok}/{len(tests)}"
    cite_total = f"{ok_cite}/{len(tests)}"
    print(f"ХАРИУЛТ: түлхүүр үгээр {answers_ok} зөв; түлхүүр үг БА зөв Зүйл иш татсан {cite_total} "
          f"(алдсан бүрийг өөрөө уншаарай)")

# ---------- 3. Үр дүнг eval_log.csv-д хадгалах ----------
new_file = not os.path.exists("eval_log.csv")
with open("eval_log.csv", "a", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    if new_file:
        w.writerow(["огноо", "embedding", "MAX_CHARS", "norm", "хэсэг", "hit@1",
                    "hit@3", "hit@5", "MRR", "хариулт зөв", "hybrid", "W_DENSE", "focus", "багц",
                    "зөв Зүйлтэй", "cdense"])
    w.writerow([datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), rag.EMBED_MODEL,
                rag.MAX_CHARS, int(rag.NORMALIZE_EMBED), len(chunks),
                f"{hit(1):.2f}", f"{hit(3):.2f}", f"{hit(5):.2f}", f"{mrr:.2f}",
                answers_ok, int(rag.HYBRID), rag.W_DENSE, rag.FOCUS_TOP, args.set,
                cite_total if args.answers else "", int(rag.CLAUSE_DENSE)])
print("\nҮр дүнг eval_log.csv-д хадгаллаа.")