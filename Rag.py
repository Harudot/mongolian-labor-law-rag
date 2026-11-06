import os
import re
import glob
import math
from collections import Counter
import hashlib
import numpy as np
import ollama

# ---------- Тохиргоо ----------
EMBED_MODEL = "qwen3-embedding:4b"  
LLM_MODEL = "qwen3.5:4b"
THINK = False          
DOCS_DIR = "docs"
CACHE_DIR = "index_cache"
MAX_CHARS = 600        # нэг хэсгийн дээд урт (600 нь 1200-аас сайн гарсан)
NORMALIZE_EMBED = False # Туршилт B: хэмжилтээр MRR-ийг бууруулсан тул унтраав
TOP_K = 5              # хариулт үүсгэхэд хэдэн хэсэг өгөх вэ
HYBRID = True          # embedding + BM25 (түлхүүр үг)-ийг RRF-ээр нийлүүлэх
BM25_PREFIX = 4        # монгол нөхцөлийг тайрахын тулд үгийн эхний N үсэг
W_DENSE = 0.3          # RRF жин (embedding). 0.3 нь хамгийн сайн гарсан (MRR 0.97)
W_BM25 = 2.0
RRF_K = 60
TEMPERATURE = 0.0      # 0 = тогтвортой хариулт (өмнө нь 0.2)
NUM_CTX = 4096         # 4 GB GPU дээр удаан бол 3072 болгож үз
FOCUS_TOP = 5          # хариултад өгөх хамгийн холбоотой заалтын тоо (0 = унтраах). 3 -> 5: 'жилд хэдэн хоног амралт' асуултад 99.3 заалт гараад орхигддог байсан
RETRY_FOCUS = (9, 0)   # NEW: хариулт олдохгүй/тоо нь баталгаажихгүй бол контекстыг томруулж дахин оролдоно (0 = төвлөрүүлэлтгүй, бүтэн хэсгүүд)
FILTER_GROUPS = True   # асуултад дурдаагүй тусгай бүлгийн заалтыг контекстоос хасах
CHECK_NUMBERS = True   # NEW: хариултын тоо контекстэд байхгүй бол "Мэдээлэл олдсонгүй" гэнэ
TOP_ARTICLE_BOOST = 1.3  # NEW: хайлтын 1-р зүйлийн заалтын оноог үржүүлнэ (1.0 = унтраах)
SEED = 42              # NEW: ижил асуултад ижил хариулт гарахын тулд (хэмжилт давтагдах)
CLAUSE_DENSE = False    # NEW: заалт бүрийг embedding-ээр ч эрэмбэлнэ (BM25-тай нийлүүлнэ)
W_CLAUSE_DENSE = 1.0   # NEW: заалтын RRF-д embedding-ийн жин
W_CLAUSE_BM25 = 1.0    # NEW: заалтын RRF-д BM25-ын жин
DEBUG = False


# ---------- Монгол тооны үг -> цифр ----------
NUM_WORDS = {
    "нэг": "1", "хоёр": "2", "гурван": "3", "гурав": "3",
    "дөрвөн": "4", "дөрөв": "4", "таван": "5", "тав": "5",
    "зургаан": "6", "зургаа": "6", "долоон": "7", "долоо": "7",
    "найман": "8", "найм": "8", "есөн": "9", "ес": "9",
    "арван": "10", "арав": "10",
}
UNITS = r"(?=\s+(?:жил|сар|өдөр|хоног|цаг|хүн|удаа))"


def normalize_numbers(text):
    pattern = r"\b(" + "|".join(NUM_WORDS) + r")\b" + UNITS
    return re.sub(
        pattern,
        lambda m: NUM_WORDS[m.group(1).lower()],
        text,
        flags=re.IGNORECASE,
    )


# ---------- Асуулт олдсон текстэд суурилсан эсэхийг шалгах ----------
STOP = {
    "хэд", "хэдэн", "хэдээс", "вэ", "бэ", "юу", "хэзээ", "хаана", "хэрхэн",
    "яаж", "байх", "байна", "ямар", "хэн", "мөн", "уу", "үү",
}


def grounded(question, context, prefix=4, min_ratio=0.5):
    words = re.findall(r"[а-яөүё]+", question.lower())
    keys = [w for w in words if len(w) >= 4 and w not in STOP]
    if not keys:
        return True
    ctx = context.lower()
    # Монгол хэл нөхцөл залгадаг тул үгийн эхний 4 үсгээр тулгана (BM25-тэй адил; 'хөлсөнд' -> 'хөлс')
    found = sum(1 for w in keys if w[:prefix] in ctx)
    return found / len(keys) >= min_ratio


# ---------- NEW: Хариултын тоо контекстэд байгаа эсэхийг шалгах ----------
_FRAC_WORDS = {"нэг": 1, "хоёр": 2, "гурван": 3, "гурав": 3, "дөрвөн": 4, "дөрөв": 4,
               "таван": 5, "тав": 5, "зургаан": 6, "зургаа": 6, "долоон": 7, "долоо": 7,
               "найман": 8, "найм": 8, "есөн": 9, "ес": 9}


def fraction_equivalents(text):
    """Хуулийн 'нэг аравны тав' (=0.5, 50%) ба 'тэг аравны гурван хувь' (=0.3%) гэх мэт
    хэллэгийн зөв тоон хэлбэрүүд. Загвар 'хэдэн хувь' гэж асуухад хөрвүүлж хариулдаг тул
    ийм хөрвүүлгийг буруу гэж хаахгүй. 'тэг аравны'-д зөвхөн аравны бутархайг (0.N) зөвшөөрнө."""
    out = set()
    for m in re.finditer(r"\b(тэг|нэг)\s+аравны\s+([а-яөүё]+)", text.lower()):
        n = _FRAC_WORDS.get(m.group(2))
        if n is None:
            continue
        out |= {f"0.{n}", f"0,{n}"}
        if m.group(1) == "нэг":
            out |= {f"{n}0", f"1.{n}", f"1,{n}"}   # 0.5 дахин = 50% = нийт 1.5 дахин
    return out


def numbers_supported(reply, context, question=""):
    """Хариултад гарсан тоо бүр контекст эсвэл асуултад байх ёстой.
    Загвар тоо зохиосон (жишээ нь '3.7 хоног') тохиолдлыг барина."""
    body = re.sub(r"\(.*?\)", " ", reply)                              # (Зүйл 93) хэсгийг хасна
    body = re.sub(r"Зүйл\s*\d+(?:\.\d+)*", " ", body, flags=re.IGNORECASE)  # Зүйл 99.5.1 гэх мэтийг хасна
    allowed = context + "\n" + question + "\n" + " ".join(fraction_equivalents(context))
    for n in re.findall(r"\d+(?:[.,]\d+)?", body):
        # өөр тооны хэсэг биш, бүтэн тоо байх ёстой
        if not re.search(r"(?<![\d.,])" + re.escape(n) + r"(?!\d)", allowed):
            return False
    return True


# ---------- 1. Баримтыг зүйлээр нь хуваах ----------
ART_RE = re.compile(r"^\s*(\d+)\s*(?:дүгээр|дугаар)\s*зүйл\.\s*(.*)$", re.M)


def split_articles(text):
    """'N дүгээр зүйл. Гарчиг' бүрээр нь (дугаар, гарчиг, агуулга) гаргана."""
    ms = list(ART_RE.finditer(text))
    arts = []
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(text)
        arts.append((int(m.group(1)), m.group(2).strip(), text[m.end():end].strip()))
    return arts


def make_chunks(num, title, body, max_chars=None):
    """Нэг зүйл = нэг хэсэг. Урт зүйлийг заалтын (догол мөрийн) хилээр таслаад,
    хэсэг бүрийн эхэнд зүйлийн дугаар, гарчгийг давтан бичнэ."""
    max_chars = max_chars or MAX_CHARS      # ажиллах үеийн утгыг авна
    header = f"Зүйл {num}. {title}"
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    pieces, cur = [], ""
    for p in paras:
        parts = [p] if len(p) <= max_chars else re.split(r"(?<=[.;])\s+", p)
        for part in parts:
            if cur and len(cur) + len(part) + 1 > max_chars:
                pieces.append(cur)
                cur = part
            else:
                cur = (cur + "\n" + part) if cur else part
    if cur:
        pieces.append(cur)
    return [
        {"article": num, "title": title, "text": f"{header}\n{pc}"}
        for pc in pieces
    ]


def plain_chunks(text, size=500, overlap=100):
    """'зүйл' бүтэцгүй энгийн текстийг тэмдэгтийн тоогоор хуваана."""
    out, start = [], 0
    while start < len(text):
        out.append(text[start:start + size])
        start += size - overlap
    return out


def load_chunks(folder):
    chunks = []
    for path in sorted(glob.glob(os.path.join(folder, "*.txt"))):
        name = os.path.basename(path)
        with open(path, encoding="utf-8") as f:
            text = f.read().replace("\xa0", " ")
        arts = split_articles(text)
        if arts:
            for num, title, body in arts:
                for c in make_chunks(num, title, body):
                    c["source"] = name
                    chunks.append(c)
        else:
            for piece in plain_chunks(text):
                chunks.append({"article": None, "title": "", "source": name, "text": piece})
    return chunks


# ---------- 2. Embedding ----------
def embed(texts, kind="doc"):
    if NORMALIZE_EMBED:
        texts = [normalize_numbers(t) for t in texts]
    # Зарим embedding модель баримт/асуултыг өөр угтвараар ялгадаг
    if "nomic" in EMBED_MODEL:
        prefix = "search_document: " if kind == "doc" else "search_query: "
    elif "qwen3-embedding" in EMBED_MODEL and kind == "query":
        prefix = "Instruct: Given a question, retrieve passages that answer it\nQuery: "
    else:
        prefix = ""
    out = []
    for i in range(0, len(texts), 16):
        batch = [prefix + t for t in texts[i:i + 16]]
        res = ollama.embed(model=EMBED_MODEL, input=batch)
        out.extend(res["embeddings"])
    vecs = np.array(out, dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs


# ---------- 3. Индекс (диск дээр cache хийнэ) ----------
def build_index(folder=DOCS_DIR):
    chunks = load_chunks(folder)
    if not chunks:
        raise SystemExit(f"'{folder}' дотор .txt файл алга.")
    key_parts = [EMBED_MODEL, str(MAX_CHARS), f"norm{int(NORMALIZE_EMBED)}"]
    key_src = "|".join(key_parts + [c["text"] for c in chunks])
    key = hashlib.sha256(key_src.encode("utf-8")).hexdigest()[:16]
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{key}.npy")
    if os.path.exists(path):
        vectors = np.load(path)
        print(f"{len(chunks)} хэсэг (cache-ээс уншлаа).")
    else:
        print(f"{len(chunks)} хэсгийг embed хийж байна ({EMBED_MODEL})...")
        vectors = embed([c["text"] for c in chunks], kind="doc")
        np.save(path, vectors)
        print("Индекс хадгаллаа.")
    if CLAUSE_DENSE:
        clause_vectors(chunks)
    return chunks, vectors


# ---------- 4. Хайлт ----------
Q_STOP = {"хэд", "хэдэн", "вэ", "бэ", "юу", "байх", "ямар", "уу", "үү", "хэдээс",
          "юм", "авдаг", "авах"}   # CHANGED: ярианы дагавар үгс, асуултын утгыг ялгадаггүй
_bm = {}

# Асуултын үг хуулийн хэллэгээс өөр байдаг тохиолдол (үгийн эхний BM25_PREFIX үсгээр).
# Жишээ: "ажлаас гарах" -> хуульд "цуцлах", "ажлын байраа орхих"
# Зориуд ганц чиглэлтэй: "өдөр" -> "хоног" гэж өргөтгөвөл "долоо хоног" бүхий
# заалтууд (жишээ нь 109.2) баярын өдрийн асуулттай андуурагдана.
SYNONYMS = {
    "гара": ["цуцл", "орхи", "сана"],
    "хоно": ["өдөр"],    # NEW: "хэдэн хоног" -> хуульд "ажлын 15 өдөр"
    "жилд": ["жил"],     # NEW: "жилд" -> хуульд "ажлын жил бүр"
}


def _toks(text):
    return [w[:BM25_PREFIX] for w in re.findall(r"[а-яөүё0-9]+", text.lower())
            if w not in Q_STOP and len(w) >= 2]


def _bm25_index(chunks):
    key = id(chunks)
    if key not in _bm:
        docs = [Counter(_toks(c["text"])) for c in chunks]
        lens = np.array([sum(d.values()) for d in docs], dtype=np.float32)
        df = Counter(w for d in docs for w in d)
        _bm.clear()
        _bm[key] = (docs, lens, df, float(lens.mean()))
    return _bm[key]


def bm25_scores(question, chunks, k1=1.2, b=0.75):
    docs, lens, df, avg = _bm25_index(chunks)
    n = len(docs)
    out = np.zeros(n, dtype=np.float32)
    q_toks = set(_toks(question))
    for w in list(q_toks):
        q_toks.update(SYNONYMS.get(w, []))
    for w in q_toks:
        if w not in df:
            continue
        idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
        for i, d in enumerate(docs):
            f = d.get(w, 0)
            if f:
                out[i] += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * lens[i] / avg))
    return out


def retrieve(question, chunks, vectors, k=None):
    k = k or TOP_K
    q = embed([question], kind="query")[0]
    dense = vectors @ q
    if not HYBRID:
        top = np.argsort(dense)[::-1][:k]
        return [(chunks[i], float(dense[i])) for i in top]
    bm = bm25_scores(question, chunks)
    fused = np.zeros(len(chunks), dtype=np.float32)
    for w, sc in ((W_DENSE, dense), (W_BM25, bm)):
        order = np.argsort(sc)[::-1]
        for rank, i in enumerate(order):
            if sc[i] > 0 or w == W_DENSE:
                fused[i] += w / (RRF_K + rank + 1)
    top = np.argsort(fused)[::-1][:k]
    return [(chunks[i], float(dense[i])) for i in top]


# ---------- Заалтаар төвлөрүүлэх ----------
# Олдсон зүйлүүдийн доторх заалт бүрийг BM25-аар асуулттай дахин эрэмбэлж, хамгийн
# холбоотой FOCUS_TOP заалтыг (дэд заалтуудтай нь) л LLM-д өгнө. Контекст жижиг болж,
# 4B загвар буруу заалт (жишээ нь 109.2 оронд 109.4) сонгох эрсдэл буурна.
CLAUSE_RE = re.compile(r"^(\d+(?:\.\d+)+)\.")
_cl = {}


def _clause_units(chunks):
    key = id(chunks)
    if key not in _cl:
        units = []
        for c in chunks:
            if c["article"] is None:
                continue
            for line in c["text"].split("\n")[1:]:
                m = CLAUSE_RE.match(line)
                if m:
                    units.append({"id": m.group(1), "article": c["article"],
                                  "title": c["title"], "text": line})
                elif units and units[-1]["article"] == c["article"]:
                    units[-1]["text"] += " " + line
        docs = [Counter(_toks(u["text"])) for u in units]
        df = Counter(w for d in docs for w in d)
        avg = sum(sum(d.values()) for d in docs) / max(len(docs), 1)
        _cl.clear()
        _cl[key] = (units, docs, df, avg)
    return _cl[key]


_cvec = {}


def clause_vectors(chunks):
    """Заалт бүрийн embedding (зүйлийн гарчиг + заалтын текст). Диск дээр cache хийнэ."""
    units = _clause_units(chunks)[0]
    texts = [f"Зүйл {u['article']}. {u['title']}\n{u['text']}" for u in units]
    key_src = "|".join([EMBED_MODEL, "clause", f"norm{int(NORMALIZE_EMBED)}"] + texts)
    key = hashlib.sha256(key_src.encode("utf-8")).hexdigest()[:16]
    if key in _cvec:
        return _cvec[key]
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"clause_{key}.npy")
    if os.path.exists(path):
        vecs = np.load(path)
        print(f"{len(units)} заалт (cache-ээс уншлаа).")
    else:
        print(f"{len(units)} заалтыг embed хийж байна ({EMBED_MODEL})... (нэг удаа)")
        vecs = embed(texts, kind="doc")
        np.save(path, vecs)
    _cvec.clear()
    _cvec[key] = vecs
    return vecs


def focus_context(question, hits, chunks, top_n=3, k1=1.2, b=0.75):
    units, docs, df, avg = _clause_units(chunks)
    arts = []
    for c, _ in hits:
        if c["article"] is not None and c["article"] not in arts:
            arts.append(c["article"])
    idx = [i for i, u in enumerate(units) if u["article"] in arts]
    if not idx:
        return None
    n = len(units)
    q_toks = set(_toks(question))
    for w in list(q_toks):
        q_toks.update(SYNONYMS.get(w, []))
    scores = {}
    for i in idx:
        d = docs[i]
        length = sum(d.values())
        sc = 0.0
        for w in q_toks:
            f = d.get(w, 0)
            if f:
                idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                sc += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * length / avg))
        if units[i]["article"] == arts[0]:   # NEW: 1-р зүйлийн заалтад бага зэрэг давуу эрх
            sc *= TOP_ARTICLE_BOOST
        scores[i] = sc
    order = sorted(idx, key=lambda i: -scores[i])
    if CLAUSE_DENSE:
        # NEW: BM25 нь зөвхөн үгээр тулгадаг ('чөлөө' != 'амралт', 'сар' != 'хоног').
        # Заалтыг embedding-ээр ч эрэмбэлж, хоёр эрэмбийг RRF-ээр нийлүүлнэ.
        cv = clause_vectors(chunks)
        qv = embed([question], kind="query")[0]
        dense = {i: float(cv[i] @ qv) for i in idx}
        r_bm = {i: r for r, i in enumerate(order)}
        r_dn = {i: r for r, i in enumerate(sorted(idx, key=lambda i: -dense[i]))}
        fused = {i: W_CLAUSE_BM25 / (RRF_K + r_bm[i] + 1) + W_CLAUSE_DENSE / (RRF_K + r_dn[i] + 1)
                 for i in idx}
        order = sorted(idx, key=lambda i: -fused[i])
    pick = {}    # заалтын индекс -> хамгийн өндөр эрэмбийн байр (бага = илүү холбоотой)
    for rank, i in enumerate(order[:top_n]):
        u = units[i]
        for j in idx:
            v = units[j]
            if v["article"] == u["article"] and (v["id"] == u["id"]
                                                  or v["id"].startswith(u["id"] + ".")):
                pick[j] = min(rank, pick.get(j, rank))
    parts = []
    for a in arts:
        # хамгийн холбоотой заалтыг эхэнд нь тавина
        js = sorted((j for j in pick if units[j]["article"] == a),
                    key=lambda j: (pick[j], j))
        if js:
            parts.append(f"Зүйл {a}. {units[js[0]]['title']}\n"
                         + "\n".join(units[j]["text"] for j in js))
    return "\n\n".join(parts) if parts else None


# ---------- Тусгай бүлгийн заалтыг шүүх ----------
# Асуултад дурдаагүй бүлэгт (насанд хүрээгүй, жирэмсэн гэх мэт) хамаарах заалтыг
# контекстоос хасна. Ингэснээр 4B загвар ерөнхий дүрмийн оронд тусгай заалтыг сонгохгүй.
SPECIAL_GROUPS = ["насанд хүрээгүй", "хөгжлийн бэрхшээл", "жирэмсэн", "хөхүүл", "эмэгтэй"]


def drop_special_groups(question, context):
    ql = question.lower()
    keep = []
    for line in context.split("\n"):
        low = line.lower()
        groups = [g for g in SPECIAL_GROUPS if g in low]
        if groups and not any(g in ql for g in groups):
            continue
        keep.append(line)
    return "\n".join(keep)


# ---------- 5. Хариулт ----------
SYSTEM_PROMPT = (
    "You are an assistant for Mongolian labor law. Use ONLY the CONTEXT. "
    "Step 1: find the exact sentence in the CONTEXT that answers the question. "
    "Step 2: compare the numbers and conditions in the question with that sentence "
    "carefully (for example 'from the 6th year to the end of the 10th year' "
    "covers the 8th year). "
    "Pay attention to WHO a clause applies to (for example minors, disabled people, "
    "pregnant women): use such a special-group clause only if the question "
    "mentions that group; otherwise use the general rule. "
    "Copy every number and number-word EXACTLY as written in the CONTEXT; "
    "never convert, compute or round them (do not turn a number-word fraction "
    "into a decimal or a percentage). "
    "Take a number only from the clause that names the situation asked about, "
    "never from a neighbouring clause. "
    "If one clause gives a general limit and then a different limit for special "
    "cases, state the general limit first and mention the special case briefly. "
    "Match the exact situation in the question (for example public holiday vs weekly "
    "day off vs night work vs overtime) and cite the clause that names that situation. "
    "If a clause states an exception or a condition (for example 'except', 'unless', "
    "'бусад тохиолдолд', '-аас бусад'), give the rule together with its exception; "
    "never state only the prohibition or only the permission. "
    "Do not stretch a clause to a different subject: the CONTEXT must name the same "
    "specific subject the question asks about; if it only covers a different payment, "
    "benefit or situation, reply exactly: Мэдээлэл олдсонгүй. "
    "Step 3: answer in correct, fluent Mongolian (Cyrillic) with spaces between words, "
    "in one or two short sentences, "
    "starting with 'Хариулт:' and finishing with the article in parentheses, "
    "for example (Зүйл 99). Do not add information that was not asked. "
    "If the CONTEXT does not contain the answer, reply exactly: Мэдээлэл олдсонгүй."
    "But if the question assumes a condition or permission (for example 'if the employer "
    "does not agree') and the CONTEXT states the rule for that same situation (for example "
    "the employee may leave after the notice period), answer with that rule and say whether "
    "the clause requires that condition or permission. "
)



def build_context(question, hits, chunks, focus_top):
    context = focus_context(question, hits, chunks, focus_top) if focus_top else None
    if not context:
        context = "\n\n".join(c["text"] for c, _ in hits)
    if FILTER_GROUPS:
        context = drop_special_groups(question, context)
    return normalize_numbers(context)


def ask_llm(context, q_norm):
    res = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nАСУУЛТ: {q_norm}"},
        ],
        think=THINK,
        keep_alive="30m",
        options={"temperature": TEMPERATURE, "seed": SEED, "num_ctx": NUM_CTX},
    )
    text = res["message"]["content"]
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


NOT_VERIFIED = ("Мэдээлэл олдсонгүй (хариултын тоог эх бичвэрээр баталгаажуулж "
                "чадсангүй, асуултаа өөрөөр томъёолж үзнэ үү)")


def answer(question, chunks, vectors):
    hits = retrieve(question, chunks, vectors)
    q_norm = normalize_numbers(question)

    # Эхлээд жижиг контекстээр оролдоно. Хариулт олдохгүй эсвэл тоо нь баталгаажихгүй бол
    # контекстыг томруулж (RETRY_FOCUS) дахин оролдоно. Бүгд бүтэлгүйтвэл "олдсонгүй".
    steps = [FOCUS_TOP] + [s for s in RETRY_FOCUS if s != FOCUS_TOP]
    rejected = False
    for focus_top in steps:
        context = build_context(question, hits, chunks, focus_top)
        if DEBUG:
            print(f"\n=== CONTEXT (focus={focus_top}) ===\n" + context + "\n===============")
        if not grounded(q_norm, context):
            continue
        text = ask_llm(context, q_norm)
        if "олдсонгүй" in text.lower():
            if DEBUG:
                print(f"[focus={focus_top}] модель 'олдсонгүй' гэсэн, томруулж дахин оролдоно")
            continue
        if CHECK_NUMBERS and not numbers_supported(text, context, q_norm):
            rejected = True
            if DEBUG:
                print(f"\n=== ХАЯГДСАН ХАРИУЛТ (focus={focus_top}) ===\n{text}\n========================")
            continue
        return text, hits

    return (NOT_VERIFIED if rejected else "Мэдээлэл олдсонгүй"), hits


def describe(chunk):
    if chunk["article"] is not None:
        return f"Зүйл {chunk['article']}. {chunk['title'][:60]}"
    return chunk["source"]


# ---------- Үндсэн хэсэг ----------
if __name__ == "__main__":
    chunks, vectors = build_index()
    print("Асуултаа бич (гарах бол 'exit'):")
    while True:
        q = input("\n> ").strip()
        if q.lower() in ("exit", "quit", ""):
            break
        reply, hits = answer(q, chunks, vectors)
        print("\n" + reply)
        print("\n--- Ашигласан эх сурвалж ---")
        for c, score in hits:
            print(f"{describe(c)}  (score={score:.2f})")