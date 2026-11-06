# app.py  --  ажиллуулах: streamlit run app.py
import csv
import datetime
import streamlit as st
import Rag as rag

st.set_page_config(page_title="Хөдөлмөрийн хуулийн туслах", page_icon="⚖️")


@st.cache_resource(show_spinner="Индекс ачаалж байна...")
def load_index():
    return rag.build_index()


chunks, vectors = load_index()

# ---------- Хажуугийн цэс ----------
with st.sidebar:
    st.header("Тохиргоо")
    show_ctx = st.checkbox("Загварт очсон эх бичвэрийг харуулах", value=True)
    log_q = st.checkbox("Асуултыг user_questions.csv-д бүртгэх", value=False)
    if st.button("Яриаг цэвэрлэх"):
        st.session_state.messages = []
        st.rerun()
    st.caption("Асуулт бүр бие даасан: өмнөх яриаг загвар санахгүй.")

st.title("⚖️ Хөдөлмөрийн хуулийн туслах")
st.caption("Энэ нь хуулийн зөвлөгөө биш. Хариултыг заавал эх бичвэр, заалттай тулгаж шалгаарай.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_extras(m):
    if m.get("sources"):
        with st.expander("Эх сурвалж"):
            for s in m["sources"]:
                st.write(s)
    if m.get("context") and show_ctx:
        with st.expander("Загварт очсон эх бичвэр"):
            st.text(m["context"])


# ---------- Өмнөх яриаг дахин зурна ----------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])
        if m["role"] == "assistant":
            render_extras(m)

# ---------- Шинэ асуулт ----------
q = st.chat_input("Асуултаа бичнэ үү...")
if q:
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.write(q)

    with st.chat_message("assistant"):
        with st.spinner("Хариулт бэлдэж байна..."):
            rag.DEBUG = False
            try:
                reply, hits = rag.answer(q, chunks, vectors)
                ctx = rag.build_context(q, hits, chunks, rag.FOCUS_TOP)
            except Exception as e:
                reply = f"Алдаа гарлаа: {e}\n(Ollama ажиллаж байгаа эсэхийг шалгаарай.)"
                hits, ctx = [], ""
        msg = {
            "role": "assistant",
            "content": reply,
            "sources": [f"{rag.describe(c)}  (cosine {s:.2f})" for c, s in hits],
            "context": ctx,
        }
        st.write(reply)
        render_extras(msg)
    st.session_state.messages.append(msg)

    if log_q:  # зөвхөн бичнэ, уншихгүй
        with open("user_questions.csv", "a", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(
                [datetime.datetime.now().isoformat(timespec="minutes"), q, reply]
            )