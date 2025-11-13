import uuid
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Embedding model
# -----------------------------
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


# -----------------------------
# PHQ-9 ITEMS + STRUCTURE
# -----------------------------

@dataclass
class PhqItem:
    item_id: int
    text: str
    domain: str
    is_functional: bool = False


PHQ_ITEMS: List[PhqItem] = [
    PhqItem(1, "Little interest or pleasure in doing things", "anhedonia"),
    PhqItem(2, "Feeling down, depressed, or hopeless", "mood"),
    PhqItem(3, "Trouble falling or staying asleep, or sleeping too much", "sleep"),
    PhqItem(4, "Feeling tired or having little energy", "energy"),
    PhqItem(5, "Poor appetite or overeating", "appetite"),
    PhqItem(6, "Feeling bad about yourself—or that you are a failure", "self_esteem"),
    PhqItem(7, "Trouble concentrating on things, such as reading or watching TV", "concentration"),
    PhqItem(8, "Moving or speaking slowly or being very fidgety or restless", "psychomotor"),
    PhqItem(9, "Thoughts that you would be better off dead or hurting yourself", "self_harm"),
    PhqItem(10, "How difficult have these problems made it for you at work, home, or with others?", "functional", True),
]


PHQ_BANDS = [
    (0, 4, "minimal"),
    (5, 9, "mild"),
    (10, 14, "moderate"),
    (15, 19, "moderately severe"),
    (20, 27, "severe"),
]

FUNCTIONAL_IMPAIRMENT_LABELS = {
    0: "not difficult at all",
    1: "somewhat difficult",
    2: "very difficult",
    3: "extremely difficult",
}


# -----------------------------
# KNOWLEDGE BASE
# -----------------------------

KNOWLEDGE_BASE = [
    {
        "topic": "snowflake",
        "chunk": "Snowflake Cortex offers serverless access to LLMs and embeddings near your data, "
                 "letting teams build AI features securely without moving data out.",
        "source": "Snowflake docs (summary)"
    },
    {
        "topic": "snowflake",
        "chunk": "AISQL adds AI tasks like summarize, classify, or translate directly inside SQL. "
                 "Cortex Analyst enables natural language questions on tables and documents.",
        "source": "Snowflake docs (summary)"
    },
    {
        "topic": "emotive_ai",
        "chunk": "Emotive or affective AI estimates emotions from voice, text, or expressions "
                 "to respond with empathy and improve user support.",
        "source": "Affective computing overview"
    },
    {
        "topic": "emotive_ai",
        "chunk": "Ethical emotive AI requires transparency, fairness, human oversight, and avoiding stereotypes.",
        "source": "Ethics notes"
    },
]


# -----------------------------
# EMBEDDINGS FOR RAG
# -----------------------------

@st.cache_resource(show_spinner=False)
def load_embedder():
    if SentenceTransformer is None:
        return None
    try:
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def build_knowledge_index():
    df = pd.DataFrame(KNOWLEDGE_BASE)
    embedder = load_embedder()
    if embedder is None:
        return df, None
    vecs = embedder.encode(df["chunk"].tolist())
    return df, np.array(vecs, dtype="float32")


def rag_query(query: str, k=2):
    df, emb = build_knowledge_index()
    if emb is None:
        return df.head(k).to_dict(orient="records")
    embedder = load_embedder()
    qvec = embedder.encode([query])
    sims = cosine_similarity(qvec, emb)[0]
    idx = sims.argsort()[::-1][:k]
    return [
        {**df.iloc[i].to_dict(), "score": float(sims[i])}
        for i in idx
    ]


# -----------------------------
# PHQ-9 LOGIC
# -----------------------------

def start_session() -> str:
    return str(uuid.uuid4())


def get_next_item(answers: Dict[int, int]) -> Optional[PhqItem]:
    for it in PHQ_ITEMS:
        if not it.is_functional and it.item_id not in answers:
            return it
    if 10 not in answers:
        return next(i for i in PHQ_ITEMS if i.item_id == 10)
    return None


def score_phq(answers: Dict[int, int]):
    total = sum(answers.get(i, 0) for i in range(1, 10))
    band = "unknown"
    for lo, hi, label in PHQ_BANDS:
        if lo <= total <= hi:
            band = label
            break

    item9_flag = answers.get(9, 0) > 0
    func = FUNCTIONAL_IMPAIRMENT_LABELS.get(answers.get(10)) if 10 in answers else None

    return {
        "total": total,
        "band": band,
        "item9_positive": item9_flag,
        "functional_impairment": func,
    }


# -----------------------------
# STREAMLIT APP
# -----------------------------

SYSTEM_BLURB = (
    "You are PediMinds, a warm, clear wellness assistant. "
    "You guide PHQ-9 screenings with empathy and explain results simply. "
    "If someone reports self-harm thoughts, you escalate safely. "
    "You also answer basic questions about Snowflake AI and Emotive AI."
)


def init_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = start_session()
    if "answers" not in st.session_state:
        st.session_state.answers = {}
    if "last_rag_answer" not in st.session_state:
        st.session_state.last_rag_answer = None
    if "phase" not in st.session_state:
        st.session_state.phase = "screening"


def reset_screening():
    st.session_state.session_id = start_session()
    st.session_state.answers = {}
    st.session_state.phase = "screening"
    st.session_state.last_rag_answer = None


def render_header():
    st.set_page_config(
        page_title="PediMinds — PHQ-9 Wellness Screener",
        page_icon="🩵",
        layout="centered",
    )
    st.title("🩵 PediMinds — PHQ-9 Wellness Screener")
    st.caption("A wellness check, not a diagnosis. If you feel unsafe, contact emergency services.")

    with st.sidebar:
        st.subheader("Paxalytica · AI for good")
        st.write(SYSTEM_BLURB)
        st.markdown("---")
        st.write("This demo runs fully on Streamlit Cloud.")


def render_screening():
    st.subheader("PHQ-9 Screening")

    # If finished:
    if st.session_state.phase == "done":
        res = score_phq(st.session_state.answers)
        st.success(f"Your PHQ-9 total is **{res['total']}** ({res['band']}).")

        if res["functional_impairment"]:
            st.info(f"Impact on your daily life: **{res['functional_impairment']}**.")

        if res["item9_positive"]:
            st.error("⚠️ You indicated some thoughts of self-harm. "
                     "This tool cannot keep you safe. Please reach out to a professional or emergency services.")

        st.markdown("---")
        if st.button("🔁 Start over"):
            reset_screening()
        return

    # Otherwise: still screening
    next_item = get_next_item(st.session_state.answers)
    if not next_item:
        st.session_state.phase = "done"
        st.experimental_rerun()
        return

    st.markdown(f"### Q{next_item.item_id}. {next_item.text}")

    if next_item.is_functional:
        labels = [
            "0 — Not difficult at all",
            "1 — Somewhat difficult",
            "2 — Very difficult",
            "3 — Extremely difficult",
        ]
    else:
        labels = [
            "0 — Not at all",
            "1 — Several days",
            "2 — More than half the days",
            "3 — Nearly every day",
        ]

    current = st.session_state.answers.get(next_item.item_id, 0)
    choice = st.radio("Choose one:", [0,1,2,3], index=current, format_func=lambda x: labels[x])

    if st.button("Next ➜"):
        st.session_state.answers[next_item.item_id] = int(choice)
        if get_next_item(st.session_state.answers) is None:
            st.session_state.phase = "done"
        st.experimental_rerun()


def render_rag():
    st.subheader("Ask PediMinds about AI")

    q = st.text_input("Ask something like: 'What is Snowflake Cortex?'")

    if st.button("Ask 🧠"):
        if q.strip():
            st.session_state.last_rag_answer = rag_query(q.strip(), k=2)
        else:
            st.warning("Type a question first.")

    if st.session_state.last_rag_answer:
        st.markdown("#### Answer")
        chunks = " ".join([h["chunk"] for h in st.session_state.last_rag_answer])
        st.write(chunks)

        st.caption("These are general summaries, not official documentation.")
        with st.expander("Sources"):
            for h in st.session_state.last_rag_answer:
                st.markdown(f"- **{h['source']}** → {h['chunk']}")


def main():
    init_state()
    render_header()
    tab1, tab2 = st.tabs(["🩺 Screening", "💡 AI Questions"])
    with tab1:
        render_screening()
    with tab2:
        render_rag()


if __name__ == "__main__":
    main()
