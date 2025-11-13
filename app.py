import uuid
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Embedding model (for RAG)
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
    PhqItem(8, "Moving or speaking slowly, or being very fidgety or restless", "psychomotor"),
    PhqItem(9, "Thoughts that you would be better off dead or of hurting yourself", "self_harm"),
    PhqItem(10, "How difficult have these problems made it for you at work, home, or with other people?", "functional", True),
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
# SMALL KNOWLEDGE BASE (RAG)
# -----------------------------

KNOWLEDGE_BASE = [
    {
        "topic": "snowflake",
        "chunk": "Snowflake Cortex offers serverless access to large language models and embeddings "
                 "close to your data, so teams can build AI features without moving data out of Snowflake.",
        "source": "Snowflake docs (summary)",
    },
    {
        "topic": "snowflake",
        "chunk": "AISQL lets you call AI tasks like summarize or classify directly from SQL. "
                 "Cortex Analyst enables natural language questions over tables and documents.",
        "source": "Snowflake docs (summary)",
    },
    {
        "topic": "emotive_ai",
        "chunk": "Emotive or affective AI tries to estimate feelings from voice, text, or expressions "
                 "so systems can respond in a more caring and supportive way.",
        "source": "Affective computing overview",
    },
    {
        "topic": "emotive_ai",
        "chunk": "Ethical emotive AI should be transparent, avoid stereotypes, reduce bias, "
                 "and always keep a human in the loop for important decisions.",
        "source": "Ethics notes",
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
        # fallback if no embeddings: just first k notes
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
    # items 1..9 first
    for it in PHQ_ITEMS:
        if not it.is_functional and it.item_id not in answers:
            return it
    # then functional (10)
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


def therapist_search_url(zip_code: str) -> str:
    """
    Build a Psychology Today therapist search link.
    This does NOT contact anyone, it just opens a directory page.
    """
    base = "https://www.psychologytoday.com/us/therapists"
    zip_clean = (zip_code or "").strip()
    if zip_clean:
        # Psychology Today supports ?search=ZIP style queries
        return f"{base}?search={zip_clean}"
    # fallback: generic US therapists page
    return base


# -----------------------------
# STREAMLIT APP STATE & HEADER
# -----------------------------

SYSTEM_BLURB = (
    "You are PediMinds, a warm, clear wellness assistant (not a doctor). "
    "You guide PHQ-9 mood check-ins, explain results in simple language, "
    "and gently point people toward real human help when needed. "
    "You can also answer short questions about Snowflake AI and emotive AI."
)


def init_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = start_session()
    if "answers" not in st.session_state:
        st.session_state.answers = {}
    if "phase" not in st.session_state:
        st.session_state.phase = "intro"   # intro → screening → done
    if "last_rag_answer" not in st.session_state:
        st.session_state.last_rag_answer = None
    if "nickname" not in st.session_state:
        st.session_state.nickname = ""
    if "feeling_word" not in st.session_state:
        st.session_state.feeling_word = ""
    if "zip_code" not in st.session_state:
        st.session_state.zip_code = ""


def reset_screening():
    st.session_state.session_id = start_session()
    st.session_state.answers = {}
    st.session_state.phase = "intro"
    st.session_state.last_rag_answer = None
    st.session_state.feeling_word = ""


def render_header():
    st.set_page_config(
        page_title="PediMinds — PHQ-9 Wellness Screener",
        page_icon="🩵",
        layout="centered",
    )
    st.title("🩵 PediMinds — PHQ-9 Wellness Screener")
    st.caption(
        "This is a **wellness check**, not a diagnosis. "
        "If you ever feel unsafe or in crisis, please contact emergency services or a trusted professional."
    )

    with st.sidebar:
        st.subheader("Paxalytica · AI for good")
        st.write(SYSTEM_BLURB)
        st.markdown("---")
        st.write(
            "PediMinds is a prototype and does **not** store your data in a database or contact anyone for you. "
            "It simply runs in your browser to help you reflect and learn."
        )


# -----------------------------
# INTRO / CHAT-LIKE WARMUP
# -----------------------------

def render_intro():
    st.subheader("Let’s check in together 💬")

    st.write(
        "Hi, I’m **PediMinds**. I’m not a doctor, but I can help you walk through a short mood check "
        "that many clinicians use. We’ll go slowly, one question at a time."
    )
    st.write(
        "First, a couple of quick questions to make this feel a bit more personal. "
        "You can skip anything you don’t want to answer."
    )

    nickname = st.text_input("What name or nickname would you like me to use (optional)?", value=st.session_state.nickname)
    feeling = st.text_input("If you had to describe how you feel **today** in one or two words, what would you say?",
                            value=st.session_state.feeling_word)
    zip_code = st.text_input("ZIP code (optional, US only, used to suggest therapist directories later):",
                             value=st.session_state.zip_code)

    st.session_state.nickname = nickname.strip()
    st.session_state.feeling_word = feeling.strip()
    st.session_state.zip_code = zip_code.strip()

    if st.button("Start wellness check ➜"):
        st.session_state.phase = "screening"
        st.rerun()


# -----------------------------
# SCREENING TAB
# -----------------------------

def render_screening():
    st.subheader("PHQ-9 Screening")

    # Intro phase: chatty warm-up
    if st.session_state.phase == "intro":
        render_intro()
        return

    # Done phase: show results
    if st.session_state.phase == "done":
        res = score_phq(st.session_state.answers)
        total = res["total"]
        band = res["band"].capitalize()
        item9_flag = res["item9_positive"]
        func = res["functional_impairment"]

        name = st.session_state.nickname or "there"

        st.success(f"{name}, your PHQ-9 total is **{total}** ({band}).")

        if func:
            st.info(f"Impact on your daily life: **{func}**.")

        # Safety and next steps
        st.markdown("---")
        st.markdown("### What this means (and what it doesn’t)")

        st.write(
            "This score is a **screening result**, not a full diagnosis. "
            "It can be a useful starting point for a conversation with a doctor, therapist, or counselor."
        )

        if item9_flag:
            st.error(
                "⚠️ You indicated some **self-harm thoughts**. "
                "This app cannot keep you safe in an emergency.\n\n"
                "- If you are in immediate danger, call your local emergency number.\n"
                "- In the U.S., you can call or text **988** (Suicide & Crisis Lifeline).\n"
            )
        else:
            st.write(
                "If these feelings have been strong, long-lasting, or are worrying you, "
                "consider reaching out to a professional for support."
            )

        # Local therapist directory suggestion (non-automatic)
        st.markdown("#### Find professional support")
        url = therapist_search_url(st.session_state.zip_code)
        if st.session_state.zip_code:
            st.write(
                f"If you’d like to look for a therapist near **{st.session_state.zip_code}**, "
                f"you can search an online directory like Psychology Today:"
            )
        else:
            st.write(
                "You can search for therapists in your area using online directories. "
                "For example:"
            )
        st.markdown(f"- [Search therapists on Psychology Today]({url})")

        st.caption(
            "PediMinds does **not** contact any therapist for you and does not share your answers with anyone. "
            "These links are just for you to explore on your own."
        )

        st.markdown("---")
        if st.button("🔁 Start over"):
            reset_screening()
        return

    # Still screening: ask next PHQ item
    next_item = get_next_item(st.session_state.answers)
    if not next_item:
        st.session_state.phase = "done"
        st.rerun()
        return

    if not next_item.is_functional:
        st.write(
            "Over the **last 2 weeks**, how often have you been bothered by the following problem?"
        )

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
    choice = st.radio(
        "Choose one option:",
        [0, 1, 2, 3],
        index=current,
        format_func=lambda x: labels[x],
    )

    if st.button("Next ➜"):
        st.session_state.answers[next_item.item_id] = int(choice)
        if get_next_item(st.session_state.answers) is None:
            st.session_state.phase = "done"
        st.rerun()


# -----------------------------
# AI QUESTIONS TAB
# -----------------------------

def render_rag():
    st.subheader("Ask PediMinds about AI")

    st.write(
        "Here you can ask simple questions about **Snowflake AI** or **emotive/affective AI**. "
        "Answers come from a small built-in note base, not from the internet."
    )

    q = st.text_input("Ask something like: *What is Snowflake Cortex?*")

    if st.button("Ask 🧠"):
        if q.strip():
            st.session_state.last_rag_answer = rag_query(q.strip(), k=2)
        else:
            st.warning("Please type a question first.")

    if st.session_state.last_rag_answer:
        st.markdown("#### Answer")
        chunks = " ".join([h["chunk"] for h in st.session_state.last_rag_answer])
        st.write(chunks)

        st.caption(
            "These are general summaries only. They do not replace official documentation, "
            "legal advice, or professional guidance."
        )
        with st.expander("Sources"):
            for h in st.session_state.last_rag_answer:
                st.markdown(f"- **{h['source']}** → {h['chunk']}")


# -----------------------------
# MAIN
# -----------------------------

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
