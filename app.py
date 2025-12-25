import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Fashion Support Chatbot", page_icon="🛍️")
st.title("🛍️ Fashion Support Chatbot ")

# ----------------------------
# Load data + build TF-IDF index
# ----------------------------
@st.cache_resource
def load_model():
    df = pd.read_csv("Womens Clothing E-Commerce Reviews.csv")

    keep_cols = ["Clothing ID","Title","Review Text","Rating","Recommended IND",
                 "Positive Feedback Count","Division Name","Department Name","Class Name"]
    df = df[keep_cols].dropna(subset=["Review Text"]).copy()
    df["Clothing ID"] = pd.to_numeric(df["Clothing ID"], errors="coerce")
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df = df.dropna(subset=["Clothing ID","Rating"]).reset_index(drop=True)

    def make_doc(row):
        title = "" if pd.isna(row["Title"]) else str(row["Title"])
        return (
            f"Clothing ID: {int(row['Clothing ID'])} | "
            f"Dept: {row['Department Name']} | Class: {row['Class Name']} | "
            f"Rating: {int(row['Rating'])} | {title} | {row['Review Text']}"
        )

    df["doc"] = df.apply(make_doc, axis=1)
    vectorizer = TfidfVectorizer(stop_words="english", max_features=50000, ngram_range=(1,2))
    X = vectorizer.fit_transform(df["doc"])
    return df, vectorizer, X

df, vectorizer, X = load_model()

# ----------------------------
# Helpers
# ----------------------------
def extract_clothing_id(q: str):
    ql = q.lower()
    m = re.search(r"(?:clothing\s*id|clothingid|id)\s*[:=#-]?\s*(\d{2,6})", ql)
    if m:
        return int(m.group(1))
    m2 = re.fullmatch(r"\s*(\d{2,6})\s*", ql)
    if m2:
        return int(m2.group(1))
    return None

def is_complaint_intent(q: str):
    ql = q.lower()
    return any(w in ql for w in ["complain", "complaint", "complaints", "issue", "problem", "bad", "worst", "return", "returned", "hate", "dislike"])

def retrieve(query, top_k=6, clothing_id=None, complaint_bias=False):
    sub_df = df
    sub_X = X

    # Filter by Clothing ID if provided
    if clothing_id is not None:
        mask = (df["Clothing ID"] == clothing_id)
        sub_df = df.loc[mask]
        sub_X = X[mask.values]
        if len(sub_df) == 0:
            return pd.DataFrame()

    # Bias toward low ratings for complaint queries
    if complaint_bias:
        low_mask = sub_df["Rating"] <= 2
        if low_mask.sum() < 10:
            low_mask = sub_df["Rating"] <= 3
        sub_df = sub_df.loc[low_mask]
        sub_X = sub_X[low_mask.values]
        if len(sub_df) == 0:
            return pd.DataFrame()

    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, sub_X).flatten()
    if sims.size == 0:
        return pd.DataFrame()

    top_idx = sims.argsort()[::-1][:top_k]
    res = sub_df.iloc[top_idx].copy()
    res["score"] = sims[top_idx]
    return res

def answer(query):
    cid = extract_clothing_id(query)
    complaint_bias = is_complaint_intent(query)

    res = retrieve(query, top_k=6, clothing_id=cid, complaint_bias=complaint_bias)

    if len(res) == 0 and cid is not None:
        return f"I couldn’t find enough matching reviews for Clothing ID {cid}. Try another ID or ask a general question."
    if len(res) == 0:
        return "I couldn’t find relevant reviews. Try: 'Clothing ID 1080 complaints' or 'Clothing ID 1080 true to size'."

    avg = res["Rating"].mean()
    rec = res["Recommended IND"].mean()*100

    heading = "Complaint-focused matches" if complaint_bias else "Top review matches"
    snippet = "\n".join([f"- (⭐{int(r.Rating)}) {str(r['Review Text'])[:140]}..." for _, r in res.head(3).iterrows()])

    return f"Avg rating (matches): {avg:.1f}/5 | Recommended: {rec:.0f}%\n\n{heading}:\n{snippet}"

# ----------------------------
# UI: Chat history
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [{"role":"assistant","content":"Ask me about fit, quality, or complaints. Try: 'Clothing ID 1080 true to size?' (Type 'help' for commands)"}]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

# ----------------------------
# UI: Input + Commands
# ----------------------------
prompt = st.chat_input("Type your question…")

if prompt:
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user"):
        st.write(prompt)

    p = prompt.strip()

    # Commands
    if p.lower() in ["help", "h", "?"]:
        bot = (
            "✅ Try:\n"
            "- Is Clothing ID 1080 true to size?\n"
            "- Clothing ID 1080 complaints\n"
            "- What are the complaints for Clothing ID 1081?\n"
            "- top dresses\n"
            "- top tops\n"
            "- departments"
        )

    elif p.lower() == "departments":
        depts = sorted(df["Department Name"].dropna().unique().tolist())
        bot = "Available departments:\n- " + "\n- ".join(depts)

    elif p.lower().startswith("top"):
        dept = "Dresses"
        if "tops" in p.lower(): dept = "Tops"
        if "jackets" in p.lower(): dept = "Jackets"
        if "intimates" in p.lower(): dept = "Intimates"
        if "bottoms" in p.lower(): dept = "Bottoms"

        sub = df[df["Department Name"] == dept]
        g = (sub.groupby("Clothing ID")
             .agg(avg_rating=("Rating","mean"), n_reviews=("Rating","count"))
             .reset_index())
        g = g[g["n_reviews"] >= 15].copy()
        g = g.sort_values(["avg_rating","n_reviews"], ascending=False).head(5)

        if len(g) == 0:
            bot = f"No top list found for {dept} (maybe not enough reviews)."
        else:
            lines = [f"Top rated {dept} (min 15 reviews):"]
            for _, r in g.iterrows():
                lines.append(f"- ID {int(r['Clothing ID'])}: {r['avg_rating']:.2f}/5 ({int(r['n_reviews'])} reviews)")
            bot = "\n".join(lines)

    else:
        bot = answer(p)

    st.session_state.messages.append({"role":"assistant","content":bot})
    with st.chat_message("assistant"):
        st.write(bot)
