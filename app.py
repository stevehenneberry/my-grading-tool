import streamlit as st
import spacy
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd

# --- 1. LOAD THE NLP MODEL ---
@st.cache_resource
def load_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except:
        return None

nlp = load_nlp()

if nlp is None:
    st.warning("The grammar engine is warming up. Please wait 30 seconds and refresh the page!")
    st.stop()

# --- 2. THE USER INTERFACE ---
st.set_page_config(page_title="Oral Homework Analyzer", layout="wide")
st.title("📝 Oral Homework Analyzer Pro")

with st.sidebar:
    st.header("Student Info")
    name = st.text_input("Full Name", value="Student")
    duration = st.number_input("Speaking Duration (Minutes)", min_value=0.1, value=2.0, step=0.1)

transcript = st.text_area("Paste your Microsoft Word Dictation here:", height=300)

# --- 3. THE ANALYSIS ---
if st.button("Analyze My Homework"):
    if transcript:
        doc = nlp(transcript)

        # Build a word -> POS lookup from the full transcript (context-aware)
        word_pos_map = {}
        for token in doc:
            if not token.is_punct and not token.is_space:
                word_pos_map[token.text.lower()] = token.pos_

        # Color function now uses context-aware POS tags
        def pos_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            tag = word_pos_map.get(word.lower(), "")
            if tag == "VERB": return "hsl(210, 100%, 50%)"   # Blue
            if tag == "ADJ":  return "hsl(120, 100%, 25%)"   # Green
            if tag == "ADV":  return "hsl(280, 100%, 50%)"   # Purple
            if tag == "NOUN": return "hsl(30, 100%, 50%)"    # Orange
            return "hsl(0, 0%, 70%)"                         # Gray

        words = [token.text for token in doc if not token.is_punct and not token.is_space]
        word_count = len(words)
        wpm = word_count / duration

        st.success(f"Analysis Complete for {name}!")
        col1, col2 = st.columns(2)
        col1.metric("Words Per Minute", f"{wpm:.1f}")
        col2.metric("Total Word Count", word_count)

        st.divider()

        left_col, right_col = st.columns([2, 1])

        with left_col:
            st.subheader("Vocabulary Cloud")
            wc = WordCloud(background_color="white", width=800, height=500).generate(transcript)
            wc.recolor(color_func=pos_color_func)

            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

            st.markdown("""
            **Color Key:**
            * 🔵 **Blue**: Verbs (Actions)
            * 🟢 **Green**: Adjectives (Descriptions)
            * 🟣 **Purple**: Adverbs (Details about actions)
            * 🟠 **Orange**: Nouns (People/Places/Things)
            * ⚪ **Gray**: Other words
            """)

        with right_col:
            st.subheader("Top 5 Adjectives")
            adjectives = [token.text.lower() for token in doc if token.pos_ == "ADJ"]
            if adjectives:
                adj_counts = Counter(adjectives).most_common(5)
                df = pd.DataFrame(adj_counts, columns=["Word", "Count"])
                st.table(df)
            else:
                st.info("No adjectives found. Try using more descriptive language!")
    else:
        st.error("Please paste your transcript first!")
