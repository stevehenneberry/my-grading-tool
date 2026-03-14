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

# --- 2. REFERENCE WORD LISTS ---
FILLER_WORDS = {
    "um", "uh", "er", "ah", "like", "basically", "literally",
    "actually", "you know", "i mean", "right", "okay", "so",
    "well", "anyway", "kind of", "sort of"
}

CONNECTIVES = {
    "however", "therefore", "although", "because", "moreover",
    "furthermore", "nevertheless", "consequently", "meanwhile",
    "additionally", "otherwise", "similarly", "in contrast",
    "as a result", "for example", "for instance", "in addition",
    "on the other hand", "in conclusion", "to summarize",
    "first", "second", "third", "finally", "next", "then",
    "after", "before", "while", "since", "unless", "despite"
}

# --- 3. THE USER INTERFACE ---
st.set_page_config(page_title="Oral Homework Analyzer", layout="wide")
st.title("📝 Oral Homework Analyzer Pro")

with st.sidebar:
    st.header("Student Info")
    name = st.text_input("Full Name", value="Student")
    duration = st.number_input("Speaking Duration (Minutes)", min_value=0.1, value=2.0, step=0.1)

transcript = st.text_area("Paste your Microsoft Word Dictation here:", height=300)

# --- 4. THE ANALYSIS ---
if st.button("Analyze My Homework"):
    if transcript:
        doc = nlp(transcript)
        transcript_lower = transcript.lower()

        # Build a word -> POS lookup from the full transcript (context-aware)
        word_pos_map = {}
        for token in doc:
            if not token.is_punct and not token.is_space:
                word_pos_map[token.text.lower()] = token.pos_

        # Color function uses context-aware POS tags
        def pos_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            tag = word_pos_map.get(word.lower(), "")
            if tag == "VERB": return "hsl(210, 100%, 50%)"   # Blue
            if tag == "ADJ":  return "hsl(120, 100%, 25%)"   # Green
            if tag == "ADV":  return "hsl(280, 100%, 50%)"   # Purple
            if tag == "NOUN": return "hsl(30, 100%, 50%)"    # Orange
            return "hsl(0, 0%, 70%)"                         # Gray

        # --- CORE STATS ---
        words = [token.text for token in doc if not token.is_punct and not token.is_space]
        word_count = len(words)
        wpm = word_count / duration

        # Vocabulary Diversity (Type-Token Ratio)
        unique_words = set(w.lower() for w in words)
        ttr = len(unique_words) / word_count if word_count > 0 else 0

        # Average Sentence Length
        sentences = list(doc.sents)
        sentence_count = len(sentences)
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0

        # --- DISPLAY: TOP METRICS ROW ---
        st.success(f"Analysis Complete for {name}!")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Words Per Minute", f"{wpm:.1f}")
        col2.metric("Total Word Count", word_count)
        col3.metric("Vocabulary Diversity", f"{ttr:.2f}",
                    help="Type-Token Ratio: unique words ÷ total words. Closer to 1.0 = more varied vocabulary.")
        col4.metric("Avg. Sentence Length", f"{avg_sentence_length:.1f} words")

        st.divider()

        # --- DISPLAY: WORD CLOUD + ADJECTIVES/VERBS TABLES ---
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
            # Top 5 Adjectives
            st.subheader("Top 5 Adjectives")
            adjectives = [token.text.lower() for token in doc if token.pos_ == "ADJ"]
            if adjectives:
                adj_counts = Counter(adjectives).most_common(5)
                df_adj = pd.DataFrame(adj_counts, columns=["Word", "Count"])
                st.table(df_adj)
            else:
                st.info("No adjectives found. Try using more descriptive language!")

            st.divider()

            # Top 5 Verbs
            st.subheader("Top 5 Verbs")
            verbs = [token.lemma_.lower() for token in doc if token.pos_ == "VERB"]
            if verbs:
                verb_counts = Counter(verbs).most_common(5)
                df_verbs = pd.DataFrame(verb_counts, columns=["Word", "Count"])
                st.table(df_verbs)
            else:
                st.info("No verbs found.")

        st.divider()

        # --- DISPLAY: DETAILED ANALYSIS (COLLAPSIBLE) ---
        with st.expander("🔍 Detailed Analysis"):

            detail_col1, detail_col2, detail_col3 = st.columns(3)

            # --- LEXICAL DENSITY ---
            with detail_col1:
                st.subheader("Lexical Density")
                st.caption("""
                Lexical density measures how much of your speech is made up of
                meaningful content words — nouns, verbs, adjectives, and adverbs —
                versus small connecting words like "the," "a," or "is."
                A higher score means your speech is richer and more information-packed.
                Most fluent speakers score between 0.40 and 0.60.
                """)
                content_pos = {"NOUN", "VERB", "ADJ", "ADV"}
                content_words = [t for t in doc if t.pos_ in content_pos and not t.is_punct and not t.is_space]
                lexical_density = len(content_words) / word_count if word_count > 0 else 0
                st.metric("Lexical Density Score", f"{lexical_density:.2f}")

            # --- FILLER WORDS ---
            with detail_col2:
                st.subheader("Filler Words")
                st.caption("""
                Filler words are sounds and phrases like "um," "uh," "like," or "you know"
                that we use to fill pauses while thinking. A few fillers are completely normal,
                but too many can make speech harder to follow. Aim to keep fillers below
                5% of your total words.
                """)
                found_fillers = []
                for filler in FILLER_WORDS:
                    count = transcript_lower.split().count(filler) if " " not in filler else transcript_lower.count(filler)
                    if count > 0:
                        found_fillers.append((filler, count))

                found_fillers.sort(key=lambda x: x[1], reverse=True)
                total_fillers = sum(c for _, c in found_fillers)
                filler_pct = (total_fillers / word_count * 100) if word_count > 0 else 0

                st.metric("Total Filler Words", total_fillers,
                          help=f"{filler_pct:.1f}% of your total words")

                if found_fillers:
                    df_fillers = pd.DataFrame(found_fillers, columns=["Filler", "Count"])
                    st.table(df_fillers)
                else:
                    st.success("No filler words detected — great job!")

            # --- CONNECTIVES ---
            with detail_col3:
                st.subheader("Connective Words")
                st.caption("""
                Connective words and phrases — like "however," "because," "therefore,"
                and "in addition" — link your ideas together and show how they relate.
                Using a variety of connectives makes your speech sound more organized
                and academic. The list below shows which ones you used.
                """)
                found_connectives = []
                for connective in CONNECTIVES:
                    if f" {connective} " in f" {transcript_lower} ":
                        count = transcript_lower.count(connective)
                        found_connectives.append((connective, count))

                found_connectives.sort(key=lambda x: x[1], reverse=True)
                total_connectives = len(found_connectives)

                st.metric("Unique Connectives Used", total_connectives)

                if found_connectives:
                    df_conn = pd.DataFrame(found_connectives, columns=["Connective", "Count"])
                    st.table(df_conn)
                else:
                    st.warning("No connectives found. Try linking your ideas with words like 'however' or 'because.'")

    else:
        st.error("Please paste your transcript first!")
