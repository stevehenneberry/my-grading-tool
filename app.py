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

# --- 3. HELPER: VOCABULARY COMPLEXITY SCORE ---
def get_complexity_indicator(ttr, avg_sentence_length, lexical_density, unique_connectives, avg_word_length):
    """
    Returns a (score, label, range_label) tuple based on a simple points system.
    Each metric contributes 0-3 points for a max of 15.
    """
    score = 0

    # TTR
    if ttr > 0.65:       score += 3
    elif ttr > 0.50:     score += 2
    elif ttr > 0.40:     score += 1

    # Average sentence length
    if avg_sentence_length > 15:    score += 3
    elif avg_sentence_length > 10:  score += 2
    elif avg_sentence_length > 7:   score += 1

    # Lexical density
    if lexical_density > 0.55:      score += 3
    elif lexical_density > 0.45:    score += 2
    elif lexical_density > 0.35:    score += 1

    # Unique connectives
    if unique_connectives > 5:      score += 3
    elif unique_connectives >= 3:   score += 2
    elif unique_connectives >= 1:   score += 1

    # Average word length (characters)
    if avg_word_length > 5.5:       score += 3
    elif avg_word_length > 4.8:     score += 2
    elif avg_word_length > 4.0:     score += 1

    if score <= 3:    return score, "A1",    "A1 range"
    elif score <= 6:  return score, "A2",    "A2 range"
    elif score <= 9:  return score, "B1",    "B1 range"
    elif score <= 12: return score, "B2",    "B2 range"
    else:             return score, "C1–C2", "C1–C2 range"

# --- 4. THE USER INTERFACE ---
st.set_page_config(page_title="Oral Homework Analyzer", layout="wide")
st.title("📝 Oral Homework Analyzer Pro")

with st.sidebar:
    st.header("Student Info")
    name = st.text_input("Full Name", value="Student")
    duration = st.number_input("Speaking Duration (Minutes)", min_value=0.1, value=2.0, step=0.1)

transcript = st.text_area("Paste your Microsoft Word Dictation here:", height=300)

# --- 5. THE ANALYSIS ---
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

        # Lexical Density
        content_pos = {"NOUN", "VERB", "ADJ", "ADV"}
        content_words = [t for t in doc if t.pos_ in content_pos and not t.is_punct and not t.is_space]
        lexical_density = len(content_words) / word_count if word_count > 0 else 0

        # Average Word Length
        avg_word_length = sum(len(w) for w in words) / word_count if word_count > 0 else 0

        # Connectives (needed for complexity score)
        found_connectives = []
        for connective in CONNECTIVES:
            if f" {connective} " in f" {transcript_lower} ":
                count = transcript_lower.count(connective)
                found_connectives.append((connective, count))
        found_connectives.sort(key=lambda x: x[1], reverse=True)
        unique_connective_count = len(found_connectives)

        # Vocabulary Complexity Indicator
        complexity_score, complexity_label, complexity_range = get_complexity_indicator(
            ttr, avg_sentence_length, lexical_density, unique_connective_count, avg_word_length
        )

        # --- DISPLAY: TOP METRICS ROW ---
        st.success(f"Analysis Complete for {name}!")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Words Per Minute", f"{wpm:.1f}")
        col2.metric("Total Word Count", word_count)
        col3.metric("Vocabulary Diversity", f"{ttr:.2f}",
                    help="Type-Token Ratio: unique words ÷ total words. Closer to 1.0 = more varied vocabulary.")
        col4.metric("Avg. Sentence Length", f"{avg_sentence_length:.1f} words")
        col5.metric("Complexity Indicator", complexity_label,
                    help="Based on vocabulary and sentence patterns only — not a substitute for a formal CEFR test.")

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

            # Row 1: Lexical Density | Filler Words | Connective Words
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
                st.metric("Unique Connectives Used", unique_connective_count)

                if found_connectives:
                    df_conn = pd.DataFrame(found_connectives, columns=["Connective", "Count"])
                    st.table(df_conn)
                else:
                    st.warning("No connectives found. Try linking your ideas with words like 'however' or 'because.'")

            st.divider()

            # Row 2: Repeated Phrase Detector (full width)
            st.subheader("🔁 Repeated Phrase Detector")
            st.caption("""
                Repeating the same short phrase too often can make speech sound less fluent.
                The table below shows any two- or three-word combinations you used more than once.
                Try replacing some of these with alternative expressions to add variety.
            """)

            # Generate bigrams and trigrams, keeping stop words for readability
            # but filtering out phrases made up entirely of stop/punctuation tokens
            all_tokens = [token for token in doc if not token.is_punct and not token.is_space]

            bigrams = []
            for i in range(len(all_tokens) - 1):
                t1, t2 = all_tokens[i], all_tokens[i + 1]
                # Skip phrases where both words are stop words
                if not (t1.is_stop and t2.is_stop):
                    bigrams.append(f"{t1.text.lower()} {t2.text.lower()}")

            trigrams = []
            for i in range(len(all_tokens) - 2):
                t1, t2, t3 = all_tokens[i], all_tokens[i + 1], all_tokens[i + 2]
                # Skip phrases where all three words are stop words
                if not (t1.is_stop and t2.is_stop and t3.is_stop):
                    trigrams.append(f"{t1.text.lower()} {t2.text.lower()} {t3.text.lower()}")

            repeated_bigrams  = [(p, c) for p, c in Counter(bigrams).items()  if c >= 2]
            repeated_trigrams = [(p, c) for p, c in Counter(trigrams).items() if c >= 2]

            all_repeated = sorted(repeated_bigrams + repeated_trigrams, key=lambda x: x[1], reverse=True)

            if all_repeated:
                df_phrases = pd.DataFrame(all_repeated, columns=["Phrase", "Times Used"])
                st.table(df_phrases)
            else:
                st.success("No repeated phrases detected — great variety!")

            st.divider()

            # Row 3: Vocabulary Complexity Indicator (full width)
            st.subheader("📊 Vocabulary Complexity Indicator")
            st.caption("""
                This is a rough estimate of your vocabulary and grammar complexity based on
                five measurable patterns in your speech: vocabulary variety, sentence length,
                word richness, use of connectives, and average word length.
                It is **not** a formal CEFR test — think of it as a helpful pointer, not an
                official result. A full CEFR assessment requires testing all four language
                skills across a range of tasks.
            """)

            ind_col1, ind_col2 = st.columns([1, 2])

            with ind_col1:
                st.metric("Estimated Level", complexity_label,
                          help=f"Score: {complexity_score}/15")
                st.progress(complexity_score / 15)

            with ind_col2:
                st.markdown(f"""
                | Factor | Your Score |
                |---|---|
                | Vocabulary Variety (TTR) | {ttr:.2f} |
                | Avg. Sentence Length | {avg_sentence_length:.1f} words |
                | Lexical Density | {lexical_density:.2f} |
                | Unique Connectives | {unique_connective_count} |
                | Avg. Word Length | {avg_word_length:.1f} characters |
                """)

    else:
        st.error("Please paste your transcript first!")
