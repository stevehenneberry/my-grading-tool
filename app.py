import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import spacy
from collections import Counter
import pandas as pd

# Load the grammar "brain"
nlp = spacy.load("en_core_web_sm")

st.title("📝 Oral Homework Analyzer Pro")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Settings")
    name = st.text_input("Student Name", value="Student")
    duration = st.number_input("Speaking duration (minutes)", min_value=0.1, value=2.0)

# --- COLOR LOGIC FOR WORDCLOUD ---
def pos_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    doc = nlp(word)
    tag = doc[0].pos_ if len(doc) > 0 else ""
    if tag == "VERB": return "hsl(210, 100%, 50%)"   # Blue
    if tag == "ADJ":  return "hsl(120, 100%, 25%)"   # Green
    if tag == "ADV":  return "hsl(280, 100%, 50%)"   # Purple (Adverbs!)
    if tag == "NOUN": return "hsl(30, 100%, 50%)"    # Orange
    return "hsl(0, 0%, 70%)"                         # Light Gray for others

# --- MAIN APP ---
transcript = st.text_area("Paste your transcript here:", height=250)

if st.button("Analyze"):
    if transcript:
        doc = nlp(transcript)
        
        # 1. Basic Stats
        words = [token.text for token in doc if not token.is_punct]
        wpm = len(words) / duration
        
        col1, col2 = st.columns(2)
        col1.metric("Words Per Minute", f"{wpm:.1f}")
        col2.metric("Total Words", len(words))

        # 2. Extract Adjectives for the Table
        adjectives = [token.text.lower() for token in doc if token.pos_ == "ADJ"]
        adj_counts = Counter(adjectives).most_common(5)
        
        # 3. Layout
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
            
            # --- COLOR KEY ---
            st.markdown("""
            **Color Key:**
            * 🔵 **Blue**: Verbs (Actions)
            * 🟢 **Green**: Adjectives (Descriptions)
            * 🟣 **Purple**: Adverbs (Adding detail to actions!)
            * 🟠 **Orange**: Nouns (Things/People)
            * ⚪ **Gray**: Other words
            """)

        with right_col:
            st.subheader("Top 5 Adjectives")
            if adj_counts:
                df = pd.DataFrame(adj_counts, columns=["Adjective", "Count"])
                st.table(df)
            else:
                st.info("No adjectives found yet!")

    else:
        st.error("Please paste some text first!")
