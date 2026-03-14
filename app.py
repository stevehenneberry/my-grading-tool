import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import spacy
from collections import Counter
import pandas as pd

# Now that it's in requirements.txt, we can just load it directly!
nlp = spacy.load("en_core_web_sm")

# ... (the rest of your code remains the same)

# --- 2. COLOR LOGIC FOR THE WORDCLOUD ---
def pos_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    doc = nlp(word)
    tag = doc[0].pos_ if len(doc) > 0 else ""
    if tag == "VERB": return "hsl(210, 100%, 50%)"   # Blue
    if tag == "ADJ":  return "hsl(120, 100%, 25%)"   # Green
    if tag == "ADV":  return "hsl(280, 100%, 50%)"   # Purple
    if tag == "NOUN": return "hsl(30, 100%, 50%)"    # Orange
    return "hsl(0, 0%, 70%)"                         # Gray (everything else)

# --- 3. THE USER INTERFACE ---
st.set_page_config(page_title="Oral Homework Analyzer", layout="wide")
st.title("📝 Oral Homework Analyzer Pro")

# Sidebar for student info
with st.sidebar:
    st.header("Student Info")
    name = st.text_input("Full Name", value="Student")
    email = st.text_input("Email Address")
    duration = st.number_input("Speaking Duration (Minutes)", min_value=0.1, value=2.0, step=0.1)

# Main text input
transcript = st.text_area("Paste your Microsoft Word Dictation here:", height=300)

# --- 4. THE ANALYSIS LOGIC ---
if st.button("Analyze My Homework"):
    if transcript:
        doc = nlp(transcript)
        
        # Calculations
        words = [token.text for token in doc if not token.is_punct and not token.is_space]
        word_count = len(words)
        wpm = word_count / duration
        
        # Display Stats
        st.success(f"Analysis Complete for {name}!")
        col1, col2 = st.columns(2)
        col1.metric("Words Per Minute", f"{wpm:.1f}")
        col2.metric("Total Word Count", word_count)

        st.divider()

        # Layout: Cloud on Left, Table on Right
        left_col, right_col = st.columns([2, 1])
        
        with left_col:
            st.subheader("Vocabulary Cloud")
            # Generate Cloud
            wc = WordCloud(background_color="white", width=800, height=500).generate(transcript)
            wc.recolor(color_func=pos_color_func)
            
            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)
            
            # The Color Key
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
