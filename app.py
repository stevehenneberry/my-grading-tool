import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import spacy
import re

# Load the grammar "brain"
nlp = spacy.load("en_core_web_sm")

st.title("📝 Oral Homework Analyzer Pro")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Settings")
    name = st.text_input("Student Name")
    duration = st.number_input("Speaking duration (minutes)", min_value=0.1, value=2.0)

# --- COLOR LOGIC ---
def pos_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    doc = nlp(word)
    tag = doc[0].pos_ if len(doc) > 0 else ""
    if tag == "VERB": return "hsl(210, 100%, 50%)"   # Blue
    if tag == "ADJ":  return "hsl(120, 100%, 25%)"   # Green
    if tag == "NOUN": return "hsl(30, 100%, 50%)"    # Orange
    return "hsl(0, 0%, 50%)"                         # Gray for others

# --- MAIN APP ---
transcript = st.text_area("Paste your transcript here:", height=250)

if st.button("Analyze"):
    if transcript:
        # Basic Stats
        words = transcript.split()
        wpm = len(words) / duration
        
        col1, col2 = st.columns(2)
        col1.metric("Words Per Minute", f"{wpm:.1f}")
        col2.metric("Total Words", len(words))

        # Color-Coded Word Cloud
        st.subheader("Vocabulary by Part of Speech")
        st.caption("🔵 Verbs | 🟢 Adjectives | 🟠 Nouns")
        
        wc = WordCloud(background_color="white", width=800, height=400).generate(transcript)
        wc.recolor(color_func=pos_color_func)
        
        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
