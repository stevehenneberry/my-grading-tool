import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re

st.title("📝 Oral Homework Analyzer Pro")

# --- SIDEBAR INPUTS ---
with st.sidebar:
    st.header("Settings")
    name = st.text_input("Student Name")
    email = st.text_input("Student Email")
    duration = st.number_input("How many minutes did you speak?", min_value=0.1, value=1.0)

# --- MAIN INPUT ---
transcript = st.text_area("Paste your Microsoft Word Dictation here:", height=300)

if st.button("Analyze My Homework"):
    if transcript:
        # --- CALCULATIONS ---
        words = transcript.split()
        word_count = len(words)
        
        # Count sentences using punctuation
        sentences = re.split(r'[.!?]+', transcript)
        sentences = [s for s in sentences if len(s.strip()) > 0]
        sentence_count = len(sentences)
        
        wpm = word_count / duration
        avg_sentence_len = word_count / sentence_count if sentence_count > 0 else 0

        # --- DISPLAY METRICS ---
        st.success(f"Analysis Complete for {name}!")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Words", word_count)
        col2.metric("Words Per Minute", f"{wpm:.1f}")
        col3.metric("Avg. Sentence Length", f"{avg_sentence_len:.1f}")

        # --- WORD CLOUD ---
        st.subheader("Your Vocabulary Visualized")
        # Generate the cloud
        wc = WordCloud(background_color="white", colormap="viridis", width=800, height=400).generate(transcript)
        
        # Display the cloud using Matplotlib
        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
        
    else:
        st.error("Please paste some text first!")
