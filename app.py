import streamlit as st

# 1. The Title of your app
st.title("📝 Oral Homework Analyzer")

# 2. The Input Fields
name = st.text_input("Your Name")
email = st.text_input("Your Email Address")
transcript = st.text_area("Paste your Microsoft Word Dictation here:", height=300)

# 3. The "Action" Button
if st.button("Analyze My Homework"):
    if transcript:
        # Simple math for now
        word_count = len(transcript.split())
        st.success(f"Great job, {name}!")
        st.metric("Total Word Count", word_count)
    else:
        st.error("Please paste some text first!")
