import io
import datetime
import streamlit as st
import spacy
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, HRFlowable
)

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
    score = 0
    if ttr > 0.65:                  score += 3
    elif ttr > 0.50:                score += 2
    elif ttr > 0.40:                score += 1
    if avg_sentence_length > 15:    score += 3
    elif avg_sentence_length > 10:  score += 2
    elif avg_sentence_length > 7:   score += 1
    if lexical_density > 0.55:      score += 3
    elif lexical_density > 0.45:    score += 2
    elif lexical_density > 0.35:    score += 1
    if unique_connectives > 5:      score += 3
    elif unique_connectives >= 3:   score += 2
    elif unique_connectives >= 1:   score += 1
    if avg_word_length > 5.5:       score += 3
    elif avg_word_length > 4.8:     score += 2
    elif avg_word_length > 4.0:     score += 1

    if score <= 3:    return score, "A1"
    elif score <= 6:  return score, "A2"
    elif score <= 9:  return score, "B1"
    elif score <= 12: return score, "B2"
    else:             return score, "C1–C2"

# --- 4. HELPER: GENERATE PDF REPORT ---
def generate_pdf_report(name, duration, wpm, word_count, ttr, avg_sentence_length,
                         lexical_density, avg_word_length,
                         adj_counts, verb_counts,
                         found_fillers, total_fillers, filler_pct,
                         found_connectives, unique_connective_count,
                         all_repeated,
                         complexity_score, complexity_label,
                         wc_image_buffer):

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        rightMargin=0.75 * inch, leftMargin=0.75 * inch,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch
    )

    styles = getSampleStyleSheet()
    story = []

    BRAND_BLUE   = colors.HexColor("#1a73e8")
    LIGHT_GRAY   = colors.HexColor("#f5f5f5")
    MID_GRAY     = colors.HexColor("#cccccc")
    DARK_GRAY    = colors.HexColor("#333333")

    title_style = ParagraphStyle("ReportTitle", parent=styles["Title"],
                                 fontSize=22, textColor=BRAND_BLUE, spaceAfter=4)
    subtitle_style = ParagraphStyle("Subtitle", parent=styles["Normal"],
                                    fontSize=11, textColor=DARK_GRAY, spaceAfter=2)
    section_style = ParagraphStyle("Section", parent=styles["Heading2"],
                                   fontSize=13, textColor=BRAND_BLUE,
                                   spaceBefore=14, spaceAfter=6)
    body_style = ParagraphStyle("Body", parent=styles["Normal"],
                                fontSize=10, textColor=DARK_GRAY, spaceAfter=4)
    caption_style = ParagraphStyle("Caption", parent=styles["Normal"],
                                   fontSize=8, textColor=colors.HexColor("#777777"),
                                   spaceAfter=6, fontName="Helvetica-Oblique")

    def section_table_style(col_widths):
        return TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0), BRAND_BLUE),
            ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, 0), 10),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
            ("FONTSIZE",    (0, 1), (-1, -1), 10),
            ("GRID",        (0, 0), (-1, -1), 0.5, MID_GRAY),
            ("TOPPADDING",  (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ])

    # ---- HEADER ----
    story.append(Paragraph("📝 Oral Homework Analyzer", title_style))
    story.append(Paragraph(f"Student: <b>{name}</b>", subtitle_style))
    story.append(Paragraph(
        f"Date: {datetime.date.today().strftime('%B %d, %Y')}  |  Speaking Duration: {duration} min",
        subtitle_style
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=BRAND_BLUE, spaceAfter=12))

    # ---- CORE STATISTICS ----
    story.append(Paragraph("Core Statistics", section_style))
    stats_data = [
        ["Metric", "Result"],
        ["Words Per Minute",          f"{wpm:.1f}"],
        ["Total Word Count",          str(word_count)],
        ["Vocabulary Diversity (TTR)", f"{ttr:.2f}"],
        ["Avg. Sentence Length",      f"{avg_sentence_length:.1f} words"],
        ["Lexical Density",           f"{lexical_density:.2f}"],
        ["Avg. Word Length",          f"{avg_word_length:.1f} characters"],
        ["Complexity Indicator",      complexity_label],
    ]
    stats_table = Table(stats_data, colWidths=[3.5 * inch, 3.5 * inch])
    stats_table.setStyle(section_table_style([3.5 * inch, 3.5 * inch]))
    story.append(stats_table)
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "⚠ Complexity Indicator is based on vocabulary and sentence patterns only — "
        "not a substitute for a formal CEFR assessment.",
        caption_style
    ))

    # ---- WORD CLOUD ----
    story.append(Paragraph("Vocabulary Cloud", section_style))
    story.append(Paragraph(
        "Word size reflects frequency. Colors indicate part of speech: "
        "Blue = Verbs, Green = Adjectives, Purple = Adverbs, Orange = Nouns, Gray = Other.",
        caption_style
    ))
    wc_image_buffer.seek(0)
    wc_img = RLImage(wc_image_buffer, width=6.5 * inch, height=3.2 * inch)
    story.append(wc_img)
    story.append(Spacer(1, 10))

    # ---- TOP ADJECTIVES & VERBS (side by side) ----
    story.append(Paragraph("Top Words by Part of Speech", section_style))

    def make_word_table(title, data, col_widths):
        rows = [[title, "Count"]] + [[w, str(c)] for w, c in data]
        t = Table(rows, colWidths=col_widths)
        t.setStyle(section_table_style(col_widths))
        return t

    adj_table  = make_word_table("Top 5 Adjectives", adj_counts,  [2.5 * inch, 0.75 * inch])
    verb_table = make_word_table("Top 5 Verbs",      verb_counts, [2.5 * inch, 0.75 * inch])

    side_by_side = Table(
        [[adj_table, verb_table]],
        colWidths=[3.5 * inch, 3.5 * inch]
    )
    side_by_side.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                                       ("LEFTPADDING",  (0, 0), (-1, -1), 0),
                                       ("RIGHTPADDING", (0, 0), (-1, -1), 12)]))
    story.append(side_by_side)

    # ---- DETAILED ANALYSIS ----
    story.append(HRFlowable(width="100%", thickness=1, color=MID_GRAY, spaceBefore=14, spaceAfter=6))
    story.append(Paragraph("Detailed Analysis", section_style))

    # Filler Words
    story.append(Paragraph("Filler Words", styles["Heading3"]))
    story.append(Paragraph(
        f"Total filler words: <b>{total_fillers}</b> ({filler_pct:.1f}% of total words). "
        "Aim to keep fillers below 5% of your total words.",
        body_style
    ))
    if found_fillers:
        filler_data = [["Filler Word", "Count"]] + [[w, str(c)] for w, c in found_fillers]
        ft = Table(filler_data, colWidths=[3 * inch, 1 * inch])
        ft.setStyle(section_table_style([3 * inch, 1 * inch]))
        story.append(ft)
    else:
        story.append(Paragraph("✓ No filler words detected.", body_style))
    story.append(Spacer(1, 8))

    # Connective Words
    story.append(Paragraph("Connective Words", styles["Heading3"]))
    story.append(Paragraph(
        f"Unique connectives used: <b>{unique_connective_count}</b>. "
        "Connectives like 'however,' 'therefore,' and 'because' help organize your ideas.",
        body_style
    ))
    if found_connectives:
        conn_data = [["Connective", "Count"]] + [[w, str(c)] for w, c in found_connectives]
        ct = Table(conn_data, colWidths=[3 * inch, 1 * inch])
        ct.setStyle(section_table_style([3 * inch, 1 * inch]))
        story.append(ct)
    else:
        story.append(Paragraph("No connectives detected. Try linking ideas with 'however' or 'because.'", body_style))
    story.append(Spacer(1, 8))

    # Repeated Phrases
    story.append(Paragraph("Repeated Phrases", styles["Heading3"]))
    story.append(Paragraph(
        "Phrases used more than once. Try replacing some with alternative expressions for more variety.",
        body_style
    ))
    if all_repeated:
        rp_data = [["Phrase", "Times Used"]] + [[p, str(c)] for p, c in all_repeated]
        rpt = Table(rp_data, colWidths=[4.5 * inch, 1.5 * inch])
        rpt.setStyle(section_table_style([4.5 * inch, 1.5 * inch]))
        story.append(rpt)
    else:
        story.append(Paragraph("✓ No repeated phrases detected.", body_style))
    story.append(Spacer(1, 8))

    # Complexity Indicator breakdown
    story.append(Paragraph("Vocabulary Complexity Indicator Breakdown", styles["Heading3"]))
    breakdown_data = [
        ["Factor", "Your Value"],
        ["Vocabulary Variety (TTR)",  f"{ttr:.2f}"],
        ["Avg. Sentence Length",      f"{avg_sentence_length:.1f} words"],
        ["Lexical Density",           f"{lexical_density:.2f}"],
        ["Unique Connectives",        str(unique_connective_count)],
        ["Avg. Word Length",          f"{avg_word_length:.1f} characters"],
        ["Overall Score",             f"{complexity_score}/15  →  {complexity_label}"],
    ]
    bt = Table(breakdown_data, colWidths=[3.5 * inch, 3.5 * inch])
    bt.setStyle(section_table_style([3.5 * inch, 3.5 * inch]))
    story.append(bt)
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "⚠ This indicator is based on measurable text patterns only and is not a substitute "
        "for a formal CEFR assessment, which evaluates all four language skills across a range of tasks.",
        caption_style
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer


# --- 5. THE USER INTERFACE ---
st.set_page_config(page_title="Oral Homework Analyzer", layout="wide")
st.title("📝 Oral Homework Analyzer Pro")

with st.sidebar:
    st.header("Student Info")
    name = st.text_input("Full Name", value="Student")
    duration = st.number_input("Speaking Duration (Minutes)", min_value=0.1, value=2.0, step=0.1)

transcript = st.text_area("Paste your Microsoft Word Dictation here:", height=300)

# --- 6. THE ANALYSIS ---
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
            if tag == "VERB": return "hsl(210, 100%, 50%)"
            if tag == "ADJ":  return "hsl(120, 100%, 25%)"
            if tag == "ADV":  return "hsl(280, 100%, 50%)"
            if tag == "NOUN": return "hsl(30, 100%, 50%)"
            return "hsl(0, 0%, 70%)"

        # --- COMPUTE ALL STATS ---
        words = [token.text for token in doc if not token.is_punct and not token.is_space]
        word_count = len(words)
        wpm = word_count / duration
        unique_words = set(w.lower() for w in words)
        ttr = len(unique_words) / word_count if word_count > 0 else 0
        sentences = list(doc.sents)
        sentence_count = len(sentences)
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        content_pos = {"NOUN", "VERB", "ADJ", "ADV"}
        content_words = [t for t in doc if t.pos_ in content_pos and not t.is_punct and not t.is_space]
        lexical_density = len(content_words) / word_count if word_count > 0 else 0
        avg_word_length = sum(len(w) for w in words) / word_count if word_count > 0 else 0

        adj_counts  = Counter([t.text.lower()   for t in doc if t.pos_ == "ADJ"]).most_common(5)
        verb_counts = Counter([t.lemma_.lower() for t in doc if t.pos_ == "VERB"]).most_common(5)

        found_fillers = []
        for filler in FILLER_WORDS:
            count = transcript_lower.split().count(filler) if " " not in filler else transcript_lower.count(filler)
            if count > 0:
                found_fillers.append((filler, count))
        found_fillers.sort(key=lambda x: x[1], reverse=True)
        total_fillers = sum(c for _, c in found_fillers)
        filler_pct = (total_fillers / word_count * 100) if word_count > 0 else 0

        found_connectives = []
        for connective in CONNECTIVES:
            if f" {connective} " in f" {transcript_lower} ":
                count = transcript_lower.count(connective)
                found_connectives.append((connective, count))
        found_connectives.sort(key=lambda x: x[1], reverse=True)
        unique_connective_count = len(found_connectives)

        all_tokens = [token for token in doc if not token.is_punct and not token.is_space]
        bigrams, trigrams = [], []
        for i in range(len(all_tokens) - 1):
            t1, t2 = all_tokens[i], all_tokens[i + 1]
            if not (t1.is_stop and t2.is_stop):
                bigrams.append(f"{t1.text.lower()} {t2.text.lower()}")
        for i in range(len(all_tokens) - 2):
            t1, t2, t3 = all_tokens[i], all_tokens[i + 1], all_tokens[i + 2]
            if not (t1.is_stop and t2.is_stop and t3.is_stop):
                trigrams.append(f"{t1.text.lower()} {t2.text.lower()} {t3.text.lower()}")
        all_repeated = sorted(
            [(p, c) for p, c in Counter(bigrams).items()  if c >= 2] +
            [(p, c) for p, c in Counter(trigrams).items() if c >= 2],
            key=lambda x: x[1], reverse=True
        )

        complexity_score, complexity_label = get_complexity_indicator(
            ttr, avg_sentence_length, lexical_density, unique_connective_count, avg_word_length
        )

        # Generate word cloud and save to buffer for both display and PDF
        wc = WordCloud(background_color="white", width=800, height=500).generate(transcript)
        wc.recolor(color_func=pos_color_func)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        wc_image_buffer = io.BytesIO()
        fig.savefig(wc_image_buffer, format="png", bbox_inches="tight", dpi=150)
        wc_image_buffer.seek(0)

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

        # --- DISPLAY: WORD CLOUD + ADJECTIVES/VERBS ---
        left_col, right_col = st.columns([2, 1])

        with left_col:
            st.subheader("Vocabulary Cloud")
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
            if adj_counts:
                st.table(pd.DataFrame(adj_counts, columns=["Word", "Count"]))
            else:
                st.info("No adjectives found. Try using more descriptive language!")

            st.divider()

            st.subheader("Top 5 Verbs")
            if verb_counts:
                st.table(pd.DataFrame(verb_counts, columns=["Word", "Count"]))
            else:
                st.info("No verbs found.")

        st.divider()

        # --- DISPLAY: DETAILED ANALYSIS (COLLAPSIBLE) ---
        with st.expander("🔍 Detailed Analysis"):

            detail_col1, detail_col2, detail_col3 = st.columns(3)

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

            with detail_col2:
                st.subheader("Filler Words")
                st.caption("""
                Filler words are sounds and phrases like "um," "uh," "like," or "you know"
                that we use to fill pauses while thinking. A few fillers are completely normal,
                but too many can make speech harder to follow. Aim to keep fillers below
                5% of your total words.
                """)
                st.metric("Total Filler Words", total_fillers,
                          help=f"{filler_pct:.1f}% of your total words")
                if found_fillers:
                    st.table(pd.DataFrame(found_fillers, columns=["Filler", "Count"]))
                else:
                    st.success("No filler words detected — great job!")

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
                    st.table(pd.DataFrame(found_connectives, columns=["Connective", "Count"]))
                else:
                    st.warning("No connectives found. Try linking your ideas with words like 'however' or 'because.'")

            st.divider()

            st.subheader("🔁 Repeated Phrase Detector")
            st.caption("""
                Repeating the same short phrase too often can make speech sound less fluent.
                The table below shows any two- or three-word combinations you used more than once.
                Try replacing some of these with alternative expressions to add variety.
            """)
            if all_repeated:
                st.table(pd.DataFrame(all_repeated, columns=["Phrase", "Times Used"]))
            else:
                st.success("No repeated phrases detected — great variety!")

            st.divider()

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

        st.divider()

        # --- DOWNLOAD PDF BUTTON ---
        pdf_buffer = generate_pdf_report(
            name=name, duration=duration, wpm=wpm, word_count=word_count,
            ttr=ttr, avg_sentence_length=avg_sentence_length,
            lexical_density=lexical_density, avg_word_length=avg_word_length,
            adj_counts=adj_counts, verb_counts=verb_counts,
            found_fillers=found_fillers, total_fillers=total_fillers, filler_pct=filler_pct,
            found_connectives=found_connectives, unique_connective_count=unique_connective_count,
            all_repeated=all_repeated,
            complexity_score=complexity_score, complexity_label=complexity_label,
            wc_image_buffer=wc_image_buffer
        )

        st.download_button(
            label="📄 Download Full Analysis as PDF",
            data=pdf_buffer,
            file_name=f"{name.replace(' ', '_')}_analysis_{datetime.date.today()}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    else:
        st.error("Please paste your transcript first!")
