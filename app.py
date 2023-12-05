import re
from home_page import show_home_page
import pandas as pd
from streamlit_option_menu import option_menu
from PIL import Image
import streamlit as st
import pickle as pkl

img = Image.open("stack_logo.png")
page_config = {"page_title": "StackOverflow Tags Prediction", "page_icon": img, "layout": "centered"}
st.set_page_config(**page_config)

page = option_menu(
    menu_title=None,
    options=["Home", "Classification", "Analysis", "Code"],
    icons=["house-fill", "motherboard", "book", "file-earmark-code"],
    default_index=0,
    orientation="horizontal",
    styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "black", "font-size": "15px"},
            "nav-link": {
                "font-size": "15px",
                "text-align": "center",
                "margin": "0px",
                "--hover-color": "#eee",
            },
            "nav-link-selected": {"background-color": "red"}
            }
)

# Home page
if page == "Home":
    show_home_page()

# Prediction page
if page == "Classification":

    st.text("")
    st.text("")
    st.markdown("""***This Machine Learning application suggests the tags based on the content that is present in the
                   question which is posted by the user on Stackoverflow.***
                """)
    st.text("")

    @st.cache_resource(show_spinner=False)
    def load_models():
        clf = pd.read_pickle("clf.zip")
        tfidf = pd.read_pickle("tfidf.zip")
        multilabel = pd.read_pickle("multilabel.zip")
        s = pkl.load(open("stop_words.pkl", "rb"))
        return clf, tfidf, multilabel, s
    clf, tfidf, multilabel, s = load_models()
    T = []
    words = st.text_input("***Enter your Question :***")
    if st.button("Predict"):
        words = re.sub('\n', ' ', words)
        words = re.sub('[!@%^&*()$:"?<>=~,;`{}|]', ' ', words)
        words = re.sub(
            r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?]))''',
            ' ', words)
        words = re.sub('_', '-', words)
        words = words.replace('[', ' ')
        words = words.replace(']', ' ')
        words = words.replace('/', ' ')
        words = words.replace('\\', ' ')
        words = re.sub(r'(\s)\-+(\s)', r'\1', words)
        words = re.sub(r'\.+(\s)', r'\1', words)
        words = re.sub(r'\.+\.(\w)', r'\1', words)
        words = re.sub(r'(\s)\.+(\s)', r'\1', words)
        words = re.sub("'", '', words)
        words = re.sub(r'\s\d+[\.\-\+]+\d+|\s[\.\-\+]+\d+|\s+\d+\s+|\s\d+[\+\-]+', ' ', words)
        words = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", words)
        words = re.sub(r'\s\#+\s|\s\++\s', ' ', words)
        final_words = [word for word in words.split()]
        clean_text = filter(lambda w: not w in s, final_words)
        words = ''
        for word in clean_text:
            words += word + ' '
        T.append(words)
        tfidf_output = tfidf.transform(T)
        results = multilabel.inverse_transform(clf.predict(tfidf_output))
        tag_arr = []
        emp_str = ''
        for result in results[0]:
            tag_arr.append(result)
        for i in tag_arr:
            emp_str += "- " + i + "\n"
        st.markdown("***Following are the tags in your question :***")
        st.markdown(emp_str)

# Analysis page
if page == "Analysis":

    st.text("")
    st.caption("### **A CLOSER LOOK INTO THE DATA :**")
    st.text("")
    st.markdown("* ***Data before pre-processing :***")
    st.text("")
    img = Image.open("so00.png")
    st.image(img)
    st.text("")
    st.markdown("* ***Data after pre-processing :***")
    st.text("")
    img = Image.open("so01.png")
    st.image(img)

    st.divider()

    st.caption("### **FREQUENCY OF TAGS APPEARING IN QUESTIONS :**")
    st.text("")
    st.text("")
    col1, col2, col3, col4, col5 = st.columns([1, 1, 8, 1, 1])
    with col3:
        img = Image.open("so1.png")
        st.image(img)
    st.text("")
    st.markdown("""You can have a look at the tags dictionary ***[here](https://drive.google.com/file/d/1JRJ-cG-6E2S8eIS2HioTg0iAoRT0YXq9/view?usp=drive_link)***
                   in more detail where you san see the frequency of each tag.
                """)
    st.markdown("**Observations :**")
    st.markdown("* There are total **153** tags which are used more than **10,000** times.")
    st.markdown("* There are **14** tags which are used more than **100,000** times.")
    st.markdown("* Most frequent tag **(i.e. c#)** is used **331505** times.")

    st.divider()

    st.caption("### **TAGS PER QUESTION :**")
    st.text("")
    st.text("")
    col1, col2, col3, col4 = st.columns([1, 2, 8, 1])
    with col3:
        img = Image.open("so2.png")
        st.image(img)
    st.text("")
    st.markdown("**Observations :**")
    st.markdown("* Maximum number of tags per question: **5**")
    st.markdown("* Minimum number of tags per question: **1**")
    st.markdown("* Average number of tags per question: **2.899**")
    st.markdown("* Most of the questions are having **2 or 3** tags.")

    st.divider()

    st.caption("### **MOST FREQUENT TAGS :**")
    st.text("")
    st.text("")
    col1, col2, col3, col4 = st.columns([1, 2, 8, 1])
    with col3:
        img = Image.open("so3.png")
        st.image(img)
    st.text("")
    st.markdown("""**Observation :** A look at the word cloud shows that **'c#', 'java', 'php', 'asp.net', 'javascript',
                   'android', 'python'** are some of the most frequent tags.
                """)

    st.divider()

    st.caption("### **A LOOK AT THE TOP 20 TAGS :**")
    st.text("")
    st.text("")
    col1, col2, col3, col4 = st.columns([1, 2, 8, 1])
    with col3:
        img = Image.open("so4.png")
        st.image(img)
    st.text("")
    st.markdown("**Observations :**")
    st.markdown("* Majority of the most frequent tags are programming languages.")
    st.markdown("* **C#** is the top most frequent programming language.")
    st.markdown("* **Android, IOS, Linux and Windows** are among the top most frequent Operating Systems.")

    st.divider()

    st.caption("### **ANALYSIS OF TAGS :**")
    st.text("")
    st.text("")
    col1, col2, col3, col4 = st.columns([1, 2, 8, 1])
    with col3:
        img = Image.open("so5.png")
        st.image(img)
    st.text("")
    st.markdown("**Observations :**")
    st.markdown("* With **5500** tags, we are covering  **99.157 %** of the questions.")
    st.markdown("* With **500** tags, we are covering  **90.956 %** of the questions.")

    st.divider()

    st.caption("### **FINAL MODEL :**")
    st.text("")
    st.text("")
    st.markdown("* ***Featurized data with Tf-Idf vectorizer :***")
    img = Image.open("so6.png")
    st.image(img)
    st.text("")
    img = Image.open("so7.png")
    st.image(img)
    st.text("")
    st.markdown("* ***Applied LinearSVC with OneVsRest Classifier :***")
    img = Image.open("so8.png")
    st.image(img)

# Code page
if page == "Code":

    st.text("")
    st.write("###### If you are more interested in the code you can directly jump into these repositories :")
    st.text("")
    st.caption("### **Deployment** : ***[link](https://github.com/sangoleshubham20/StackoverflowTagsPrediction_DeploymentCode)***")
    st.caption("### **Modelling** : ***[link](https://github.com/sangoleshubham20/StackoverflowTagsPrediction_ModellingCode)***")