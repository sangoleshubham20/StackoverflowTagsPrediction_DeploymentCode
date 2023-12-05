import streamlit as st
from PIL import Image


def show_home_page():
    """
    This function displays the Home Page.
    """
    st.header("Tagging Stack Overflow questions using Machine Learning")
    st.text("")
    st.text("")
    img = Image.open("stack_home.jpg")
    st.image(img, caption="Credits: reddit.com")
    st.text("")

    st.caption("### **OVERVIEW :**")
    st.markdown("""***[Stack Overflow](https://stackoverflow.com/)*** is a well-known online community for programmers
                   and developers, providing a platform for knowledge sharing, problem-solving, and collaboration.
                   Amidst the vast sea of coding resources, Stack Overflow stands out as a go-to destination for tech
                   professionals worldwide. While most people are familiar with Stack Overflow as a go-to source for
                   solving programming queries, there are some extraordinary facts that make this platform even more
                   fascinating. From its humble beginnings to its massive user base, Stack Overflow has established
                   itself as an indispensable resource for the tech community.
              """)
    st.markdown("""Stack Overflow was created in 2008 by ***[Jeff Atwood](https://en.wikipedia.org/wiki/Jeff_Atwood)***
                   and ***[Joel Spolsky](https://en.wikipedia.org/wiki/Joel_Spolsky)*** , as a more open alternative to
                   earlier Q&A sites such as ***[Experts-Exchange](https://go.experts-exchange.com/)*** . The name for
                   the website was chosen by voting in April 2008 by readers of ***[Coding Horror](https://blog.codinghorror.com/)*** ,
                   Atwood’s popular programming blog. As of **March 2022**, Stack Overflow has over **20 million**
                   registered users and has received over **24 million** questions and **35 million** answers
                   (not counting deleted users and questions). Based on the type of tags assigned to questions, the top
                   eight most discussed topics on the site are: **Java, JavaScript, C#, PHP, Android, jQuery, Python and
                   HTML**.
                """)
    st.divider()

    st.caption("### **OBJECTIVE :**")
    st.markdown("""We need to build a machine learning model which **extract tags from questions** that are being
                   asked by the users on Stack Overflow.
                """)
    st.divider()

    st.caption("### **BUSINESS CONSTRAINTS :**")
    st.markdown("""* **Interpretability of the model** is not that consequential because a user doesn't need to know why
                   the model has assigned specific labels to a user's question.
                """)
    st.markdown("* Predict as many tags as possible with **high precision and recall**.")
    st.markdown("""* There are **no low-latency requirements** but at the same time, we also don't want our
                   latency to be in several minutes.
                """)
    st.markdown("""* **Errors must be minimized** since, incorrect tags could impact customer experience on Stack Overflow.
                """)
    st.divider()

    st.caption("### **MAPPING THE PROBLEM TO A MACHINE LEARNING PROBLEM :**")
    st.markdown("""As we already mentioned, we need to assign labels to a user's question. Therefore, it is a **Multi-label
                   classification problem.**
                """)
    st.markdown("Performance metric(s) :")
    st.markdown("""* ***Micro-Averaged F1-Score (Mean F Score)*** : The F1 score can be interpreted as a weighted 
                   average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at
                   0. The relative contribution of precision and recall to the F1 score are equal. In the multi-class
                   and multi-label case, this is the weighted average of the F1 score of each class.
                """)
    st.markdown("""* ***Micro f1-score*** : It calculate metrics globally by counting the total true positives, false
                   negatives and false positives. This is a better metric when we have class imbalance.
                """)
    st.markdown("""* ***Macro f1-score*** : It calculate metrics for each label, and find their unweighted mean. This
                   does not take label imbalance into account.
                """)
    st.markdown("* ***Hamming loss*** : The Hamming loss is the fraction of labels that are incorrectly predicted.")
    st.divider()

    st.caption("### **DATA DESCRIPTION :**")
    st.text("")
    st.markdown("You can download the dataset from ***[here](https://www.kaggle.com/c/facebook-recruiting-iii-keyword-extraction/data)***")
    st.markdown("We have 2 data files:")
    st.markdown("* **Train.csv**")
    st.markdown("* **Test.csv**")
    st.markdown("Size of Train.csv: **6.75 GB**")
    st.markdown("Size of Test.csv: **2 GB**")
    st.markdown("Number of rows in Train.csv: **6,034,195**")
    st.markdown("""The questions are randomized and contains a mix of verbose text sites as well as sites related to
                   math and programming. The number of questions from each site may vary, and no filtering has been
                   performed on the questions (such as closed questions).
                """)
    st.markdown("Data file's information :")
    st.markdown("""* **Train.csv :** A comma separated file containing the description of the questions used for
                training. Fields are **'ID'** (Unique identifier for each question), **'Title'** (the question's title), **'Body'**
                (the body of the question), **'Tags'** (the tags associated with the question in a space-seperated format.
                All lowercase, does not contain tabs or ampersands.)
                """)
    st.markdown("* **Test.csv :** It contains the same columns but **without the Tags** which is our target variable.")
    st.divider()

    st.markdown("""***LIBRARIES : `base64`, `matplotlib`, `numpy`, `pandas`, `pickle`, `PIL`,
                `re`, `scikit-learn`, `scipy`, `seaborn`, `streamlit`***""")
