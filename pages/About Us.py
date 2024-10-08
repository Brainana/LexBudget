import streamlit as st

from PIL import Image

st.set_page_config(page_title="About the Lexington Chatbot Team", page_icon="📈")


col1 , col2 = st.columns([9, 1] , gap = "small")

col1.markdown("# Lexington Chatbot Team")

col1.markdown("### Our Mission:")

col1.write(
    "Our mission is to improves accessibility, understanding, and engagement with the town budget for residents by implementing measures to "
    "simplify technical language, condense the document length, enhance digital accessibility features, provide historical data and comparisons, and "
    "establish clear avenues for community engagement and feedback. Second enhances transparency and community engagement in Lexington by utilizing "
    "AI-generated transcriptions from town meeting recordings to provide detailed insights, including key topics, time allocation, public comments, and "
    "feedback, thereby fostering a sense of community involvement and accountability. "
    "The implementation of a chatbot will further facilitate easy interaction, allowing residents to ask specific questions and receive prompt replies."
)
col1.write("")

col1_sub, col2_sub = col1.columns(2)


with col1_sub:
    col1_sub.markdown("### Team Members:")

    col1_sub.write("Jerry Xu (Project Lead)")
    col1_sub.write("Kevin Zhu")
    col1_sub.write("Jasmine Gu")
    col1_sub.write("Justin Wang")
    col1_sub.write("Cassidy Xu")
    col1_sub.write("Emma He")
    col1_sub.write("Willam Yang")
    col1_sub.write("Andrew Pan")
    col1_sub.write("Joley Leung (Graphics)")

with col2_sub:
    col2_sub.markdown("### Mentors:")

    col2_sub.write("Andrei Radulescu-Banu")
    col2_sub.write("Chester Curme")
    col2_sub.write("Jeannie Lu")
    col2_sub.write("Wei Ding")
    col2_sub.write("John Truelove")
    col2_sub.write("Nagarjuna Venna")
    col2_sub.write("Neerja Bajaj")
    


img = Image.open("images/teamlogo.png")
col2.image( img ,  width = 200 , channels = "BGR")





