import streamlit as st

from PIL import Image

st.set_page_config(page_title="About the Lexington Chatbot Team", page_icon="📈")


col1 , col2 = st.columns([9, 1] , gap = "small")

col1.markdown("# Lexington Chatbot Team")

col1.markdown("### Our Mission:")

col1.write(
    "Our mission is to inform Lexington residents accurately by providing instant, well-cited answers to frequently asked questions about the "
    "school building project, including financial details and voting logistics. Beyond delivering reliable information, the chatbot is designed to "
    "raise awareness and boost engagement through its interactive interface, which prompts users with suggested follow-up questions to deepen "
    "their understanding of key issues. Additionally, the tool will gather and analyze user feedback—tracking recurring questions, response satisfaction, "
    "and knowledge gaps. By combining education, engagement, and data-driven insights, this project aims to foster an informed and participatory community "
    "ahead of the vote."
)
col1.write("")

col1_sub, col2_sub = col1.columns(2)


with col1_sub:
    col1_sub.markdown("### Team Members:")

    col1_sub.write("Jerry Xu (Project Lead)")
    col1_sub.write("Justin Wang")
    col1_sub.write("Jasmine Gu")
    col1_sub.write("Joley Leung (Graphics)")

with col2_sub:
    col2_sub.markdown("### Mentors:")

    col2_sub.write("Wei Ding")
    col2_sub.write("Jeannie Lu")


    

col2.markdown('<a href="https://lexyouthstem.org/" target="_blank"><img src="https://brainana.github.io/LexBudgetDocs/images/teamlogo.png" width="200"></a>', unsafe_allow_html=True)





