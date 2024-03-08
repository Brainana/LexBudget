import streamlit as st

from PIL import Image

st.set_page_config(page_title="About the Lexington Chatbot Team", page_icon="📈")
# st.sidebar.success("About Us")

# st.write(
#     "Our Mission"
# )

col1 , col2 = st.columns([9, 1] , gap = "small")
col1.markdown("# Lexington Chat bot Team")

col1.markdown("### Our Mission:")

col1.write(
    "Our mission is to improves accessibility, understanding, and engagement with the town budget for residents by implementing measures to "
    "simplify technical language, condense the document length, enhance digital accessibility features, provide historical data and comparisons, and "
    "establish clear avenues for community engagement and feedback. Second enhances transparency and community engagement in Lexington by utilizing "
    "AI-generated transcriptions from town meeting recordings to provide detailed insights, including key topics, time allocation, public comments, and "
    "feedback, thereby fostering a sense of community involvement and accountability. "
    "The implementation of a chatbot will further facilitate easy interaction, allowing residents to ask specific questions and receive prompt replies."
)

col1.markdown("### Team Members:")

col1.write(
"Jerry Xu (Project Lead)"
)
col1.write(
"Kevin Zhu"
)
col1.write(
"Jasmine Gu"
)
col1.write(
"Justin Wang"
)
col1.write(
"Emma He"
)
col1.write(
"Willam Yang"
)
col1.write(
"Joley Leung (Graphics)"
)

img = Image.open("images/teamlogo.png")
col2.image( img ,  width = 200 , channels = "BGR")

# col12, col23, col3 = st.columns(3)
#
# with col12:
#     st.write(' ')
#
# with col23:
#     col23.write("Jerry Xu (Project Lead)")
#     col23.write("Kevin Zhu")
#     col23.write("Jasmine Gu")
#     col23.write("Justin Wang")
#     col23.write("Emma He")
#     col23.write("Willam Yang")
#     col23.write("Joley Leung (Graphics)")
#
# with col3:
#     st.write(' ')






