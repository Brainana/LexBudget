import streamlit as st
import streamlit.components.v1 as components

st.markdown('Using Microsoft Copilot')

# embed copilot into streamlit webpage
iframe_src = "https://copilot.microsoft.com/?showntbk=1"
components.iframe(iframe_src, width=350, height=800)