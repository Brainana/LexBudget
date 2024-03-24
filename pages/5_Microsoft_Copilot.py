import streamlit as st
import streamlit.components.v1 as components

# embed copilot into streamlit webpage
iframe_src = "https://copilot.microsoft.com/?showntbk=1"
components.iframe(iframe_src, width=1200, height=800)