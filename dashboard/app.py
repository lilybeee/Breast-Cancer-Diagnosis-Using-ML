"""
app.py - dashboard entry point = page 1 (Overview).   Run:  streamlit run app.py
Other pages live in pages/ (Streamlit lists them in the sidebar automatically).
"""
import streamlit as st

import data_loader as dl
import ui

ui.setup("Overview")
import streamlit as st

st.switch_page("pages/3_Calcification_and_Mass_Analysis.py")