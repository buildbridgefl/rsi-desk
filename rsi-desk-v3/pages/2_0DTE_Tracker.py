"""0DTE trade tracker — shows up as its own page in the sidebar."""
import streamlit as st

st.set_page_config(page_title="0DTE Tracker", page_icon="🎯", layout="wide")

from ui import zerodte_tab  # noqa: E402

zerodte_tab.render()
