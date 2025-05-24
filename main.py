import sys
import streamlit as st
from interface import HoloMarineInterface

if __name__ == "__main__":
    try:
        ui = HoloMarineInterface()
        ui.render()
    except Exception as e:
        st.error(f"Fatal Error: {e}")
        sys.exit(1)
