"""Memory Layer Pro — Streamlit entry point (SCAFFOLD).

Teaching demo only: synthetic data, no personal information. The full chat
and trace UI arrives with the UI task; this entry proves config validation
and fails with an actionable message when the environment is incomplete.
"""

import streamlit as st

from config import ConfigError, load, validate


def main() -> None:
    st.set_page_config(page_title="Memory Layer Pro (MVP)")
    st.title("Memory Layer Pro — MVP scaffold")
    st.info("Teaching demo. Synthetic data only; nothing here is production-ready.")
    try:
        settings = validate(load())
    except ConfigError as exc:
        st.error(str(exc))
        st.stop()
        return
    st.success(f"Configuration valid. Model: {settings.nvidia_model}")
    st.write("Chat and pipeline trace arrive with the next tasks.")


if __name__ == "__main__":
    main()
