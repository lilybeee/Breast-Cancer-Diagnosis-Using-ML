"""ui.py - shared page setup, banner, data-file status and missing-data handling."""
import streamlit as st

import data_loader as dl


def setup(title):
    """Call first on every page."""
    st.set_page_config(page_title=f"{title} | Breast Mass Dashboard", layout="wide")
    st.caption("Research prototype on a curated dataset (CBIS-DDSM masses). "
               "Not a clinical tool; no external validation.")
    if dl.run_info().get("fast_mode"):
        st.error("These files come from a FAST_MODE dry run. Do not report these numbers.")
    # with st.sidebar:
    #     st.markdown("**Data files**")
    #     for key, ok in dl.available().items():
    #         st.write(("[x] " if ok else "[ ] ") + key)


def need(*keys):
    """Return the requested datasets, or show a clear message and stop the page."""
    missing = [k for k in keys if dl.load(k) is None]
    if missing:
        names = ", ".join(dl.DATASETS[k]["file"] for k in missing)
        st.warning(f"Missing data file(s): {names}. Copy them from the notebook's outputs/ folder into {dl.DATA_DIR}.")
        st.stop()
    return [dl.load(k) for k in keys]


def fmt_ci(v, lo, hi):
    return f"{v:.3f} [{lo:.3f}-{hi:.3f}]"


def fmt_ms(mean, std):
    return f"{mean:.3f} +/- {std:.3f}"
