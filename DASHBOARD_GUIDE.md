# Dashboard Guide (Streamlit)

## 1. Quick start

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

Before the first run, **re-run the notebook (at least Part C)**: it now also writes `test_scores.csv`, `dev_oof_scores.csv` and `run_info.json`, which the Operating Point page needs.
`python setup_dashboard.py --check` lists which data files are found.

Teammates only need `setup_dashboard.py` (one file). It regenerates every other file.

## 2. Structure

```
dashboard/
  app.py                  entry point = page 1 (Overview)
  data_loader.py          the ONLY file that reads files (registry of datasets)
  ui.py                   page setup, banner, sidebar file status, "missing file" handling, number formatting
  charts.py               metric maths (same as notebook) + matplotlib figures; no file access
  pages/
    1_Model_Comparison.py
    2_Operating_Point.py
    3_Explainability.py
  data/                   notebook outputs go here (+ calcification_summary.csv)
  requirements.txt        streamlit, pandas, numpy, matplotlib, scikit-learn
  .streamlit/config.toml  theme colours
```

Dependency rule: **pages -> ui, charts, data_loader. `charts` and `data_loader` never import pages.** Pages never open files themselves.

## 3. How `data_loader.py` should look

Contract (everything a page may call):

| Function | Returns |
|---|---|
| `load(key)` | DataFrame for a registered dataset, or `None` if the file is missing |
| `load_csv(name, **kw)` | DataFrame for an ad-hoc CSV in `data/` (or `None`) |
| `run_info()` | dict from `run_info.json` (`{}` if missing) |
| `primary_model()` | name of the primary model |
| `image(name)` | path of a PNG in `data/` (or `None`) |
| `available()` | `{key: file exists}` (drives the sidebar status) |

Shape of the file:

```python
DATA_DIR = Path(os.environ.get("DASH_DATA_DIR", Path(__file__).parent / "data"))

DATASETS = {                                   # key -> file + pandas.read_csv options
    "cv_summary": dict(file="cv_endterm_summary.csv", read=dict(index_col=0)),
    "stats":      dict(file="stat_tests_vs_LR.csv"),
    ...
    # <<NEW_DATASETS>>                         # --add-dataset inserts new lines above this marker
}

@_cache                                        # st.cache_data; key includes the file's modified time
def _read_csv(p, read_items, mtime): return pd.read_csv(p, **dict(read_items))
def load(key): ...                             # None when the file does not exist
```

Rules: no Streamlit widgets in the loader; missing file -> `None`, never an exception; all file names live in `DATASETS` only; do not remove the `# <<NEW_DATASETS>>` marker. Set `DASH_DATA_DIR` to point the dashboard at a different folder (for example a second run).

## 4. How `app.py` should look

```python
import streamlit as st
import data_loader as dl
import ui

ui.setup("Overview")                           # 1. FIRST call: page config, banner, sidebar file status
st.title("...")
cv, test = ui.need("cv_summary", "test")       # 2. load data or stop with a clear message
primary = dl.primary_model()                   # 3. KPI cards (st.columns + st.metric), then protocol text
```

Rules: `ui.setup()` first; read data only via `ui.need()` / `dl.*`; keep layout code in the page and maths in `charts.py`.

## 5. How a page should look

Every file in `pages/` follows the same 4 steps (this is exactly what `--new-page` generates):

```python
import streamlit as st
import charts, data_loader as dl, ui

ui.setup("My Page")                            # 1. setup (always first)
st.title("My Page")
(df,) = ui.need("my_key")                      # 2. data (stops politely if the file is missing)
with st.sidebar:                               # 3. controls
    model = st.selectbox("Model", list(df.index))
st.pyplot(charts.some_figure(df, model))       # 4. output (tables, figures, text)
```

File names start with a number (`4_Name.py`) so Streamlit orders them in the sidebar.

`charts.py` helpers: `thr_for_recall`, `op_metrics`, `ppv_npv`, `dot_plot`, `roc_pr`, `prevalence_fig`, `odds_ratio_fig`, `hbar`, `grouped_bar`. `ui.py` helpers: `setup`, `need`, `fmt_ci`, `fmt_ms`.

## 6. What teammates can do

| Task | Command / action |
|---|---|
| Add a page | `python setup_dashboard.py --new-page "Calcification Details"` creates `pages/<next number>_Calcification_Details.py` from the template. Edit it; Streamlit shows it in the sidebar. |
| Add a data file | put the CSV in `data/`, then `python setup_dashboard.py --add-dataset cal_cv cal_cv_summary.csv --index-col` (`--index-col` if the first column is a label/index). Use `ui.need("cal_cv")` in the page. |
| Add a figure | add a function to `charts.py` that returns a matplotlib figure; call `st.pyplot(...)` in the page. |
| Rebuild generated files | `python setup_dashboard.py --force` (overwrites the generated files, so keep your own edits in new pages). |
| Calcification project | run its notebook, copy its CSVs to `data/`, register them (above), and create a page. The Model Comparison page already reads `calcification_summary.csv` (columns `model, cv_auc, test_auc`; pre-filled from your team's calcification slides, replace it with real exports when available). |

## 7. Data contract (notebook file -> page)

| File | Used by |
|---|---|
| `cv_endterm_summary.csv`, `cv_midterm_summary.csv` | Overview, Model Comparison |
| `test_endterm_results.csv` | Overview, Model Comparison, Operating Point (consistency check) |
| `stat_tests_vs_LR.csv`, `paired_bootstrap_auc_vs_LR.csv` | Model Comparison |
| `midterm_vs_wide_tuning_cv.csv`, `midterm_vs_endterm_test.csv`, `calcification_summary.csv` | Model Comparison |
| `dev_oof_scores.csv`, `test_scores.csv` | Operating Point (thresholds and curves are recomputed live) |
| `lr_odds_ratios.csv`, `permutation_importance.csv`, `error_analysis_primary.csv` | Explainability |
| `run_info.json` | Overview, banner (shows a warning if the files came from `FAST_MODE`) |
| `B4_calibration.png`, `A_decision_tree.png` | optional images |

## 8. Behaviour you can rely on
- A missing file never crashes a page: it shows "Missing data file(s): ..." and stops.
- On Operating Point, with the notebook's target (0.90) the dashboard's specificity/recall should match `test_endterm_results.csv` (a consistency caption says so). Other targets are recomputed from the training out-of-fold scores, using the same rule as the notebook.
- Numbers always come from the files; nothing is typed into the code (except the calcification CSV noted above).

## 9. Deploying
Push the `dashboard/` folder (including small `data/` files) to a GitHub repo and deploy it on Streamlit Community Cloud with `app.py` as the main file.

## 10. Testing status (please read)
- Streamlit is **not installed** in my environment, so I could not launch the dashboard in a browser. I ran every page with a small stand-in for Streamlit against output files from a dry run of the notebook (synthetic data). All 4 pages and 2 generated pages ran without errors, produced their tables and figures, and stopped cleanly when data files were missing. The generator commands (`--new-page`, `--add-dataset`, `--check`, `--force`) were also tested.
- The dashboard's threshold, specificity and recall matched the notebook's table to within CSV rounding (about 5e-5) on that dry run.
- Not tested: real Streamlit rendering and layout, or real data. Expect to fix small display issues on first launch; send me the error text.
- Charts use matplotlib (no hover tooltips). To switch to Plotly later, replace the figure functions in `charts.py` and use `st.plotly_chart`.
