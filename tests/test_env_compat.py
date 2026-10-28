"""Environment compatibility guard: Streamlit trace rendering needs pyarrow.

`app._render_detail` uses `st.table`, which imports pyarrow at render time.
pyarrow 26+ requires NumPy 2, but this project pins `numpy<2` for Intel-mac
torch wheels. This test fails with the same ImportError the UI hits instead
of letting a bad pin reach the browser.
"""

import importlib


def test_pyarrow_imports_with_installed_numpy():
    numpy = importlib.import_module("numpy")
    assert tuple(int(p) for p in numpy.__version__.split(".")[:2]) < (2, 0), (
        f"expected project pin numpy<2, found {numpy.__version__}"
    )
    # Must not raise "pyarrow requires NumPy 2.0 or newer".
    pyarrow = importlib.import_module("pyarrow")
    assert pyarrow.__version__
