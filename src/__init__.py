"""Healthcare RAG Intelligence Platform."""

import os

# faiss-cpu and torch each bundle their own OpenMP runtime. On macOS, loading
# the second one aborts the process ("Initializing libomp.dylib, but found
# libomp.dylib already initialised"). Tolerating the duplicate is the
# supported workaround and must be set before either library is imported,
# which is why it lives in the package __init__ rather than at a call site.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "0.1.0"
