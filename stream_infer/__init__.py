__version__ = "0.4.3"

from .inference import Inference
from .player import Player

try:
    from .streamlit_ import StreamlitApp
except ImportError:

    class StreamlitApp:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "StreamlitApp needs streamlit package support. Please use 'pip install \"stream-infer[server]\"' to install the server version."
            )
