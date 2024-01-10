from enum import Enum
from pydantic import BaseModel


class Mode(Enum):
    REALTIME = "realtime"
    OFFLINE = "offline"


class ProducerType(Enum):
    OPENCV = "opencv"
    PYAV = "pyav"


class RecordMode(Enum):
    NONE = "none"
    OPENCV = "opencv"
    PYAV = "pyav"
