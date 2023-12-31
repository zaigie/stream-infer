def trans_position2time(position: int) -> str:
    return f"{position // 3600:02d}:{position // 60 % 60:02d}:{position % 60:02d}"


def trans_time2position(time: str) -> int:
    h, m, s = time.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)
