import cv2


class OpenCVProducer:
    def __init__(self, width: int, height: int, cvt_code: int = cv2.COLOR_RGB2BGR):
        self.width = width
        self.height = height
        self.cvt_code = cvt_code

    def read(self, path):
        """
        Reads frames from a video file/stream_url/v4l2 device.

        Args:
            path (str): The path to the video file/stream_url/v4l2 device.
                eg. "/path/to/video.mp4" or "rtmp://localhost/live" or "/dev/video0"

        Yields:
            numpy.ndarray: frame
        """
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open {path}")

        while True:
            try:
                ret, frame = cap.read()
                if not ret:
                    break

                height, width, channel = frame.shape
                if width != self.width or height != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))

                frame = cv2.cvtColor(frame, self.cvt_code)
                # cv2.imwrite("frame.jpg", frame)
                yield frame
            except Exception as e:
                break

        cap.release()

    def get_info(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open {path}")

        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count == 0:
            frame_count = -1

        cap.release()

        return {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
        }
