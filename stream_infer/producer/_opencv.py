import cv2

from ..log import logger


class OpenCVProducer:
    def __init__(self, width: int, height: int, cvt_code=None):
        self.width = width
        self.height = height
        self.cvt_code = cvt_code

    def read(self, source, fps=None, position=0):
        """
        Reads frames from a video file/stream_url/v4l2 device.
        Optionally skips frames to meet the specified fps.

        Args:
            source (str): The path to the video file/stream_url/v4l2 device.
            fps (int, optional): Target frames per second. If None, no frame skipping is done.
            position (int, optional): The position in seconds from where to start reading the video.

        Yields:
            numpy.ndarray: frame
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Failed to open {source}")

        original_fps = cap.get(cv2.CAP_PROP_FPS)

        # Skip to the requested second
        if position > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, position * 1000)  # position in milliseconds

        frame_interval = 1.0
        if fps is not None and original_fps > fps:
            frame_interval = original_fps / fps

        frame_index = int(original_fps * position)
        next_frame_to_process = frame_index
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index >= next_frame_to_process:
                try:
                    height, width, _ = frame.shape
                    if width != self.width or height != self.height:
                        frame = cv2.resize(frame, (self.width, self.height))

                    if self.cvt_code is not None:
                        frame = cv2.cvtColor(frame, self.cvt_code)

                    yield frame
                    next_frame_to_process += frame_interval
                except Exception as e:
                    logger.error(f"Error processing frame: {e}")
                    raise e

            frame_index += 1

        cap.release()

    def get_info(self, source):
        """
        Extracts video properties.

        Args:
            source (str): The path to the video file/stream_url/v4l2 device.

        Returns:
            dict: Video properties including width, height, fps, and frame count.
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Failed to open {source}")

        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_seconds = int(frame_count / fps)

        cap.release()

        return {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "total_seconds": total_seconds,
        }
