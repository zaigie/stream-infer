import cv2

from ..log import logger


class OpenCVProducer:
    def __init__(self, width: int, height: int, cvt_code=None):
        self.width = width
        self.height = height
        self.cvt_code = cvt_code

    def read(self, path, fps=None):
        """
        Reads frames from a video file/stream_url/v4l2 device.
        Optionally skips frames to meet the specified fps.

        Args:
            path (str): The path to the video file/stream_url/v4l2 device.
            fps (int, optional): Target frames per second. If None, no frame skipping is done.

        Yields:
            numpy.ndarray: frame
        """
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open {path}")

        original_fps = cap.get(cv2.CAP_PROP_FPS)

        # Calculate the frame skip interval as a float if fps is set and original_fps is higher
        frame_interval = 1.0
        if fps is not None and original_fps > fps:
            frame_interval = original_fps / fps

        frame_index = 0
        next_frame_to_process = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame if the current index matches or exceeds the next frame to process
            if frame_index >= next_frame_to_process:
                try:
                    height, width, _ = frame.shape
                    if width != self.width or height != self.height:
                        frame = cv2.resize(frame, (self.width, self.height))

                    if self.cvt_code is not None:
                        frame = cv2.cvtColor(frame, self.cvt_code)

                    yield frame
                    next_frame_to_process += (
                        frame_interval  # Update the next frame index to process
                    )
                except Exception as e:
                    logger.error(f"Error processing frame: {e}")
                    raise e

            frame_index += 1

        cap.release()

    def get_info(self, path):
        """
        Extracts video properties.

        Args:
            path (str): The path to the video file/stream_url/v4l2 device.

        Returns:
            dict: Video properties including width, height, fps, and frame count.
        """
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
