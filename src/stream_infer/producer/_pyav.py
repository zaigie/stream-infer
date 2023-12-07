import av
import cv2

from ..log import logger


class PyAVProducer:
    def __init__(self, width: int, height: int, format=None):
        self.width = width
        self.height = height
        self.format = "bgr24" if format is None else format

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
        try:
            container = av.open(path)
            video_stream = next(s for s in container.streams if s.type == "video")
            original_fps = video_stream.base_rate

            # Calculate the frame skip interval as a float if fps is set and original_fps is higher
            frame_interval = 1.0
            if fps is not None and original_fps > fps:
                frame_interval = original_fps / fps

            frame_index = 0
            next_frame_to_process = 0
            for frame in container.decode(video=0):
                # Process frame if the current index matches or exceeds the next frame to process
                if frame_index >= next_frame_to_process:
                    try:
                        frame = frame.to_ndarray(format=self.format)
                        height, width, _ = frame.shape
                        if width != self.width or height != self.height:
                            frame = cv2.resize(frame, (self.width, self.height))

                        yield frame
                        next_frame_to_process += (
                            frame_interval  # Update the next frame index to process
                        )
                    except Exception as e:
                        logger.error(f"Error processing frame: {e}")
                        raise e

                frame_index += 1

        except av.AVError as e:
            logger.error(f"Failed to open {path}: {e}")
            raise ValueError(f"Failed to open {path}: {e}")

        container.close()

    def get_info(self, path):
        """
        Extracts video properties.

        Args:
            path (str): The path to the video file/stream_url/v4l2 device.

        Returns:
            dict: Video properties including width, height, fps, and frame count.
        """
        try:
            container = av.open(path)
            video_stream = next(s for s in container.streams if s.type == "video")

            width = video_stream.width
            height = video_stream.height
            fps = video_stream.base_rate  # or video_stream.average_rate

            if hasattr(video_stream, "frames"):
                frame_count = video_stream.frames
            else:
                frame_count = -1

            return {
                "width": width,
                "height": height,
                "fps": fps,
                "frame_count": frame_count,
            }

        except av.AVError as e:
            raise ValueError(f"Failed to open {path}: {e}")
