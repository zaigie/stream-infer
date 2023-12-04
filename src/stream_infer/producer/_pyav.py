import av
import cv2


class PyAVProducer:
    def __init__(self, width: int, height: int, format: str = "rgb24"):
        self.width = width
        self.height = height
        self.format = format

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

            # Calculate the frame skip rate if fps is set and original_fps is higher
            skip_rate = 1
            if fps is not None and original_fps > fps:
                skip_rate = int(original_fps // fps)

            frame_index = 0
            for frame in container.decode(video=0):
                # Skip frames based on the calculated skip rate
                if frame_index % skip_rate == 0:
                    try:
                        frame = frame.to_ndarray(format=self.format)
                        height, width, channel = frame.shape
                        if width != self.width or height != self.height:
                            frame = cv2.resize(frame, (self.width, self.height))

                        yield frame
                    except Exception as e:
                        break

                frame_index += 1

        except av.AVError as e:
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
