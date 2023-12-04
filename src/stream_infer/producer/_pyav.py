import av
import cv2


class PyAVProducer:
    def __init__(self, width: int, height: int, format: str = "rgb24"):
        self.width = width
        self.height = height
        self.format = format

    def read(self, path):
        """
        Reads frames from a video file/stream_url/v4l2 device.

        Args:
            path (str): The path to the video file/stream_url/v4l2 device.
                eg. "/path/to/video.mp4" or "rtmp://localhost/live" or "/dev/video0"

        Yields:
            numpy.ndarray: frame
        """
        try:
            container = av.open(path)
        except av.AVError as e:
            raise ValueError(f"Failed to open {path}: {e}")

        for frame in container.decode(video=0):
            try:
                frame = frame.to_ndarray(format=self.format)
                height, width, channel = frame.shape
                if width != self.width or height != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))

                # cv2.imwrite("frame.jpg", frame)
                yield frame
            except Exception as e:
                break

        container.close()

    def get_info(self, path):
        try:
            container = av.open(path)
            video_stream = next(s for s in container.streams if s.type == "video")

            width = video_stream.width
            height = video_stream.height
            fps = video_stream.average_rate  # or video_stream.base_rate

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
            print(f"Failed to open {path}: {e}")
