import av
import cv2

from ..log import logger


class PyAVProducer:
    def __init__(self, width: int = None, height: int = None, format=None):
        self.width = width
        self.height = height
        self.format = "bgr24" if format is None else format

    def read(self, source, fps=None, position=0):
        """
        从视频文件/流URL/v4l2设备读取帧。
        可选择跳过帧以满足指定的fps。

        Args:
            source (str): 视频文件/流URL/v4l2设备的路径。
            fps (int, optional): 目标每秒帧数。如果为None，则不跳过任何帧。
            position (int, optional): 开始读取视频的秒数位置。

        Yields:
            numpy.ndarray: 帧
        """
        try:
            container = av.open(source)
            video_stream = next(s for s in container.streams if s.type == "video")
            original_fps = video_stream.base_rate

            # Seek to the specified position
            if position > 0:
                logger.warning(
                    "Using PyAVProducer and specifying position is not recommended because there is not yet a good solution to the problem of startup delays but it still works"
                )
                start_frame = int(position * original_fps)

            frame_interval = 1.0
            if fps is not None and original_fps > fps:
                frame_interval = original_fps / fps

            frame_index = 0
            next_frame_to_process = start_frame if position > 0 else frame_index
            for frame in container.decode(video=0):
                if frame_index >= next_frame_to_process:
                    try:
                        frame = frame.to_ndarray(format=self.format)
                        height, width, _ = frame.shape
                        if (
                            self.width is not None
                            and self.height is not None
                            and (width != self.width or height != self.height)
                        ):
                            frame = cv2.resize(frame, (self.width, self.height))

                        yield frame
                        next_frame_to_process += frame_interval
                    except Exception as e:
                        logger.error("Error processing frame", e)
                        raise e

                frame_index += 1

        except Exception as e:
            logger.error(f"Failed to open {source}", e)
            raise ValueError(f"Failed to open {source}")

        container.close()

    def get_info(self, source):
        """
        提取视频属性。

        Args:
            source (str): 视频文件/流URL/v4l2设备的路径。

        Returns:
            dict: 包含宽度、高度、fps和帧数的视频属性。
        """
        try:
            container = av.open(source)
            video_stream = next(s for s in container.streams if s.type == "video")

            width = video_stream.width
            height = video_stream.height
            fps = video_stream.base_rate  # or video_stream.average_rate

            if hasattr(video_stream, "frames"):
                frame_count = video_stream.frames
            else:
                frame_count = 0

            total_seconds = int(frame_count / fps)

            return {
                "width": width,
                "height": height,
                "fps": fps,
                "frame_count": frame_count,
                "total_seconds": total_seconds,
            }

        except av.AVError as e:
            logger.error(f"Failed to open {source}", e)
            raise ValueError(f"Failed to open {source}")
