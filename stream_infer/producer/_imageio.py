import imageio
import cv2

from ..log import logger


class ImageIOProducer:
    def __init__(self, width: int, height: int, cvt_code=None):
        self.width = width
        self.height = height
        self.cvt_code = cvt_code

    def read(self, path, fps=None):
        """
        Reads frames from a video file.

        Args:
            path (str): The path to the video file.
                e.g., "/path/to/video.mp4"

        Yields:
            numpy.ndarray: frame
        """
        try:
            reader = imageio.get_reader(path)
        except Exception as e:
            logger.error(f"Error reading video: {e}")
            raise e

        original_fps = reader.get_meta_data()["fps"]

        # Calculate the frame skip interval as a float if fps is set and original_fps is higher
        frame_interval = 1.0
        if fps is not None and original_fps > fps:
            frame_interval = original_fps / fps

        frame_index = 0
        next_frame_to_process = 0
        for frame in reader:
            try:
                height, width, _ = frame.shape
                if width != self.width or height != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))

                if self.cvt_code is not None:
                    frame = cv2.cvtColor(frame, self.cvt_code)

                # cv2.imwrite("frame.jpg", frame)
                yield frame
                next_frame_to_process += (
                    frame_interval  # Update the next frame index to process
                )
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                raise e

            frame_index += 1

    def get_info(self, path):
        reader = imageio.get_reader(path)
        fps = reader.get_meta_data()["fps"]
        frame_count = reader.count_frames()

        frame = reader.get_next_data()
        height, width, _ = frame.shape

        reader.close()

        return {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
        }
