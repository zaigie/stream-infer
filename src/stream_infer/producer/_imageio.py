import imageio
import cv2


class ImageIOProducer:
    def __init__(self, width: int, height: int, cvt_code: int = cv2.COLOR_RGB2BGR):
        self.width = width
        self.height = height
        self.cvt_code = cvt_code

    def read(self, path):
        """
        Reads frames from a video file.

        Args:
            path (str): The path to the video file.
                e.g., "/path/to/video.mp4"

        Yields:
            numpy.ndarray: frame
        """
        reader = imageio.get_reader(path)
        for frame in reader:
            try:
                height, width, channel = frame.shape
                if width != self.width or height != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))

                frame = cv2.cvtColor(frame, self.cvt_code)

                # cv2.imwrite("frame.jpg", frame)
                yield frame
            except Exception as e:
                print(f"Error processing frame: {e}")
                break

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
