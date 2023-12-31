import os
import streamlit as st
from .inference import Inference
from .player import Player
from .producer import OpenCVProducer
from .util import trans_position2time
from .log import logger


class StreamInferApp:
    def __init__(self, inference: Inference, save_output_path: str = "./outputs"):
        if not isinstance(inference, Inference):
            logger.error("inference must be an instance of Inference")
            return
        self.inference = inference
        self.save_output_path = save_output_path
        self.setup_ui()
        self.setup_output_ui()
        if self.need_save:
            if not os.path.exists(save_output_path):
                os.makedirs(save_output_path)
        self.annotate_frame_func = self.default_annotate_frame
        self.output_func = self.default_output

    def setup_output_ui(self):
        if not self.show_output:
            return
        self.output_widgets = {}
        algos = self.inference.list_algos()
        for idx, tab in enumerate(st.tabs(algos)):
            self.output_widgets[algos[idx]] = tab.empty()

    def setup_ui(self):
        st.set_page_config(
            page_title="StreamInfer",
            page_icon="📸",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        with st.sidebar:
            st.header("参数控制")
            self.video_path = st.text_input(
                "流URL或视频路径",
                placeholder="网络流地址/绝对路径",
                disabled=self.get_state("input_disabled", bool),
            )
            res_width, res_height = st.columns(2)
            self.width = res_width.number_input(
                "推理宽度", 320, 1920, 1920, disabled=self.get_state("input_disabled", bool)
            )
            self.height = res_height.number_input(
                "推理高度", 180, 1080, 1080, disabled=self.get_state("input_disabled", bool)
            )
            self.frame_rate = st.slider(
                "帧率", 1, 60, 30, disabled=self.get_state("input_disabled", bool)
            )
            self.color_channel = st.selectbox(
                "颜色通道", ("BGR", "RGB"), disabled=self.get_state("input_disabled", bool)
            )
            self.start_position = st.number_input(
                "起始播放位置(秒)", 0, 1000, 0, disabled=self.get_state("input_disabled", bool)
            )
            st.divider()

            self.show_frame = st.toggle(
                "展示视频帧", value=True, disabled=self.get_state("control_disabled", bool)
            )
            self.show_output = st.toggle(
                "展示推理数据", disabled=self.get_state("control_disabled", bool)
            )
            self.need_save = st.toggle(
                "保存推理结果", value=True, disabled=self.get_state("control_disabled", bool)
            )

            st.button(
                "开始推理",
                on_click=self.start_infer,
                use_container_width=True,
                disabled=self.get_state("control_disabled", bool),
            )
            if self.get_state("mode") == "infering":
                st.button(
                    "停止/返回推理",
                    on_click=self.stop_infer,
                    use_container_width=True,
                    type="primary",
                )

        st.header("推理结果")
        if self.need_save:
            st.caption("结果将保存到 ./outputs 文件夹")
        if self.show_frame:
            self.image_frame = st.image([])
        self.video_progress = st.progress(0, text="视频待加载")

    def get_state(self, key, key_type=str):
        if key not in st.session_state:
            st.session_state[key] = key_type()
        return st.session_state[key]

    def set_state(self, key, value):
        st.session_state[key] = value

    def start_infer(self):
        if not self.video_path:
            st.error("请输入视频路径")
            return
        self.set_state("control_disabled", True)
        self.set_state("input_disabled", True)
        self.set_state("mode", "infering")

    def stop_infer(self):
        self.set_state("control_disabled", False)
        self.set_state("input_disabled", False)
        self.set_state("mode", "standby")

    def fmt_output(self, result):
        output = ""
        for key, data in result.items():
            output += f"第 {key} 秒\n"
            output += "-------------------------------\n"
            output += f"{data}\n"
            output += "=======================================\n\n\n"
        return output

    def save_output(self):
        for algo_name in self.inference.list_algos():
            result = self.inference.dispatcher.get_result(algo_name)
            if result is None:
                result = ""
            path = os.path.join(self.save_output_path, f"{algo_name}.log")
            with open(path, "w+", encoding="utf-8") as f:
                f.write(self.fmt_output(result))

    def run_inference(self, clear: bool = False):
        producer = OpenCVProducer(self.width, self.height)
        player = Player(self.inference.dispatcher, producer, path=self.video_path)

        video_info = player.info
        if video_info["frame_count"] < 0:
            st.error("不能加载实时流")
            return
        total_sec = int(video_info["frame_count"] / video_info["fps"])
        total_sec_display = trans_position2time(total_sec)
        self.video_progress.progress(0, text=f"00:00:00 / {total_sec_display}")
        temp_current_time = 0

        for frame, current_frame in player.play(
            self.frame_rate, position=self.start_position
        ):
            if self.inference.dispatcher.get_current_time() != temp_current_time:
                temp_current_time = self.inference.dispatcher.get_current_time()
                self.video_progress.progress(
                    temp_current_time / total_sec,
                    text=f"{trans_position2time(temp_current_time)} / {total_sec_display}",
                )

            current_algo_names = self.inference.auto_run_specific(
                player.play_fps, current_frame
            )
            if self.show_frame:
                for algo in self.inference.list_algos():
                    _, data = self.inference.dispatcher.get_last_result(algo)
                    if data is None:
                        continue
                    frame = self.annotate_frame_func(algo, data, frame)
            if len(current_algo_names) == 0:
                yield None, -1, None, frame
            else:
                for idx, current_algo_name in enumerate(current_algo_names):
                    position, data = self.inference.dispatcher.get_last_result(
                        current_algo_name
                    )
                    if idx != len(current_algo_names) - 1:
                        display_frame = None
                    else:
                        display_frame = frame
                    yield current_algo_name, position, data, display_frame

        if self.need_save:
            self.save_output()
        self.stop_infer()
        if clear:
            self.inference.dispatcher.clear()
            st.rerun()
        else:
            st.stop()

    def default_annotate_frame(self, name, data, frame):
        return frame

    def set_annotate_frame(self, func):
        def custom_annotate_frame_wrapper(name, data, frame):
            return func(self, name, data, frame)

        self.annotate_frame_func = custom_annotate_frame_wrapper

    def default_output(self, name, position, data):
        self.output_widgets[name].text(f"{position}: {data}")

    def set_output(self, func):
        def custom_output_wrapper(name, position, data):
            return func(self, name, position, data)

        self.output_func = custom_output_wrapper

    def start(self):
        if self.get_state("mode") != "infering":
            st.stop()

        for current_algo_name, position, data, frame in self.run_inference():
            if self.show_frame and frame is not None:
                self.image_frame.image(frame, channels=self.color_channel)
            if self.show_output and current_algo_name:
                self.output_func(current_algo_name, position, data)
