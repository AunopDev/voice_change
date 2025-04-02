import sys
import sounddevice as sd
import numpy as np
import noisereduce as nr
from scipy.signal import butter, lfilter
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QComboBox, QCheckBox, QPushButton, QFrame
from PyQt5.QtCore import Qt, QTimer  # ใช้ QTimer สำหรับอัปเดต UI ใน Main Thread
import logging

logging.basicConfig(level=logging.INFO)


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def highpass_filter(data, cutoff=100, fs=16000, order=5):
    b, a = butter_highpass(cutoff, fs, order)
    return lfilter(b, a, data)


def calculate_rms_dBFS(data):
    rms = np.sqrt(np.mean(np.square(data)))
    dBFS = 20 * np.log10(rms + 1e-6)  # ป้องกัน -inf
    return dBFS


class VoiceApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("แปลงเสียงแบบเรียลไทม์")
        self.setGeometry(100, 100, 500, 400)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.device_select = self.create_device_select()
        layout.addWidget(self.device_select)

        self.listen_checkbox = self.create_checkbox("ฟังเสียงตัวเอง")
        self.noise_checkbox = self.create_checkbox("ตัดเสียงรบกวน")
        layout.addWidget(self.listen_checkbox)
        layout.addWidget(self.noise_checkbox)

        self.toggle_button = QPushButton("เริ่มทำงาน", self)
        self.toggle_button.clicked.connect(self.toggle_audio)
        layout.addWidget(self.toggle_button)

        self.status_frame = self.create_status_frame()
        layout.addWidget(self.status_frame)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.is_running = False
        self.is_listening = False
        self.is_noise_reduction = False
        self.noise_sample = None
        self.stream = None
        self.output_stream = None

    def create_device_select(self):
        device_select = QComboBox(self)
        device_select.addItems(self.get_devices())
        device_select.currentIndexChanged.connect(self.restart_audio)
        return device_select

    def create_checkbox(self, text):
        checkbox = QCheckBox(text, self)
        checkbox.stateChanged.connect(self.update_settings)
        return checkbox

    def create_status_frame(self):
        status_frame = QFrame(self)
        status_frame.setFrameShape(QFrame.StyledPanel)
        status_frame.setStyleSheet("background-color: #f0f0f0; padding: 10px;")

        status_layout = QVBoxLayout()
        self.status_label = QLabel("สถานะ: พร้อมทำงาน", self)
        status_layout.addWidget(self.status_label)

        self.dB_label = QLabel("ระดับเสียง: 0.00 dBFS", self)
        self.dB_label.setStyleSheet("color: green; font-weight: bold;")
        status_layout.addWidget(self.dB_label)

        status_frame.setLayout(status_layout)
        return status_frame

    def get_devices(self):
        devices = sd.query_devices()
        input_devices = [device['name']
                         for device in devices if device['max_input_channels'] > 0]
        if not input_devices:
            self.status_label.setText("ไม่พบอุปกรณ์เสียงที่ใช้งานได้")
        return input_devices

    def restart_audio(self):
        if self.stream and self.stream.active:
            self.stop_audio()
        if self.is_running:
            self.start_audio()

    def toggle_audio(self):
        if self.is_running:
            self.stop_audio()
            self.toggle_button.setText("เริ่มทำงาน")
        else:
            self.start_audio()
            self.toggle_button.setText("หยุดทำงาน")
        self.is_running = not self.is_running

    def start_audio(self):
        device_name = self.device_select.currentText()
        device_info = next(device for device in sd.query_devices()
                           if device['name'] == device_name)
        self.status_label.setText(f"เริ่มรับเสียงจาก {device_name}...")

        self.stream = sd.InputStream(device=device_info['index'],
                                     channels=1, samplerate=16000, dtype=np.float32,
                                     callback=self.audio_callback)
        self.stream.start()
        self.output_stream = sd.OutputStream(
            channels=1, samplerate=16000, dtype=np.float32)
        self.output_stream.start()

    def stop_audio(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
        if self.output_stream:
            self.output_stream.stop()
            self.output_stream.close()
        self.status_label.setText("หยุดทำงาน")

    def update_settings(self):
        self.is_listening = self.listen_checkbox.isChecked()
        self.is_noise_reduction = self.noise_checkbox.isChecked()
        if self.is_listening:
            self.status_label.setText("กำลังฟังเสียงตัวเอง...")
        elif self.is_noise_reduction:
            self.status_label.setText("กำลังตัดเสียงรบกวน...")
        else:
            self.status_label.setText("พร้อมทำงาน")

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)

        indata = indata.astype(np.float32)

        if self.noise_sample is None:
            self.noise_sample = np.mean(indata, axis=0)

        if self.is_noise_reduction:
            try:
                indata = nr.reduce_noise(
                    y=indata.flatten(), sr=16000, y_noise=self.noise_sample, prop_decrease=0.8)
                indata = highpass_filter(indata, cutoff=100, fs=16000)
                indata = indata.astype(np.float32)
            except Exception as e:
                print(f"Error reducing noise: {e}", file=sys.stderr)

        dBFS = calculate_rms_dBFS(indata)
        QTimer.singleShot(0, lambda: self.dB_label.setText(
            f"ระดับเสียง: {dBFS:.2f} dBFS"))

        if self.is_listening and self.output_stream:
            try:
                self.output_stream.write(indata)
            except Exception as e:
                print(f"Error playing sound: {e}", file=sys.stderr)

    def closeEvent(self, event):
        self.stop_audio()
        super().closeEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VoiceApp()
    window.show()
    sys.exit(app.exec_())
