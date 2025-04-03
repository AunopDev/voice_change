import sys
import sounddevice as sd
import numpy as np
from scipy.signal import butter, lfilter
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QComboBox, QCheckBox, QPushButton, QFrame, QSlider, QHBoxLayout
from PyQt5.QtCore import Qt, QTimer
import logging

logging.basicConfig(level=logging.INFO)

# Butterworth Highpass Filter
def butter_highpass(cutoff, fs, order=6):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(data, cutoff=100, fs=16000, order=6):
    b, a = butter_highpass(cutoff, fs, order)
    return lfilter(b, a, data)

def calculate_rms_dBFS(data):
    rms = np.sqrt(np.mean(np.square(data)))
    dBFS = 20 * np.log10(rms + 1e-6)
    return dBFS

# VoiceApp GUI Application
class VoiceApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("แปลงเสียงแบบเรียลไทม์")
        self.setGeometry(100, 100, 600, 500)

        # Initialize state variables
        self.is_running = False  # Add this line
        self.is_listening = False  # Add this line
        self.stream = None
        self.output_stream = None

        self.initUI()

    def create_slider(self, label, min_val, max_val, default_val, description):
        slider_layout = QVBoxLayout()
        slider_label = QLabel(f"{label}")
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(default_val)
        
        # Create a label to show the current value
        value_label = QLabel(f"ค่าปัจจุบัน: {default_val}")
        
        # Update value when slider value changes
        slider.valueChanged.connect(lambda: self.update_slider_value(slider, value_label, description))
        
        slider_layout.addWidget(slider_label)
        slider_layout.addWidget(slider)
        slider_layout.addWidget(value_label)

        container = QWidget()  # Create a new QWidget here, which is a container for the layout
        container.setLayout(slider_layout)  # Set the layout to the container
        
        # Return the container and the slider for later use
        return container, slider, value_label

    def initUI(self):
        # Initialize variables for audio processing
        self.noise_gate_enabled = False
        self.noise_gate_threshold = 0.5  # Default threshold for Noise Gate
        self.noise_suppression_enabled = False
        self.noise_suppression_level = 0.5  # Default level for Noise Suppression
        self.compressor_enabled = False
        self.compressor_ratio = 2.0  # Default ratio for Compressor
        self.equalizer_enabled = False
        self.equalizer_settings = {100: 0, 1000: 0, 5000: 0}  # Default EQ settings
        self.limiter_enabled = False
        self.limiter_threshold = 0  # Default limiter threshold in dB

        layout = QVBoxLayout()
        
        self.device_select = self.create_device_select()
        layout.addWidget(self.device_select)
        
        self.listen_checkbox = self.create_checkbox("ฟังเสียงตัวเอง")
        layout.addWidget(self.listen_checkbox)
        
        # Noise Gate
        self.noise_gate_checkbox = self.create_checkbox("เปิดใช้งาน Noise Gate")
        self.noise_gate_checkbox.stateChanged.connect(lambda: self.toggle_feature(self.noise_gate_checkbox, self.noise_gate_slider_container))
        layout.addWidget(self.noise_gate_checkbox)
        self.noise_gate_slider_container, self.noise_gate_slider, self.noise_gate_value_label = self.create_slider(
            "Noise Gate: กำหนดระดับเสียงขั้นต่ำก่อนถูกตัดออก", 0, 100, 50, "Noise Gate"
        )
        self.noise_gate_slider_container.setVisible(False)
        layout.addWidget(self.noise_gate_slider_container)
        
        # Noise Suppression
        self.noise_suppression_checkbox = self.create_checkbox("เปิดใช้งาน Noise Suppression")
        self.noise_suppression_checkbox.stateChanged.connect(lambda: self.toggle_feature(self.noise_suppression_checkbox, self.noise_suppression_slider_container))
        layout.addWidget(self.noise_suppression_checkbox)
        self.noise_suppression_slider_container, self.noise_suppression_slider, self.noise_suppression_value_label = self.create_slider(
            "Noise Suppression: ลดเสียงรบกวน", 0, 100, 50, "Noise Suppression"
        )
        self.noise_suppression_slider_container.setVisible(False)
        layout.addWidget(self.noise_suppression_slider_container)
        
        # Compressor
        self.compressor_checkbox = self.create_checkbox("เปิดใช้งาน Compressor")
        self.compressor_checkbox.stateChanged.connect(lambda: self.toggle_feature(self.compressor_checkbox, self.compressor_slider_container))
        layout.addWidget(self.compressor_checkbox)
        self.compressor_slider_container, self.compressor_slider, self.compressor_value_label = self.create_slider(
            "Compressor: ปรับไดนามิกเสียง", 0, 100, 50, "Compressor"
        )
        self.compressor_slider_container.setVisible(False)
        layout.addWidget(self.compressor_slider_container)
        
        # Equalizer
        self.equalizer_checkbox = self.create_checkbox("เปิดใช้งาน Equalizer")
        self.equalizer_checkbox.stateChanged.connect(lambda: self.toggle_feature(self.equalizer_checkbox, self.equalizer_slider_container))
        layout.addWidget(self.equalizer_checkbox)
        self.equalizer_slider_container = QWidget()
        eq_layout = QVBoxLayout()
        self.equalizer_sliders = []
        for band in [100, 1000, 5000]:
            eq_container, eq_slider, eq_value_label = self.create_slider(
                f"EQ {band}Hz: ปรับเสียงย่าน {band}Hz", -10, 10, 0, f"EQ {band}Hz"
            )
            self.equalizer_sliders.append((eq_slider, eq_value_label))
            eq_layout.addWidget(eq_container)
        self.equalizer_slider_container.setLayout(eq_layout)
        self.equalizer_slider_container.setVisible(False)
        layout.addWidget(self.equalizer_slider_container)
        
        # Limiter
        self.limiter_checkbox = self.create_checkbox("เปิดใช้งาน Limiter")
        self.limiter_checkbox.stateChanged.connect(lambda: self.toggle_feature(self.limiter_checkbox, self.limiter_slider_container))
        layout.addWidget(self.limiter_checkbox)
        self.limiter_slider_container, self.limiter_slider, self.limiter_value_label = self.create_slider(
            "Limiter: จำกัดระดับเสียงสูงสุด", -10, 10, 0, "Limiter"
        )
        self.limiter_slider_container.setVisible(False)
        layout.addWidget(self.limiter_slider_container)
        
        # Toggle Button
        self.toggle_button = QPushButton("เริ่มทำงาน", self)
        self.toggle_button.clicked.connect(self.toggle_audio)
        layout.addWidget(self.toggle_button)
        
        # Status Frame
        self.status_frame = self.create_status_frame()
        layout.addWidget(self.status_frame)
        
        # Main Widget
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def toggle_feature(self, checkbox, slider_container):
        slider_container.setVisible(checkbox.isChecked())
        feature_name = checkbox.text().split(" ")[-1]
        setattr(self, f"{feature_name.lower()}_enabled", checkbox.isChecked())

    def create_device_select(self):
        device_select = QComboBox(self)
        device_select.addItems(self.get_devices())
        device_select.currentIndexChanged.connect(self.restart_audio)
        return device_select

    def create_checkbox(self, text):
        checkbox = QCheckBox(text, self)
        checkbox.stateChanged.connect(self.update_settings)
        return checkbox

    def update_slider_value(self, slider, value_label, description):
        value = slider.value()
        value_label.setText(f"ค่าปัจจุบัน: {value}")
        if description == "Noise Gate":
            self.noise_gate_threshold = value / 100  # Normalize to 0-1
        elif description == "Noise Suppression":
            self.noise_suppression_level = value / 100  # Normalize to 0-1
        elif description == "Compressor":
            self.compressor_ratio = value / 10  # Normalize to a ratio
        elif description.startswith("EQ"):
            band = int(description.split()[1][:-2])  # Extract frequency band
            self.equalizer_settings[band] = value  # Store EQ gain in dB
        elif description == "Limiter":
            self.limiter_threshold = value  # Store limiter threshold in dB

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
        return [device['name'] for device in devices if device['max_input_channels'] > 0]

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
        if not device_name:
            self.status_label.setText("ไม่พบอุปกรณ์เสียงที่ใช้งานได้")
            return
        
        device_info = next((device for device in sd.query_devices() if device['name'] == device_name), None)
        if not device_info:
            self.status_label.setText("อุปกรณ์เสียงไม่ถูกต้อง")
            return

        self.status_label.setText(f"เริ่มรับเสียงจาก {device_name}...")
        self.stream = sd.InputStream(device=device_info['index'], channels=1, samplerate=16000, dtype=np.float32, callback=self.audio_callback)
        self.stream.start()
        self.output_stream = sd.OutputStream(channels=1, samplerate=16000, dtype=np.float32)
        self.output_stream.start()

    def stop_audio(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        if self.output_stream:
            self.output_stream.stop()
            self.output_stream.close()
            self.output_stream = None
        self.status_label.setText("หยุดทำงาน")

    def update_settings(self):
        self.is_listening = self.listen_checkbox.isChecked()

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        if indata is None or indata.size == 0:
            return

        # Copy the input data to avoid modifying the original
        processed_data = np.copy(indata)

        # Apply Highpass Filter to remove low-frequency noise
        processed_data = highpass_filter(processed_data, cutoff=100, fs=16000)

        # Apply Noise Gate
        if self.noise_gate_enabled:
            rms = np.sqrt(np.mean(np.square(processed_data)))
            if rms < self.noise_gate_threshold:
                processed_data = np.zeros_like(processed_data)
            else:
                self.status_label.setText(f"Noise Gate: RMS={rms:.2f}, Threshold={self.noise_gate_threshold:.2f}")

        # Apply Noise Suppression
        if self.noise_suppression_enabled:
            processed_data *= (1 - self.noise_suppression_level)
            self.status_label.setText(f"Noise Suppression: Level={self.noise_suppression_level:.2f}")

        # Apply Compressor
        if self.compressor_enabled:
            threshold = 0.5  # Example threshold
            processed_data = np.where(
                processed_data > threshold,
                threshold + (processed_data - threshold) / self.compressor_ratio,
                processed_data
            )
            self.status_label.setText(f"Compressor: Ratio={self.compressor_ratio:.2f}, Threshold={threshold:.2f}")

        # Apply Equalizer
        if self.equalizer_enabled:
            for band, gain in self.equalizer_settings.items():
                processed_data *= 10**(gain / 20)
            self.status_label.setText(f"Equalizer: Settings={self.equalizer_settings}")

        # Apply Limiter
        if self.limiter_enabled:
            max_val = 10**(self.limiter_threshold / 20)
            processed_data = np.clip(processed_data, -max_val, max_val)
            self.status_label.setText(f"Limiter: Threshold={self.limiter_threshold:.2f}")

        # Update dBFS display
        dBFS = calculate_rms_dBFS(processed_data)
        QTimer.singleShot(0, lambda: self.dB_label.setText(f"ระดับเสียง: {dBFS:.2f} dBFS"))

        # Convert processed_data to float32 for output_stream
        processed_data = processed_data.astype(np.float32)

        # Output audio if listening is enabled
        if self.is_listening and self.output_stream and self.output_stream.active:
            try:
                self.output_stream.write(processed_data)
            except sd.PortAudioError as e:
                logging.error(f"เกิดข้อผิดพลาด: {e}")

    def closeEvent(self, event):
        self.stop_audio()
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VoiceApp()
    window.show()
    sys.exit(app.exec_())
