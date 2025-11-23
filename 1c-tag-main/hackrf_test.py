import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
import socket
import time  # Add this import

# UDP configuration
UDP_IP = "0.0.0.0"
UDP_PORT = 7355

# Audio stream parameters
CHUNK = 512
# CHUNK = 1024 * 2
RATE = 48000

# UDP socket setup
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.setblocking(False)

# PyQtGraph setup
app = pg.mkQApp("FFT Viewer")
win = pg.GraphicsLayoutWidget(show=True, title="Real-time FFT")
win.resize(1000, 1200)  # Increased height to accommodate the waterfall plot
win.setWindowTitle('Live FFT Viewer')

# FFT plot
fft_plot = win.addPlot(title="FFT")
fft_curve = fft_plot.plot(pen='y')
fft_plot.setYRange(-100, 100)

# Set up frequency axis for FFT plot
freqs = np.fft.rfftfreq(CHUNK, d=1/RATE)
fft_plot.setLabel('bottom', 'Frequency', units='Hz')
fft_plot.setLabel('left', 'Magnitude', units='dB')

# Add a new plot for FFT norm over time
win.nextRow()
norm_plot = win.addPlot(title="FFT Norm over Time")
norm_curve = norm_plot.plot(pen='c')
norm_plot.setLabel('bottom', 'Time', units='s')
norm_plot.setLabel('left', 'FFT Norm')
# norm_plot.setYRange(1, 2)

# Add waterfall plot
win.nextRow()
waterfall_plot = win.addPlot(title="Spectrogram")
waterfall = pg.ImageItem()
waterfall_plot.addItem(waterfall)
waterfall_plot.setLabel('bottom', 'Time', units='s')
waterfall_plot.setLabel('left', 'Frequency', units='Hz')

# Set up colormap for the waterfall
colormap = pg.colormap.get('viridis')
waterfall.setColorMap(colormap)

# Variables for waterfall
waterfall_height = 100  # Number of time steps to display
waterfall_data = np.zeros((waterfall_height, CHUNK//4))  # Initialize with zeros
waterfall_pos = 0  # Current position in the waterfall

# Variables to store norm values and time
max_points = 100  # Maximum number of points to display
norm_values = []
time_values = []
start_time = time.time()  # Use time.time() instead of pg.ptime.time()

def update():
    global start_time, waterfall_pos, waterfall_data
    try:
        # Get audio data from UDP
        data, addr = sock.recvfrom(CHUNK * 2)  # *2 because each sample is 2 bytes
        data_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32767.0
        # print(data_np.shape)
        
        # compute fft
        fft_data = np.fft.rfft(data_np)
        fft_mag = np.abs(fft_data)[:len(fft_data)//2]

        # get the norm of the fft
        fft_norm = np.linalg.norm(fft_mag)
        # print(fft_norm)
        
        # convert to db scale
        fft_db = 20 * np.log10(fft_mag + 1e-10)
        
        # update FFT plot
        fft_curve.setData(np.arange(len(fft_db)), fft_db)

        # Update norm plot
        current_time = time.time() - start_time  # Use time.time() instead of pg.ptime.time()
        norm_values.append(fft_norm)
        time_values.append(current_time)

        # Limit the number of points displayed
        if len(norm_values) > max_points:
            norm_values.pop(0)
            time_values.pop(0)

        norm_curve.setData(time_values, norm_values)
        
        # Update waterfall
        # Use a subset of the FFT data for better visualization
        waterfall_data[waterfall_pos] = fft_db[:CHUNK//4]
        waterfall_pos = (waterfall_pos + 1) % waterfall_height
        
        # Roll the data to show newest at the top
        rolled_data = np.roll(waterfall_data, -waterfall_pos, axis=0)
        # Use the data without transposing to keep frequency on y-axis
        waterfall.setImage(rolled_data, autoLevels=False, levels=[-80, 20])
        
        # Set the correct scale for the axes
        freq_max = freqs[CHUNK//4 - 1]  # Maximum frequency
        waterfall.setRect(QtCore.QRectF(0, 0, waterfall_height, freq_max))

    except BlockingIOError:
        print("BlockingIOError")
        pass
    except Exception as e:
        print(e)
        pass

# Update timer
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(10)  # 10ms update rate (100 fps)

if __name__ == '__main__':
    pg.exec()