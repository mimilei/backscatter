from nanovna import *
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib
import time
import cv2
import skrf
import glob
from tqdm import tqdm
import joblib
import threading

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def get_vna_devices(opt):
    nv = []
    ports = getport_all()
    cnt = 0
    for p in ports:
        device = NanoVNAV2(p)
        # device.set_frequencies(start = opt.start, stop = opt.stop, points = opt.points)
        nv.append(VNAObject(dotdict(opt.__dict__.copy()), device))
    return nv

class VNAObject():
    def __init__(self, opt, device):
        self.device = device
        self.opt = opt
        self.calON = True
        self.port_number = 0
        self.init_vna()
        if self.calON:
            self.set_calibration()

    def init_vna(self, start=None, stop=None, points=None):
        if start is None:
            start = self.opt.start
        if stop is None:
            stop = self.opt.stop
        if points is None:
            points = self.opt.points
        self.opt.start, self.opt.stop, self.opt.points = start, stop, points
        self.device.set_frequencies(start, stop, points)
        self.device.set_sweep(start, stop, points)

        # self.device.serial.write([0x20,0x44,0x01])
        self.device.serial.write([0x20,0x40,0x01])
        self.device.serial.write([0x20,0xE0,0x01])
        self.device.serial.write([0x20,0xE1,0x01])

    def get_data(self, port):
        s = self.device.scan()
        if self.calON:
            s = self.calibrate(s)
        return s[port]

    def get_data_(self):
        s = self.device.scan()
        if self.calON:
            s = self.calibrate(s)
        return s[0], s[1]

    def set_calibration(self):
        step = self.cali_valid_test()
        # load networks of the raw calibration standard measurements
        # name = ["litevna"]
        # name = ["dipole-small"]
        # name = ["litevna_vivaldi"]
        # name = ["litevna+rfswitch-ch1","litevna+rfswitch-ch2"]
        name = ["litevna+amp+rfswitch-ch1","litevna+amp+rfswitch-ch2"]

        cal = []
        for i in range(len(name)):
            short_raw = skrf.Network(f'./utils/{name[i]}/cal_short.s2p')
            open_raw = skrf.Network(f'./utils/{name[i]}/cal_open.s2p')
            match_raw = skrf.Network(f'./utils/{name[i]}/cal_match.s2p')
            thru_raw = skrf.Network(f'./utils/{name[i]}/cal_thru.s2p')

            f = short_raw.frequency
            f_cal = np.arange(f.start,f.stop+1,f.step)
            f_vna = self.device.frequencies
            indices = np.where(np.in1d(f_cal, f_vna))[0]

            short_raw = skrf.Network(frequency=f_vna, s=short_raw.s[indices])
            open_raw = skrf.Network(frequency=f_vna, s=open_raw.s[indices])
            match_raw = skrf.Network(frequency=f_vna, s=match_raw.s[indices])
            thru_raw = skrf.Network(frequency=f_vna, s=thru_raw.s[indices])

            # create an ideal 50-Ohm line for the short, open, match and through reference responses ("ideals")
            line = skrf.DefinedGammaZ0(frequency=short_raw.frequency, Z0=50)
            # create and run the calibration
            _cal = skrf.calibration.TwoPortOnePath(ideals=[line.short(nports=2), line.open(nports=2), line.match(nports=2), line.thru()],
                                measured=[short_raw, open_raw, match_raw, thru_raw],
                                n_thrus=1, source_port=1)
            _cal.run()
            cal.append(_cal)
        self.cal = cal

    def calibrate(self,s):
        data = np.zeros((len(s[0]), 2, 2), dtype=complex)
        data[:,0,0], data[:,1,0] = s[0], s[1]
        data = skrf.Network(frequency=self.device.frequencies, s=data)
        data = self.cal[self.port_number].apply_cal((data)).s
        data = np.array(data)
        data = (data[:,0,0],data[:,1,0])
        return data

    def cali_valid_test(self):
        step = self.device.frequencies[1]-self.device.frequencies[0]
        valid = step - int(step)
        valid = valid == 0
        if not valid:
            print("The frequencies should be in step of 1e6x.")
            exit()
        return step

class VNAStream():
    def __init__(self, opt, devices):
        self.init_params()
        self.nv = devices
        self.opt = opt
        self.setup_rf_switcher()

        if self.opt.plot:
            self.init_plot()
        else:
            while True:     
                self.data_store()
                self.switch_rf()
                self.print_fps()

    def setup_rf_switcher(self):
        try:
            self.max_port = 2
            self.port_number = 0
            # self.ser_switch = serial.Serial('/dev/cu.usbmodem1101', 9600, timeout=1)
            # self.ser_switch = serial.Serial('/dev/cu.usbmodem11401', 9600, timeout=1)
            self.ser_switch = serial.Serial('/dev/cu.usbmodem1401', 9600, timeout=1)
            print("=====> RF Switcher connected")
        except Exception as e:
            self.ser_switch = None
            self.max_port = 1
            self.port_number = 0
            print(e)
            print("=====> No RF Switcher connection")

    def switch_rf(self):
        if self.ser_switch is not None:
            # self.port_number = 1
            self.port_number = (self.port_number + 1) % self.max_port
            self.nv[0].port_number = self.port_number
            self.ser_switch.write(str(self.port_number+1).encode())
            self.ser_switch.flush()
            time.sleep(0.001)
            # print(self.get_status())

    def get_status(self):
        """
        Get current port status
        
        Returns:
            str: Response from Arduino
        """
        self.ser_switch.write(b's')
        self.ser_switch.flush()
        
        return self.read_response()

    def read_response(self):
        """
        Read response from Arduino
        
        Returns:
            str: Response message
        """
        response = ""
        while self.ser_switch.in_waiting > 0:
            line = self.ser_switch.readline().decode('utf-8').strip()
            response += line + "\n"
        return response.strip()

    def init_params(self):
        self.fig = None
        self.isSpectrogram = False  # Make this default
        self.isRawImp = False
        self.isRecord = False
        self.mode = "start"
        self.s_data = []
        self.timestamp = []
        self.highlight_freq = None
        self.starttime = time.time()
        self.fps = 1
        self.win_t = 0.5 # calibration window size; seconds
        self.isTemporalCalibration = False

    def on_release(self, event):
        mkey = event.key
        if mkey == "o":
            self.isRecord = False
            # self.jc.stop()
            print("End to save")

    def on_close(self, event):
        if self.s_data:
            self.save_data()
            print("Quit with save")

    def on_press(self, event, force_key=None):
        mkey = event.key
        print(mkey)
        #--------------------------#
        #        Data Setup        #
        #--------------------------#

        if mkey == 'q':
            time.sleep(0.3)
        if mkey == 'i':
            self.isRawImp = not self.isRawImp
        # time data
        elif mkey == "t":
            self.opt.timedomain = not self.opt.timedomain
            if not self.opt.timedomain:
                self.isSpectrogram = False
            self.init_plot()
        elif mkey == "p":
            self.isSpectrogram = not self.isSpectrogram
            if self.isSpectrogram:
                self.opt.timedomain = True
            else:
                self.opt.timedomain = False
            self.init_plot()
        elif mkey == "c": # classification mode
            if self.mode == "start":
                self.predictions = []
                self.labels = ["Human hand", "None", "Long tag", "Middle tag", "Short tag"]
                self.clf = joblib.load("./models/model.pkl")
                self.mode = "c"
            else:
                self.mode = "start"
        elif mkey == "j":
            self.calibration_just_toggle()
        elif mkey == "v":
            self.calibration()
        elif mkey == "r":
            print("refresh training data buffer")
            self.s_data = []
        elif mkey == "S":
            self.save_data()
            print("Data saved")
        elif mkey == "o":
            self.isRecord = not self.isRecord
            if self.isRecord:
                print("Recording...")
            else:
                print("Recording stop")

        print(self.mode)

    def save_data(self):
        s = np.array(self.s_data)
        s11 = s[:,:,:self.opt.points]
        s21 = s[:,:,self.opt.points:]
        path = "data/"
        filenames = glob.glob(path+"data_*.npz")
        for filen in range(1,300):
            filename = path+'data_{}'.format(filen)
            if not np.any(np.isin(filenames,filename+'.npz')):
                break

        np.savez(filename+'.npz',
                s11=s11,
                s21=s21,
                ts=self.timestamp,
                start_freq=[self.nv[0].opt.start]*len(s11),
                stop_freq=[self.nv[0].opt.stop]*len(s11))
        self.s_data = []
        self.timestamp = []

    def calibration_just_toggle(self):
        self.isTemporalCalibration = not self.isTemporalCalibration    

    def calibration(self):
        self.isTemporalCalibration = not self.isTemporalCalibration
        if self.opt.timedomain:
            # self.default_value = [np.zeros(self.num_lines)]
            self.default_value = []
            for _ in range(self.max_port):
                self.default_value.append([np.zeros(self.num_lines)])
        else:
            # self.default_value = [np.zeros((self.num_lines, self.opt.points))]
            self.default_value = []
            for _ in range(self.max_port):
                self.default_value.append([np.zeros((self.num_lines, self.opt.points))])
        # for i in range(self.num_lines):
        #     if self.opt.timedomain:
        #         recent_y = self.line[i].get_ydata()[-1]
        #         if not recent_y!=recent_y:
        #             self.default_value[0][i] += recent_y
        #     else:
        #         if np.sum(np.abs(self.default_value[0][i])) == 0:
        #             self.default_value[0][i] = self.line[i].get_ydata()
        #         else:
        #             self.default_value[0][i] = np.zeros(self.opt.points)

    def init_plot(self):
        self.num_lines = self.opt.points*len(self.nv)*4
        if self.fig is None:
            # Create a subplot for each port in the RF switcher
            self.fig, self.axes = plt.subplots(self.max_port, 2, 
                                              gridspec_kw={'width_ratios': [1, 1]},
                                              figsize=(8, 5))
            
            # If there's only one port, axes will be 1D, so reshape to 2D for consistency
            if self.max_port == 1:
                self.axes = np.array([self.axes])
            
            self.fig.canvas.mpl_connect('key_press_event', self.on_press)
            self.fig.canvas.mpl_connect('close_event', self.on_close)
            self.fig.patch.set_facecolor('black')
            anim = animation.FuncAnimation(self.fig, self.animate, interval=1, blit=False)
            
            # Set background color for all subplots
            for row in range(self.max_port):
                for col in range(2):
                    self.axes[row, col].set_facecolor('black')
            
            n = self.skrf_network(np.zeros(self.opt.points))
            if self.opt.smith:
                # Only add Smith chart to the first port's subplot
                n.plot_s_smith(ax=self.axes[0, 1], chart_type='z', draw_labels=True)
                self.axes[0, 1].set_facecolor("white")
                self.axes[0, 1].set_xlim(-1.1, 1.1)
                self.axes[0, 1].set_ylim(-1.1, 1.1)
            cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

        if self.opt.timedomain:
            if self.isRawImp:
                self.ymax, self.ymin = 80, -80
            else:
                self.ymax, self.ymin = 3, -3
            start, stop = 0, int(self.fps*5) # 5 seconds
            num = stop - start
            self.num_lines = self.opt.points*len(self.nv)*4
        else:
            if self.isRawImp:
                self.ymax, self.ymin = 80, -80
            else:
                self.ymax, self.ymin = -0, -25
            start, stop = self.opt.start, self.opt.stop
            num = self.opt.points
            self.num_lines = len(self.nv)*4

        cm = plt.get_cmap('gist_ncar')
        self.color = [cm(1.*i/self.num_lines) for i in range(self.num_lines)]
        self.color[0] = (0.2, 0.2, 1.0, 1.0)

        # Clear all axes and set up for each port
        for port in range(self.max_port):
            for i in range(2):  # Two columns (S11 and S21)
                self.axes[port, i].cla()
                
                if self.opt.timedomain:
                    self.axes[port, i].set_xlim(start, stop)
                    self.axes[port, i].set_xticks([start, stop])
                    self.axes[port, i].set_xticklabels([f"{start/self.fps:.1f}", f"{stop/self.fps:.1f}"])
                    self.axes[port, i].set_ylim(self.ymin, self.ymax)
                    self.axes[port, i].set_title(f"Port {port+1} - S{i+1}1", fontsize=20, c='white')
                    if port == self.max_port-1:  # Only add xlabel to bottom row
                        self.axes[port, i].set_xlabel("Time (s)", fontsize=15, c='white')
                else:
                    self.axes[port, i].set_xlim(start, stop)
                    self.axes[port, i].set_ylim(self.ymin, self.ymax)
                    self.axes[port, i].set_title(f"Port {port+1} - S{i+1}1", fontsize=20, c='white')
                    if port == self.max_port-1:  # Only add xlabel to bottom row
                        self.axes[port, i].set_xlabel("Frequency (Hz)", fontsize=15, c='white')
                
                # Add legend only to the first row
                if port == 0 and not self.opt.timedomain:
                    self.axes[port, i].legend(["Mag(dB)", "Phase(°)"], loc='upper right', fontsize=10)
                
                # Set tick colors for all axes
                self.axes[port, i].tick_params(colors='white', which='both')

        # Create line objects for all ports
        self.line = []
        for port in range(self.max_port):
            port_lines = []
            for i in range(self.num_lines):
                idx = 0 if i < self.num_lines/2 else 1
                color = self.color[i]
                if self.isSpectrogram:
                    color = (color[0], color[1], color[2], 0)
                lineobj, = self.axes[port, idx].plot(np.linspace(start, stop, num),
                                    np.ones(num).astype(float)*np.nan,
                                    lw=4, color=color)
                port_lines.append(lineobj)
            self.line.append(port_lines)

        # Initialize default values for all ports
        if self.opt.timedomain:
            self.default_value = []
            for _ in range(self.max_port):
                self.default_value.append([np.zeros(self.num_lines)])
        else:
            self.default_value = []
            for _ in range(self.max_port):
                self.default_value.append([np.zeros((self.num_lines, self.opt.points))])

        # Set up spectrograms if needed
        if self.isSpectrogram:
            self.sptrgrm = []
            self.Sxx = []
            for port in range(self.max_port):
                port_sptrgrm = []
                port_Sxx = []
                for i in range(2):  # Two columns (S11 and S21)
                    self.axes[port, i].set_ylim(0, self.opt.points-1)
                    x, y = np.meshgrid(np.arange(stop-start), np.arange(0, self.opt.points))
                    sptrgrm = self.axes[port, i].pcolormesh(x, y, np.zeros(x.shape), 
                                                          shading='nearest', 
                                                          vmin=self.ymin, vmax=self.ymax, 
                                                          cmap='seismic')
                    port_sptrgrm.append(sptrgrm)
                    port_Sxx.append(np.ones((stop-start, self.opt.points), dtype=np.float)*np.nan)
                    
                    _y = y[:, 0][::len(y)//6]
                    self.axes[port, i].set_yticks(_y)
                    self.axes[port, i].set_yticklabels(np.around(np.linspace(self.opt.start, self.opt.stop, len(_y))/1e9, 1))
                    
                    if i == 0 and port == 0:  # Only add ylabel to first subplot
                        self.axes[port, i].set_ylabel("Frequency (GHz)", fontsize=15, c='white')
                
                self.sptrgrm.append(port_sptrgrm)
                self.Sxx.append(port_Sxx)

        # Adjust layout to prevent overlap
        self.fig.tight_layout()
        plt.show()

    def onclick(self, event):
        target_freq = event.xdata
        if target_freq is None:
            self.highlight_freq = None
            return
        freqs = np.array(self.nv[0].device.frequencies)
        diff = np.abs(freqs - target_freq)
        idx = np.argmin(diff)
        self.highlight_freq = idx

    def update_signal(self, x, idx, port_number):
        _default_value = np.mean(self.default_value[port_number], axis=0)
        old_y1 = self.line[port_number][idx].get_ydata()
        x = [x_ - _default_value[idx] for x_ in x.tolist()]
        new_y1 = np.r_[old_y1[1:], x]
        self.line[port_number][idx].set_ydata(new_y1)

    def data_store(self):
        s11_mag_lst, s11_phase_lst = [], []
        s11_lst = []
        s21_mag_lst, s21_phase_lst = [], []
        s21_lst = []
        for dev_idx, nv in enumerate(self.nv):
            # s = nv.get_data(port=0)
            s11, s21 = nv.get_data_()
            s11_mag = self.logmag(s11)
            s11_phase = self.phase(s11)
            s21_mag = self.logmag(s21)
            s21_phase = self.phase(s21)
            s11_mag_lst.append(s11_mag)
            s11_phase_lst.append(s11_phase)
            s11_lst.append(s11)
            s21_mag_lst.append(s21_mag)
            s21_phase_lst.append(s21_phase)
            s21_lst.append(s21)
        self.ts = self.get_time()
        self.s11_mag, self.s11_phase = s11_mag_lst, s11_phase_lst
        self.s11 = s11_lst
        self.s21_mag, self.s21_phase = s21_mag_lst, s21_phase_lst
        self.s21 = s21_lst

    def plot_it(self):
        if self.isRawImp:
            Z11 = 50 * (1+np.array(self.s11)) / (1-np.array(self.s11))
            Z21 = 50 * 2 * (1-np.array(self.s21)) / np.array(self.s21)
            val1, val2 = np.real(Z11), np.imag(Z11)
            val3, val4 = np.real(Z21), np.imag(Z21)
        else:
            val1, val2 = self.s11_mag, self.s11_phase
            val3, val4 = self.s21_mag, self.s21_phase
        
        # Only update the current port's subplot
        for dev_idx, (s11, s11_phase, s21, s21_phase) in enumerate(zip(val1, val2, val3, val4)):
            _default_value = np.mean(self.default_value[self.port_number], axis=0)
            
            if self.opt.timedomain:
                if self.isTemporalCalibration:
                    x = np.array([s11, s11_phase, s21, s21_phase]).flatten()
                    self.default_value[self.port_number].append(x)
                    if len(self.default_value[self.port_number]) > self.fps * self.win_t:
                        self.default_value[self.port_number].pop(0)
                    
                chunk_size = int(self.num_lines/len(self.nv)/4)
                for i in range(chunk_size):
                    # Update the current port's lines
                    self.update_signal(np.array([s11[i]]), i, self.port_number)
                    self.update_signal(np.array([s21[i]]), chunk_size*2+i, self.port_number)
                    
                if self.isSpectrogram:
                    self.Sxx[self.port_number][0] = np.r_[self.Sxx[self.port_number][0][1:], 
                                                        np.array([s11 - _default_value[:chunk_size]])]
                    self.sptrgrm[self.port_number][0].set_array(self.Sxx[self.port_number][0].T)
                    
                    self.Sxx[self.port_number][1] = np.r_[self.Sxx[self.port_number][1][1:], 
                                                        np.array([s21 - _default_value[chunk_size*2:chunk_size*3]])]
                    self.sptrgrm[self.port_number][1].set_array(self.Sxx[self.port_number][1].T)
            else:
                if self.opt.smith and self.port_number == 0:  # Only update Smith chart for first port
                    self.smith(self.s11[dev_idx], dev_idx)
                
                if self.isTemporalCalibration:
                    x = np.array([s11, s11_phase, s21, s21_phase])
                    self.default_value[self.port_number].append(x)
                    if len(self.default_value[self.port_number]) > self.fps * self.win_t:
                        self.default_value[self.port_number].pop(0)

                # Update the current port's lines
                self.line[self.port_number][dev_idx*len(self.nv)+0].set_ydata(s11 - _default_value[0])
                self.line[self.port_number][dev_idx*len(self.nv)+1].set_ydata(s11_phase - _default_value[1])
                self.line[self.port_number][dev_idx*len(self.nv)+2].set_ydata(s21 - _default_value[2])
                self.line[self.port_number][dev_idx*len(self.nv)+3].set_ydata(s21_phase - _default_value[3])

    def interaction_mode(self):
        if self.isRecord:
            # try:
            #     saving_freq = 1 # because of switch, it should be odd number
            #     self.saving_cnt = (self.saving_cnt + 1) % saving_freq
            # except:
            #     self.saving_cnt = 0

            # _default_value = np.mean(self.default_value,axis=0).reshape(4,-1)
            # _s11_mag = np.array(self.s11_mag) - _default_value[0]
            # _s21_mag = np.array(self.s21_mag) - _default_value[2]
            # feat = np.concatenate([_s11_mag,_s21_mag],axis=1)
            # if self.saving_cnt == 0:
            if len(self.s_data) % self.max_port == self.port_number:
                feat = np.concatenate([self.s11,self.s21],axis=1)
                self.s_data.append(feat)

                how_many_seconds = -1
                if how_many_seconds != -1:
                    if self.pbar is None:
                        self.pbar = tqdm(total=int(self.fps*how_many_seconds)-1, desc="Recording progress", unit="frame")
                    self.pbar.update(1)

                if how_many_seconds != -1 and len(self.s_data) > int(self.fps*how_many_seconds): # save 20 seconds and stop
                    self.isRecord = False
                    print("Recording stop")
                    self.save_data()
        elif self.mode == "c":
            _default_value = np.mean(self.default_value[self.port_number],axis=0).reshape(4,-1)
            _s11_mag = np.array(self.s11_mag) - _default_value[0]
            _s21_mag = np.array(self.s21_mag) - _default_value[2]
            feat = np.concatenate([_s11_mag,_s21_mag],axis=1)
            self.predictions.append(self.clf.predict(feat)[0])
            while len(self.predictions) > 25:
                self.predictions.pop(0)
            predictions = np.argmax(np.bincount(self.predictions))
            print(f"Predicted: {self.labels[predictions-1]}")
            self.axes[self.port_number, 0].set_title(f"{self.labels[predictions-1]}",fontsize=30,c='white')
        else:
            self.pbar = None


    def get_time(self):   
        current_time = time.time()
        return current_time

    def print_fps(self):
        self.fps = np.around(1/(time.time()-self.starttime),1)
        print("System frame rate:",self.fps,"FPS",end='\r')
        self.starttime = time.time()

    def animate(self, frame):
        self.data_store()
        self.plot_it()
        self.interaction_mode()
        self.switch_rf()
        self.print_fps()

    def get_data(self, port):
        s = self.nv.scan()
        s = s[port]
        return s

    def phase(self, x):
        a = np.angle(x)
        a = np.rad2deg(a)
        try:
            a = a / 360 * abs(self.ymax-self.ymin)
            a = a + (self.ymin+self.ymax)/2
        except:
            pass
        return a

    def logmag(self,x):
        return 20*np.log10(np.abs(x))
        # return x

    def linmag(self, x):
        return np.abs(x)

    def groupdelay(self, x):
        gd = np.convolve(np.unwrap(np.angle(x)), [1,-1], mode='same')
        return gd

    def vswr(self, x):
        vswr = (1+np.abs(x))/(1-np.abs(x))
        return vswr

    def polar(self, x):
        return np.angle(x), np.abs(x)

    def tdr(self, x):
        window = np.blackman(len(x))
        NFFT = 256
        td = np.abs(np.fft.ifft(window * x, NFFT))
        t_axis = np.linspace(0, time, NFFT)
        return t_axis, td

    def skrf_network(self, x, freq=None):
        n = skrf.Network()
        if freq==None:
            n.frequency = skrf.Frequency.from_f(self.nv[0].device.frequencies / 1e6, unit='mhz')
        else:
            n.frequency = freq
        n.s = x
        return n

    def smith(self, x, dev_idx):
        self.line_smith[dev_idx*2].set_xdata(np.real(x))
        self.line_smith[dev_idx*2].set_ydata(np.imag(x))
        if self.highlight_freq is not None:
            self.line_smith[dev_idx*2+1].set_xdata([np.real(x)[self.highlight_freq]])
            self.line_smith[dev_idx*2+1].set_ydata([np.imag(x)[self.highlight_freq]])
        else:
            self.line_smith[dev_idx*2+1].set_ydata([np.nan])

def main(opt):
    nano_vna = get_vna_devices(opt)
    VNAStream(opt, devices=nano_vna)


if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser(usage="%prog: [options]")
    parser.add_option("-V", "--vna", dest="numvna",
                      type="int",default=1,
                      help="number of VNA", metavar="NUMVNA")
    parser.add_option("-t", "--timedomain", dest="timedomain",
                      action="store_true",default=False,
                      help="time domain visualization", metavar="TIMEDOAIM")
    parser.add_option("-r", "--raw", dest="rawwave",
                      type="int", default=None,
                      help="plot raw waveform", metavar="RAWWAVE")
    parser.add_option("-p", "--plot", dest="plot",
                      action="store_true", default=False,
                      help="plot rectanglar", metavar="PLOT")
    parser.add_option("-s", "--smith", dest="smith",
                      action="store_true", default=False,
                      help="plot smith chart", metavar="SMITH")
    parser.add_option("-L", "--polar", dest="polar",
                      action="store_true", default=False,
                      help="plot polar chart", metavar="POLAR")
    parser.add_option("-D", "--delay", dest="delay",
                      action="store_true", default=False,
                      help="plot delay", metavar="DELAY")
    parser.add_option("-G", "--groupdelay", dest="groupdelay",
                      action="store_true", default=False,
                      help="plot groupdelay", metavar="GROUPDELAY")
    parser.add_option("-W", "--vswr", dest="vswr",
                      action="store_true", default=False,
                      help="plot VSWR", metavar="VSWR")
    parser.add_option("-H", "--phase", dest="phase",
                      action="store_true", default=False,
                      help="plot phase", metavar="PHASE")
    parser.add_option("-U", "--unwrapphase", dest="unwrapphase",
                      action="store_true", default=False,
                      help="plot unwrapped phase", metavar="UNWRAPPHASE")
    parser.add_option("-T", "--timedomain2", dest="tdr",
                      action="store_true", default=False,
                      help="plot TDR", metavar="TDR")
    parser.add_option("-c", "--scan", dest="scan",
                      action="store_true", default=False,
                      help="scan by script", metavar="SCAN")
    parser.add_option("-S", "--start", dest="start",
                      type="float", default=1e6,
                      help="start frequency", metavar="START")
    parser.add_option("-E", "--stop", dest="stop",
                      type="float", default=900e6,
                      help="stop frequency", metavar="STOP")
    parser.add_option("-N", "--points", dest="points",
                      type="int", default=101,
                      help="scan points", metavar="POINTS")
    parser.add_option("-P", "--port", type="int", dest="port",
                      help="port", metavar="PORT")
    parser.add_option("-d", "--dev", dest="device",
                      help="device node", metavar="DEV")
    parser.add_option("-v", "--verbose",
                      action="store_true", dest="verbose", default=False,
                      help="verbose output")
    parser.add_option("-C", "--capture", dest="capture",
                      help="capture current display to FILE", metavar="FILE")
    parser.add_option("-e", dest="command", action="append",
                      help="send raw command", metavar="COMMAND")
    parser.add_option("-o", dest="save",
                      help="write touch stone file", metavar="SAVE")
    (opt, args) = parser.parse_args()

    main(opt)