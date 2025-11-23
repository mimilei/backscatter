import skrf
from skrf.vi import vna
from serial.tools import list_ports
import numpy as np

VIDPIDs = set([(0x0483, 0x5740), (0x04b4,0x0008)]);

# Get nanovna device automatically
def getport() -> str:
    device_list = list_ports.comports()
    print(f"getport: {device_list}")
    print("want /dev/cu.usbmodem4001")
    for device in device_list:
        if (device.vid, device.pid) in VIDPIDs:
            print(f"found device: {device.device}")
            return device.device
    raise OSError("device not found")

# connect to NanoVNA on /dev/cu.usbmodemDEMO1 (MacOS)
nanovna = skrf.vi.vna.NanoVNAv2('ASRL'+getport())

# LiteVNA 64 can't sweep 1 MHz → 4 GHz in one sweep, so we break it into 4
f_start = [1e6,1e9+1e6,2e9+1e6,3e9+1e6] 
f_stop = [1e9,2e9,3e9,4e9]
f_step = 1e6

for name in ['open','short','match','thru']:
    s = []
    input('Connect {} and press ENTER:'.format(name))
    for i in range(4):
        num = int(1 + (f_stop[i] - f_start[i]) / f_step)
        nanovna.set_frequency_sweep(f_start[i], f_stop[i], num)
        nw_raw = nanovna.get_snp_network(ports=(0, 1))
        s.append(nw_raw.s)
    s = np.concatenate(s)
    nw_raw = skrf.Network(frequency=np.arange(1e6,4e9+1,1e6)/1e9,s=s)
    nw_raw.write_touchstone('./data/cal_{}.s2p'.format(name))
    print(nw_raw.frequency.start,nw_raw.frequency.stop,nw_raw.frequency.step)