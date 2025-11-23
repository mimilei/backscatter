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


# connect to NanoVNA on /dev/ttyACM0 (Linux)
# nanovna = skrf.vi.vna.NanoVNAv2('ASRL/dev/ttyACM0::INSTR')

# connect to NanoVNA on /dev/cu.usbmodemDEMO1 (MacOS)
nanovna = skrf.vi.vna.NanoVNAv2('ASRL'+getport())

# for Windows users: ASRL1 for COM1
# nanovna = skrf.vi.vna.NanoVNAv2('ASRL1::INSTR')

# configure frequency sweep (for example 1 MHz to 4.4 GHz in 1 MHz steps)
# f_start = 1e6
# f_stop = 4e9
# f_step = 1e6

# num = int(1 + (f_stop - f_start) / f_step)
# nanovna.set_frequency_sweep(f_start, f_stop, num)

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

# measure all 12 combinations of the 4-port
# n_ports = 4
# for i_src in range(n_ports):
#     for i_sink in range(n_ports):
#         if i_sink != i_src:
#             input('Connect vna_p1 -> dut_p{}, vna_p2 -> dut_p{} and press ENTER:'.format(i_src + 1, i_sink + 1))
#             nw_raw = nanovna.get_snp_network(ports=(0, 1))
#             nw_raw.write_touchstone('./data/dut_raw_{}{}'.format(i_sink + 1, i_src + 1))