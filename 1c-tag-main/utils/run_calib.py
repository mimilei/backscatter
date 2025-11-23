import skrf
from skrf.calibration import TwoPortOnePath
import numpy as np
import matplotlib.pyplot as plt

# load networks of the raw calibration standard measurements
name = "data"
short_raw = skrf.Network(f'./{name}/cal_short.s2p')
open_raw = skrf.Network(f'./{name}/cal_open.s2p')
match_raw = skrf.Network(f'./{name}/cal_match.s2p')
thru_raw = skrf.Network(f'./{name}/cal_thru.s2p')


# create an ideal 50-Ohm line for the short, open, match and through reference responses ("ideals")
line = skrf.DefinedGammaZ0(frequency=short_raw.frequency, Z0=50)

# create and run the calibration
cal = TwoPortOnePath(ideals=[line.short(nports=2), line.open(nports=2), line.match(nports=2), line.thru()],
                     measured=[short_raw, open_raw, match_raw, thru_raw],
                     n_thrus=1, source_port=1)

cal.run()

short_raw_cal = cal.apply_cal((short_raw))
open_raw_cal = cal.apply_cal((open_raw))
match_raw_cal = cal.apply_cal((match_raw))
thru_raw_cal = cal.apply_cal((thru_raw))


### VIS DATA IN SMITH CHART

def skrf_network(x, freq=None):
    n = skrf.Network()
    if freq==None:
        n.frequency = skrf.Frequency.from_f(freq / 1e6, unit='mhz')
    else:
        n.frequency = freq
    n.s = x
    return n


raw = [short_raw, open_raw, match_raw, thru_raw]
calibrated = [short_raw_cal, open_raw_cal, match_raw_cal, thru_raw_cal]
ideal = [line.short(nports=2),line.open(nports=2),line.match(nports=2),line.thru()]

for i in range(4):
    raw_, calibrated_, ideal_ = raw[i].s, calibrated[i].s, ideal[i].s
    raw_, calibrated_, ideal_ = raw_[:,0:1,0:1], calibrated_[:,0:1,0:1], ideal_[:,0:1,0:1]
    n = skrf_network(ideal_,freq=short_raw.frequency)
    n.plot_s_smith(color='blue',marker='x',draw_labels=True)
    n = skrf_network(calibrated_,freq=short_raw.frequency)
    n.plot_s_smith(color='green',marker='o',draw_labels=True)
    n = skrf_network(raw_,freq=short_raw.frequency)
    n.plot_s_smith(color='red',draw_labels=True)
    plt.show()



#######
# import skrf
# from skrf.calibration import TwoPortOnePath
# import numpy as np
# import matplotlib.pyplot as plt
# import glob
# import numpy as np

# short_flist = glob.glob("./data/cal_short_*.s2p")
# open_flist = glob.glob("./data/cal_open_*.s2p")
# match_flist = glob.glob("./data/cal_match_*.s2p")
# thru_flist = glob.glob("./data/cal_thru_*.s2p")

# short_flist.sort()
# open_flist.sort()
# match_flist.sort()
# thru_flist.sort()


# frequency = np.arange(1e6,4e9+1,1e6)
# short_raw = []
# open_raw = []
# match_raw = []
# thru_raw = []
# # load networks of the raw calibration standard measurements
# for i in range(4):
#     short_dat = skrf.Network(short_flist[i])
#     open_dat = skrf.Network(open_flist[i])
#     match_dat = skrf.Network(match_flist[i])
#     thru_dat = skrf.Network(thru_flist[i])
#     short_raw.append(short_dat.s)
#     open_raw.append(open_dat.s)
#     match_raw.append(match_dat.s)
#     thru_raw.append(thru_dat.s)

# short_raw = np.concatenate(short_raw)
# open_raw = np.concatenate(open_raw)
# match_raw = np.concatenate(match_raw)
# thru_raw = np.concatenate(thru_raw)

# short_raw = skrf.Network(frequency=frequency/1e9,s=short_raw)
# open_raw = skrf.Network(frequency=frequency/1e9,s=open_raw)
# match_raw = skrf.Network(frequency=frequency/1e9,s=match_raw)
# thru_raw = skrf.Network(frequency=frequency/1e9,s=thru_raw)

# short_raw.write_touchstone('./data/cal_short.s2p')
# open_raw.write_touchstone('./data/cal_open.s2p')
# match_raw.write_touchstone('./data/cal_match.s2p')
# thru_raw.write_touchstone('./data/cal_thru.s2p')

# # create an ideal 50-Ohm line for the short, open, match and through reference responses ("ideals")
# line = skrf.DefinedGammaZ0(frequency=short_raw.frequency, Z0=50)

# # create and run the calibration
# cal = TwoPortOnePath(ideals=[line.short(nports=2), line.open(nports=2), line.match(nports=2), line.thru()],
#                      measured=[short_raw, open_raw, match_raw, thru_raw],
#                      n_thrus=1, source_port=1)

# cal.run()

# short_raw_cal = cal.apply_cal((short_raw))
# open_raw_cal = cal.apply_cal((open_raw))
# match_raw_cal = cal.apply_cal((match_raw))
# thru_raw_cal = cal.apply_cal((thru_raw))


# ### VIS DATA IN SMITH CHART

# def skrf_network(x, freq=None):
#     n = skrf.Network()
#     if freq==None:
#         n.frequency = skrf.Frequency.from_f(freq / 1e6, unit='mhz')
#     else:
#         n.frequency = freq
#     n.s = x
#     return n


# raw = [short_raw, open_raw, match_raw, thru_raw]
# calibrated = [short_raw_cal, open_raw_cal, match_raw_cal, thru_raw_cal]
# ideal = [line.short(nports=2),line.open(nports=2),line.match(nports=2),line.thru()]

# for i in range(3):
#     raw_, calibrated_, ideal_ = raw[i].s, calibrated[i].s, ideal[i].s
#     raw_, calibrated_, ideal_ = raw_[:,0:1,0:1], calibrated_[:,0:1,0:1], ideal_[:,0:1,0:1]
#     n = skrf_network(ideal_,freq=short_raw.frequency)
#     n.plot_s_smith(color='blue',marker='x',draw_labels=True)
#     n = skrf_network(calibrated_,freq=short_raw.frequency)
#     n.plot_s_smith(color='green',marker='o',draw_labels=True)
#     n = skrf_network(raw_,freq=short_raw.frequency)
#     n.plot_s_smith(color='red',draw_labels=True)
#     plt.show()