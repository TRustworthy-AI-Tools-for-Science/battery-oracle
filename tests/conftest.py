import numpy as np

#: One 6-D protocol [C_rate_1, C_rate_2 mA, dur_1, dur_2 h, D_rate mA, dur_d h]
#: (a moderate two-stage charge and a 1 h discharge).
PROTOCOL = np.array([68.0, 34.0, 1.0, 0.5, 100.0, 1.0])
