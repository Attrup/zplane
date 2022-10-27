import zplane as zp
import scipy.signal as sig

# Create trasnfer function for a lowpass filter
b, a = sig.butter(3, 0.1, btype='low', output='ba')
tf = sig.TransferFunction(b, a, dt=1/44100)

# Normalised frequency response with grid 
zp.freq(tf, type='solid', grid=True, name='3rd order Lowpass filter')

# Pole-Zero plot
zp.pz(tf)

# Non logartihmic Bode plot cut short of Nyquist Frequency
zp.bode(tf, log=False, stop=20000)

# Impulse response
zp.impulse(tf, name='Lowpass filter')

# Normalising system (In this case norm does nothing as sig.butter normalises the generated filter)
zp.norm(tf)
