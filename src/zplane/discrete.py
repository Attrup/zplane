import matplotlib.pyplot as plt
import scipy.signal as sig
import numpy as np


def freq(tf: sig.TransferFunction, res=200, type='stem', grid=False, full=False, norm=True, name: str = None, filename: str = None):
    """
    Plot the frequency response of a transfer function
    -----
    Wrapper for the scipy.signal.freqz function. Input `tf` must be a transfer function object from the scipy.signal library.
    Allows plotting with options directly from a transfer function.
    
    The following options are available for the plot:
    - `res`: Resolution of the calculated frequency response. Higher is more detailed. Must be greater than 1.
    - `type`: Sets the plot types, chose between the options: `stem`, `point`, `solid` & `dashed`
    - `grid`: Enable gridlines on the plot
    - `full`: Mirrors the frequency response from 0..1 on to -1..0 to achieve -1..1 coverage. 
            This option will NOT correctly reflect systems with poles/zeros that are not real or complex conjugate pairs
    - `norm`: Normalize the frequency response
    - `name`: Appends the name to the title of the plot
    - `filename`: Saves the plot as an image with the specified name instead of displaying it. A path can be specified, but it has to end with the desired file name.
    """

    # Calculate frequency response
    if res < 1: raise ValueError('The specified resolution value is below 1')

    w, h = sig.freqz(tf.num, tf.den, worN=res)

    # Normalize if opted
    h = abs(h)/max(abs(h)) if norm else abs(h)

    # Convert magnitude to dB and normalize frequency
    h_dB = 20*np.log10(h)
    w_norm = w/np.pi

    # Extend plot
    if full:
        h_dB = np.append(np.flip(h_dB)[:len(h_dB) - 1], h_dB)
        w_norm = np.append(np.flip(w_norm)[:len(w_norm) - 1] * -1, w_norm)

    # Plot
    plt.axhline(0, color='silver')

    match type:
        case 'stem':
            plt.stem(w_norm, h_dB, )
            plt.axhline(0, color='red')
        case 'point':
            plt.plot(w_norm, h_dB, linestyle='', marker='.')
        case 'solid':
            plt.plot(w_norm, h_dB)
        case 'dashed':
            plt.plot(w_norm, h_dB, linestyle='dashed')
        case _:
            raise Exception('Invalid plot type')

    if grid: plt.grid(alpha=0.5)

    plt.title(f'Frequency response of {name}' if name else 'Frequency response')
    plt.xlabel('Frequency (normalized)/$\pi$')
    plt.ylabel('Magnitude (normalized) [dB]' if norm else 'Magnitude [dB]')
    plt.xlim([-1, 1] if full else [0, 1])

    # Save or display plot
    plt.savefig(filename + '.png', dpi=600) if filename else plt.show()

    # Clear figure
    plt.clf()


def pz(tf: sig.TransferFunction, name: str = None, filename: str = None):
    """
    Plot pole/zero locations of a transfer function
    -----
    Input `tf` must be a transfer function object from the scipy.signal library.
    
    The following options are available for the plot:
    - `name`: Appends the name to the title of the plot
    - `filename`: Saves the plot as an image with the specified name instead of displaying it. A path can be specified, but it has to end with the desired file name.
    """
    # Get poles
    poles = np.round(tf.poles, 2)
    zeros = np.round(tf.zeros, 2)

    # Draw the unit circle and axis
    ax = plt.gca()
    ax.cla()
    ax.add_patch(plt.Circle((0, 0), 1, color='lightgray', fill=False))
    plt.axhline(0, color='lightgray', linewidth=1, alpha=0.8)
    plt.axvline(0, color='lightgray', linewidth=1, alpha=0.8)

    # Calculate and set plot limits
    offset = 1.3

    xmax = max(np.amax(np.real(poles)), np.amax(np.real(zeros)))
    xmin = min(np.amin(np.real(poles)), np.amin(np.real(zeros)))
    ymax = max(np.amax(np.imag(poles)), np.amax(np.imag(zeros)))
    ymin = min(np.amin(np.imag(poles)), np.amin(np.imag(zeros)))

    center_x = (xmax + xmin)/2
    center_y = (ymax + ymin)/2

    pad = max(abs(xmax - xmin)/2, abs(ymax - ymin)/2) * offset
    
    # Guard to ensure that pad has a positive value
    pad = pad if pad else offset

    ax.set_xlim(center_x - pad, center_x + pad)
    ax.set_ylim(center_y - pad, center_y + pad)

    # Plot
    pad_modifyer = np.log10(1 + pad) * 0.08

    for (type, marker, col, side) in [(poles, 'x', 'red', -2.5), (zeros, 'o', 'blue', 1)]:
        unique = np.unique(type)

        for x in unique:
            if unique.size < type.size:
                num = np.count_nonzero(type == x)

                if num > 1:
                    plt.text(np.real(x) + pad_modifyer, np.imag(x) + pad_modifyer * side, str(num), fontsize=9)

            plt.plot(np.real(x), np.imag(x), marker, color=col, fillstyle='none')

    plt.grid(alpha=0.15)
    ax.set_aspect(1.0)
    plt.title(f'Pole-Zero plot of {name}' if name else 'Pole-Zero plot')
    plt.xlabel('Real Axis')
    plt.ylabel('Imaginary Axis')

    # Save or display plot
    plt.savefig(filename + '.png', dpi=600) if filename else plt.show()

    # Clear figure
    plt.clf()


def bode(tf: sig.TransferFunction, log=True, res=50, start=None, stop=None, name: str = None, filename: str = None):
    """
    Plot phase and gain response of a transfer function
    -----
    Input `tf` must be a transfer function object from the scipy.signal library, with the time interval dt set to 1/fs.
    The plot defaults to the range freq = 0 to Nyquist freq, but custom start and stop frequencies can be set. 
    Very narrow custom frequency intervals do not work well with a logartihmic frequency axis, due to low resolution at high frequencies.

    The following options are available for the plot:
    - `log`: Sets the frequency axis to be logarithmic, creating a true bode plot
    - `res`: The number of frequencies where the transfer function is evaluated. Value is multiplied with 1000.
    - `start`: Set a new starting frequency for the plot. `start`must be an integer greater than 0 and less than the Nyquist frequency of the transfer function. Cannot be equal to `stop`.
    - `stop`: Set a new stop frequency for the plot. `stop`must be an integer greater than the starting frequency and less than or equal to the Nyquist frequency of the transfer function. Cannot be equal to `start`.
    - `name`: Appends the name to the title of the plot
    - `filename`: Saves the plot as an image with the specified name instead of displaying it. A path can be specified, but it has to end with the desired file name.
    """

    # Check that transer function is valid
    if not tf.dt: raise Exception('Specified transfer function must have a dt value')

    # Calculate sampling frequency, Nyquist frequency and resolution
    fs = round(1/tf.dt)
    nq = round(fs / 2)

    res *= 1000

    # Create Bodeplot components
    w = np.logspace(0.0, np.log10(res/2), num=res) / res if log else np.array(range(0, res + 1)) / (2 * res)
    
    w, mag, phase = sig.dbode(tf, w=w)

    # Set custom frequency interval if chosen, and check if provided start/stop values are valid
    index = lambda val: np.absolute(w - val).argmin()

    if start or stop:
        start = start if start else 0
        stop = stop if stop else nq

        # Check that start and stop values are valid
        if start < 0 or start >= nq or start >= stop:
            raise ValueError('The start frequency must be greater than 0 and less than Nyquist frequency and stop (if set)')
        elif stop < 0 or stop > nq:
            raise ValueError('The stop frequency must be greater than 0 and start (if set) and less than Nyquist frequency')

        start = index(start)
        stop = index(stop)

        w = w[start:stop]
        mag = mag[start:stop]
        phase = phase[start:stop]
        
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig.suptitle(f'Frequency response of {name}' if name else 'Frequency response')
    ax1.set_ylabel("Magnitude [dB]")
    ax1.semilogx(w, mag) if log else ax1.plot(w, mag)
    ax2.set_ylabel("Phase [degrees]")
    ax2.semilogx(w, phase) if log else ax2.plot(w, phase)
    ax2.set_xlabel("Frequency [Hz]")

    # Save or display plot
    plt.savefig(filename + '.png', dpi=600) if filename else plt.show()

    # Clear figure
    plt.clf()

def impulse(tf: sig.TransferFunction, samp=False, name=None, filename=None):
    """
    Plot impulse response of a transfer function
    -----
    Input `tf` must be a transfer function object from the scipy.signal library, with the time interval dt set to 1/fs.
    
    The following options are available for the plot:
    - `samp`: Use samples instead of time for the first axis
    - `name`: Appends the name of the plot to the title on the plot
    - `filename`: Saves the plot as an image with the specified name instead of displaying it. A path can be specified, but it has to end with the desired file name.
    """

    # Check that transer function is valid
    if not tf.dt: raise Exception('Specified transfer function must have a dt value')

    # Get impulse response
    t, y = sig.dimpulse((tf.num, tf.den, 1), n=len(sig.dimpulse(tf)[0])) if samp else sig.dimpulse(tf)

    # Plot
    plt.step(t, np.squeeze(y)) if samp else plt.plot(t, np.squeeze(y))
    
    plt.grid(alpha=0.15)
    plt.title(f'Impulse response of {name}' if name else 'Impulse response')
    plt.xlabel('n [samples]' if samp else 'Time [s]')
    plt.ylabel('Amplitude [a.u.]')

    # Save or display plot
    plt.savefig(filename + '.png', dpi=600) if filename else plt.show()

    # Clear figure
    plt.clf()

def norm(tf: sig.TransferFunction):
    """
    Normalize the gain of a transfer function
    -----
    Input `tf` must be a transfer function object from the scipy.signal library.
    Normalizes the specified transfer function, to achieve unity gain at the frequencies with maximum gain.
    """
    _, gain = sig.freqz(tf.num, tf.den)
    tf.num = tf.num/max(abs(gain))