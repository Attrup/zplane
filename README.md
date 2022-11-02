# Z-Plane
A handful of freqeuntly used plots when working with discrete systems or digital signal processing in Pyton. Z-Plane is built upon the scipy.signal module, using the `TransferFunction` class to pass all required system information to plot functions in a single object. This allows all plots to be executed in a single line of code, while Z-Plane handles the calculations and setup required to get the perfect plot.

The following functions are available:
- `freq`: Normalized frequency response
- `pz`: Pole-Zero plot 
- `bode`: Bode plot (Gain and Phase), non logarithmic frequency axis possible
- `impulse`: Impulse response
- `norm`: Normalize transfer function
- `fir2tf`: Get FIR transfer function from impulse response

## Installation
```bash
pip install zplane
```

## Use
- Import `zplane`
- Call any of the available functions, and pass a valid scipy.signal TransferFunction

Have a look at the [examples](https://github.com/Attrup/zplane/blob/main/examples/examples.py) for a quick demonstration on how to use the functions. Please look up the `TransferFunction` [documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.TransferFunction.html), if you are unsure how to create a valid instance of a `TransferFunction`.
