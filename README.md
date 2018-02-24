# Reservoir Computing

Facilitating RC research

## Features

* [Nonlinear transient computing (NTC)](https://github.com/masterdezign/rc/tree/master/examples/NTC)
* Echo State Networks (ESNs) (upcoming)


## Getting Started

Use [Stack](http://haskellstack.org).

     $ git clone https://github.com/masterdezign/rc.git && cd rc
     $ stack build --install-ghc

### [Example 1](https://github.com/masterdezign/rc/tree/master/examples/NTC). NTC, time series forecasting


     $ stack exec ntc

     Error: 3.1103181863915367e-3

Visualize the prediction results:

     $ python3 examples/NTC/plot.py

![Figure](https://raw.githubusercontent.com/masterdezign/rc/master/examples/NTC/mg-prediction.png)

Great!
