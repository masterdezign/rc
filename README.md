# Reservoir Computing

Facilitating RC research

## Features

* [Nonlinear transient computing (NTC)](https://github.com/masterdezign/rc/tree/master/examples/NTC)
* Echo State Networks (ESNs)


## Getting Started

Use [Stack](http://haskellstack.org).

     $ git clone https://github.com/masterdezign/rc.git && cd rc
     $ stack build --install-ghc

### [Example 1](https://github.com/masterdezign/rc/tree/master/examples/NTC). NTC


     $ stack exec ntc

     Error: 3.1103181863915367e-3

Below is visualized prediction result:

     $ python3 examples/NTC/plot.py

![Figure](https://raw.githubusercontent.com/masterdezign/rc/master/examples/NTC/mg-prediction.png)

Great!
