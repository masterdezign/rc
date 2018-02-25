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


## Further reading

* Appeltant, L., et al. “Information Processing Using a Single
  Dynamical Node as Complex System.” Nature Communications, vol. 2,
  2011, p. 468., doi:10.1038/ncomms1476.
* Larger, L., et al. “Photonic Information Processing beyond Turing: an Optoelectronic Implementation of Reservoir Computing.” Optics Express, vol. 20, no. 3, 2012, p. 3241., doi:10.1364/oe.20.003241.
* Rabinovič, Mihail Izrailevič, et al. Principles of Brain Dynamics: Global State Interactions. The MIT Press, 2012.
* Jaeger, H. “Harnessing Nonlinearity: Predicting Chaotic Systems and Saving Energy in Wireless Communication.” Science, vol. 304, no. 5667, Feb. 2004, pp. 78–80., doi:10.1126/science.1091277.
* Bogdan Penkovsky. Theory and Modeling of Complex Nonlinear Delay Dynamics Applied to Neuromorphic Computing. Artificial Intelligence [cs.AI]. Université Bourgogne Franche-Comté, 2017. English. 〈tel-01591441v2〉
