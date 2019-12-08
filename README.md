# py-population_simul
Python based simulator for population transition (markov chains)

![fig_3](https://user-images.githubusercontent.com/19597283/53855186-90e5bf00-3f9a-11e9-8753-6098814321f9.jpg)

### About

We use object oriented programming and markov chains to simmulate a population behavior. Population at each staged get stored and can be used to analyze either income or costos resulting from the stage-population mix. The program outputs grid plots and pyramid graphs to diplay the evolution of the population.

This program contains:

* Classes for State and Simul, simulate markov transitions and store them in state at [simFunc](simFunc.py).
* Plotting functions for pyramid and heatmap at [plotfun](plotfun.py).
* Read databases, perform calculations at [calc](calc.py).
