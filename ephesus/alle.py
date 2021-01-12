import allesfitter
import os
pathalle = os.getcwd() + '/'
allesfitter.show_initial_guess(pathalle)
allesfitter.mcmc_fit(pathalle)
allesfitter.mcmc_output(pathalle)
