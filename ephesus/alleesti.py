import allesfitter
import os
pathalle = os.getcwd() + '/'
#allesfitter.show_initial_guess(pathalle)
allesfitter.estimate_noise(pathalle)
#allesfitter.estimate_noise_out_of_transit(pathalle)
#allesfitter.mcmc_output(pathalle)

