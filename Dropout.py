import numpy as np

red_barrel = np.loadtxt('red_barrel.tar.gz')
red_other = np.loadtxt('red_other.tar.gz')
sky = np.loadtxt('sky.tar.gz')
ground = np.loadtxt('ground.tar.gz')

red_barrel_sample = red_barrel[np.random.choice(red_barrel.shape[0], 200000, False), :]
red_other_sample = red_other[np.random.choice(red_other.shape[0], 400000, False), :]
sky_sample = sky[np.random.choice(sky.shape[0], 400000, False), :]
ground_sample = ground[np.random.choice(ground.shape[0], 400000, False), :]

np.savetxt('red_barrel_sample.tar.gz', red_barrel_sample)
np.savetxt('red_other_sample.tar.gz', red_other_sample)
np.savetxt('sky_sample.tar.gz', sky_sample)
np.savetxt('ground_sample.tar.gz', ground_sample)
