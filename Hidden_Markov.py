# Predict next day temp

'''
### RULES ###
1. Cold = 0 -- Hot = 1
2. First day 80% hot 
3. Cold has 30% of being followed by hot
4. Hot has 20% of being followed by cold 
5. Cold SD 0 and 5 -- Hot SD 15 and 10
'''

import tensorflow_probability as tfp 
import tensorflow as tf 

# Define Distriutions 
tfd = tfp.distributions # Shortcut for later
initial_distribution = tfd.Categorical(probs = [0.8, 0.2]) # Rule 2 
transition_distribution = tfd.Categorical(probs = [[0.7, 0.3], [0.2, 0.8]]) # Rules 3 & 4
observation_distribution = tfd.Normal(loc = [0., 15.], scale = [5., 10.]) # Rule 5
                                    # loc is mean    # Scale is standard deviation
# Define Model 
model = tfd.HiddenMarkovModel(
    initial_distribution = initial_distribution,
    transition_distribution = transition_distribution,
    observation_distribution = observation_distribution,
    num_steps = 7
)

# Run Model 
mean = model.mean()
with tf.compat.v1.Session() as sess:
    print(mean.numpy())