## TO DO
1. Silly but before we do anything else, let the robot run as is for a long time, say 1000 rounds, max 200 steps per episode and see that it actually learns something: overfitting would be fine at this stage since there's no variation.
2. Add randomness to robot episodic starting position: ideally 3 diff starting places, plus rotate by some random angle 0-45 degrees. This can also be called or treated as model validation.
3. Adjust logging and model saving, maybe tensorboard/wandb for logging, could be weird given we work in docker container.