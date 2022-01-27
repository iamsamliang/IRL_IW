# Construing the Reward Function: Maximum Likelihood Inverse Reinforcement Learning with Limited Cognitive Resources

### To set up

Set up a virtual environment, e.g.:

```
$ virtualenv -p python3 myenv
```

Install requirements into env:

```
$ source myenv/bin/activate
(myenv) $ pip install -r requirements
```

### Code

This code implements the Maximum Likelihood IRL algorithm from this paper https://icml.cc/2011/papers/478_icmlpaper.pdf and a classifier trained to output a probability distribution over what actions to take in a state for GridWorld MDPs. Moreover, I explore and investigate the performance of construals in MLIRL. Construals are introduced in this paper https://arxiv.org/abs/2105.06948. I introduced construals into the MLIRL algorithm by construing the reward function, naming this new algorithm Construed Maximum Likelihood IRL (CMLIRL). All algorithms can be found in algorithms.py. For more information on CMLIRL, read my paper that discusses it in detail and experimental results: paper_link.
