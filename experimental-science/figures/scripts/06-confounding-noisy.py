"""Lab 6.3 of *Deep Learning as an Experimental Science*: attribution along a path, noisy labels.

The 2^4 factorial of Lab 6.2 (architecture, optimizer, label smoothing, duration; see
06-confounding-paths.py, whose code this script reuses) with 20% of the training labels of
every split replaced by uniformly random labels (labkit's label_noise, seed = block). Six
blocks (splits 40-45). With noisy labels the recipe's components interact strongly, and the
attributions along different orders disagree.

Run from the volume root:  python3 figures/scripts/06-confounding-noisy.py
Writes figures/data/06-confounding-noisy-{cells,simple,shapley,recipes}.dat and prints the
numbers quoted in Chapter 6. Three to thirteen minutes on two threads, depending on the load
of the machine (three and a half to eight CPU-minutes).
"""
import os
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("confounding_paths",
                                              os.path.join(HERE, "06-confounding-paths.py"))
paths = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paths)

if __name__ == "__main__":
    paths.main(label_noise=0.2, blocks=range(40, 46), prefix="06-confounding-noisy")
