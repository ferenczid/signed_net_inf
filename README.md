# signed_net_inf

Bayesian inference of signed social networks from contact patterns.

Code accompanying:

> D. Ferenczi, J.-G. Young, L. Peel. **Inferring signed social networks from contact patterns.** *Journal of Physics: Complexity*, 2026. https://doi.org/10.1088/2632-072X/ae76ae

## Installation

Tested with Python 3.10 on Ubuntu.

```bash
git clone https://github.com/ferenczid/signed_net_inf.git
cd signed_net_inf
conda env create -f environment.yml
conda activate signed_net_inf
```

## Quick start

The tutorial notebook runs the inference on a small example:

```bash
cd tutorial_notebook
jupyter notebook signed_inference_tutorial.ipynb
```

Or execute it without opening it:

```bash
jupyter nbconvert --to notebook --execute signed_inference_tutorial.ipynb --output /tmp/tutorial_test.ipynb
```



## Repository structure

| Folder | Contents |
|---|---|
| `tutorial_notebook/` | Step-by-step example of the inference method on a small network |
| `baselines/` | Baseline comparison experiments |
| `real_life_experiments/` | Empirical data experiments |

Each folder contains its own copy of `signed_inference.py` (the copies in `tutorial_notebook/` and `baselines/` are identical; `real_life_experiments/` has a modified version). Run scripts from inside their own folder, because they import this module locally.


## Data

The experiments in `real_life_experiments/` use the SocioPatterns high-school contact data collected in December 2013. The `.npy` files in that folder are derived from it. The original data is distributed under a Creative Commons Attribution-NonCommercial-ShareAlike license, so these derived files are shared under the same terms. If you use them, please cite:

> R. Mastrandrea, et al., PLoS ONE 10(9), e0136497 (2015). 
Note: the `diff_days/` folder and the day-level files are large (the repository is several hundred MBs on disk).

## Citation

If you use this code, please cite the paper above.

```bibtex
@article{ferenczi2026signed,
  title   = {Inferring signed social networks from contact patterns},
  author  = {Ferenczi, D{\'a}vid and Young, Jean-Gabriel and Peel, Leto},
  journal = {Journal of Physics: Complexity},
  doi     = {10.1088/2632-072X/ae76ae}
  year    = {2026}
}
```

## License

The code is released under the MIT License (see `LICENSE`). The data files derived from the SocioPatterns dataset are under CC BY-NC-SA, as described above.
