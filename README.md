# Deep Value Networks Learn to Evaluate and Iteratively Refine Structured Outputs
By Michael Gygli, Mohammad Norouzi, Anelia Angelova ([paper](https://arxiv.org/pdf/1703.04363.pdf))

This is a tensorflow/python implementation of the paper.

This impplementation uses the Weizmann horse data set.
On the GPU it uses ~1GB VRAM.


Data is located at `dvn/data/weizmann_horse_db/rgb|gray|figure_ground`

Training:
```
cd dvn/src
python run.py --train --loglevel=debug
```
if `loglevel=debug` debugging information is written to
`src/log`

Testing:
```
cd svn/src
python run.py --loglevel=debug
```