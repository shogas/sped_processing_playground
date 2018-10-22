# SPED data processing
This is a collection of some python code for processing SPED data, mainly using [pyxem].

## Pyxem template matching
The [pyxem template matching notebook](PyXem template matching.ipynb) is the most complete code. It shows an example of running template matching on a GaAs nanowire SPED dataset and should hopefully be easy to follow. Feedback on the implementation is welcome.

## Preprocessing
The [preprocessing notebook](Preprocessing.ipynb) contains some code for testing different parameters for background removal.

## Interactive pattern parameters
The [pattern generation visualizer](pattern_gen_visualizer.py) script plots a kinematic simulation of a crystal structure using [pyxem]. Usefull for finding approximately correct parameters before template matching. When first loading, the figure on the right shows a set of points rotated using a rotation list covering the inverse pole figure of the given structure. Usefull for creating and debugging the rotation list implementation. After changing the parameters, the figure on the right shows the rotation for the current angle, as well as a local rotation list generated around that point. For visualization, the spread of this local rotation list is large, but for actual usage, the spread and resolution would be much lower. Not very user friendly, and matplotlib is not really meant for this type of plotting, so it is slow. I am considering implementing a more efficient and user friendly version of this.

[pyxem]: https://github.com/pyxem/pyxem
