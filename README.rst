shotshaper
======

Introduction
------------

Shotshaper is a sports projectile trajectory simulator. The code uses generic
solvers from the SciPy library to solve ODEs describing the trajectories.

The current focus of the simulator is on simulating disc golf discs. But
implementations are also provided for spherical projectiles with drag and spin.

Over time the simulator is intended to act as a library of sports projectiles
and their simulation for use in research and education.

Installation
------------

To install the package, first ensure that the following required libraries are installed:

- numpy
- scipy
- matplotlib

These can be installed using the provided ``requirements.txt``,

.. code-block:: console

        pip install -r requirements.txt

Next, the package can be installed using pythontools:

.. code-block:: console

        python setup.py install

Alternatively, just add the shotshaper directory to the PYTHONPATH.

Documentation
-------------

Examples on how to use the package are given in the examples directory. Documentation is under construction.

Contributions
-------------

Contributions are encouraged via pull requests, feature requests and bug reports on GitHub. 

License
-------

This software is released under the GPL v3.0 license. See the LICENSE file for license rights and limitations.
