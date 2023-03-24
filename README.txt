  ***************************************************************************
  * All the software  contained in this library  is protected by copyright. *
  * Permission  to use, copy, modify, and  distribute this software for any *
  * purpose without fee is hereby granted, provided that this entire notice *
  * is included  in all copies  of any software which is or includes a copy *
  * or modification  of this software  and in all copies  of the supporting *
  * documentation for such software.                                        *
  ***************************************************************************
  * THIS SOFTWARE IS BEING PROVIDED "AS IS", WITHOUT ANY EXPRESS OR IMPLIED *
  * WARRANTY. IN NO EVENT, NEITHER  THE AUTHORS, NOR THE PUBLISHER, NOR ANY *
  * MEMBER  OF THE EDITORIAL BOARD OF  THE JOURNAL  "NUMERICAL ALGORITHMS", *
  * NOR ITS EDITOR-IN-CHIEF, BE  LIABLE FOR ANY ERROR  IN THE SOFTWARE, ANY *
  * MISUSE  OF IT  OR ANY DAMAGE ARISING OUT OF ITS USE. THE ENTIRE RISK OF *
  * USING THE SOFTWARE LIES WITH THE PARTY DOING SO.                        *
  ***************************************************************************
  * ANY USE  OF THE SOFTWARE  CONSTITUTES  ACCEPTANCE  OF THE TERMS  OF THE *
  * ABOVE STATEMENT AND OF THE ACCOMPANYING FILE LICENSE.txt.               *
  ***************************************************************************

   AUTHORS:

       Marco Caliari
       University of Verona, Italy
       Email: marco.caliari@univr.it

       Fabio Cassini
       University of Trento, Italy
       Email: fabio.cassini@unitn.it

       Franco Zivcovich
       Laboratoire Jacques--Louis Lions, Sorbonne University, France
       Email: franco.zivcovich@sorbonne-universite.fr

   REFERENCE:

       A mu-mode BLAS approach for multidimensional tensor-structured
       problems
       NUMERICAL ALGORITHMS 92, 2483-2508 (2023)
       DOI: https://doi.org/10.1007/s11075-022-01399-4

   SOFTWARE REVISION DATE:

       V1.0, August 2022

   SOFTWARE LANGUAGE:

       MATLAB 9.6 (R2019a)
       GNU Octave 7.1.0


=====================================================================
SOFTWARE
=====================================================================

The KronPACK package provides an implementation of the mu-mode related
tensor operations needed for the effective solution of n-dimensional tensor
structured problems, such as action of the matrix exponential,
preconditioning of linear systems for stiff differential equations,
multivariate approximation, and pseudospectral decomposition.

The software has been developed and tested using MATLAB version 9.6 and
GNU Octave version 7.1.0.
No other MathWorks products or external toolboxes are required.

To reproduce some numerical experiments of the accompanying manuscript,
the toolboxes Tensorlab and Tensor Toolbox for MATLAB are needed.

=====================================================================
HOW TO INSTALL AND CHECK THE INSTALLATION
=====================================================================

Please follow these steps:

- Extract the archive in a directory of your choice. This creates a
  directory called KronPACK with all files of the toolbox.

- Start MATLAB or GNU Octave and add the directory KronPACK/src to the
  search path.

- The software installation can be checked by running the example scripts
  in the directory KronPACK/examples

- The source functions KronPACK/src contain several GNU Octave built-in
  self-tests that can be executed by running the command

  test function_name

  For each test, the first argument in the GNU Octave assert command is
  compared to the second argument, up to the tolerance possibly given
  in the third argument. Therefore, it is possible to perform the same
  tests in MATLAB with a suitable modification.

=====================================================================
SOFTWARE UPDATES AND BUG FIXES
=====================================================================

In addition to the refereed version of the software published along
with the journal paper, the KronPACK software will be maintained
in the GitHub code repository

  https://github.com/caliarim/KronPACK

Please check this location for software updates and bug fixes.


=====================================================================
PACKAGE
=====================================================================

The KronPACK package is organized into a main directory and subdirectories:

KronPACK          - contains this README.txt file, a LICENSE file, Contents.m
                    which provides an overview of all files in the package
                    (can be listed from within MATLAB or GNU Octave using
                    "help KronPACK")
KronPACK/src      - contains all the source functions
KronPACK/examples - contains all numerical experiments

See the file Contents.m for a list of the files contained in the package.


=====================================================================
EXAMPLE SCRIPTS
=====================================================================

We include in KronPACK/examples a number of example scripts that can be
executed to reproduce the numerical experiments of the accompanying paper.
