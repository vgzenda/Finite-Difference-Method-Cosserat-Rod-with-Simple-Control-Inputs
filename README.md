# Finite-Difference-Method-Cosserat-Rod-with-Simple-Control-Inputs
A simple python implementation of a finite difference simulation of Cosserat rod dynamics with simple tension force inputs.



## Clamped Cosserat Rod Simulator with Simple Tension Inputs

This is a simple implementation of a finite difference method simulation of the dynamics of a Cosserat rod based on notes found in the additionalMaterials folder. 

The uncontrolled dynamics may be simulated in the test_rod_dynamics.py file

The dynamics with simplified tenson forces as control inputs may be simulated in the test_control_inputs.py

The physical parameters of the rod may be modified in the Cosserat_Rod class.


## TODO
 - Proper control inputs using an active constitutive law
 - More robust solver for the shooting method


## Requirements:
    numpy, matplotlib, scipy
