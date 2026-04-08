# vasp-grace

`vasp-grace` is a lightweight Python wrapper that allows you to use the **GRACE Machine Learning Potentials** as a drop-in replacement for VASP. By mimicking the VASP executable, it seamlessly integrates GRACE with higher-level materials science workflows (e.g., Phonopy, USPEX, CALYPSO) that normally parse VASP inputs and outputs.

## Features
- **Direct VASP Compatibility:** Reads standard `POSCAR` and `INCAR` files.
- **Geometry Optimization:** Supports VASP's `IBRION` and `ISIF` tags for structural optimization natively via ASE.
- **Output Generation:** Writes standard `OUTCAR`, `OSZICAR`, and `CONTCAR` files mimicking VASP output structures.
- **MD SIMULATION USING ASE
- **Backend:** Native Python integration using the GRACE `tensorpotential` ASE calculator (GPU accelerated via TensorFlow).

## Installation

1. Create a Python environment and install the required dependencies (ensure TensorFlow is installed with GPU support for best performance):
   ```bash
   pip install ase numpy tensorflow
   pip install tensorpotential
