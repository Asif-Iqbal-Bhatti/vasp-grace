# vasp-grace

**vasp-grace** is a lightweight Python interface that enables the use of **GRACE Machine Learning Potentials** as a drop-in replacement for the VASP executable. By emulating VASP’s input/output behavior, it integrates seamlessly with established atomistic simulation workflows that rely on VASP-compatible formats.

## Key Features

- **VASP-Compatible Interface**  
  Fully supports standard VASP input files such as `POSCAR` and `INCAR`, allowing existing workflows to run without modification.

- **Geometry Optimization**  
  Implements structural optimization via ASE, with support for common VASP tags such as `IBRION` and `ISIF`.

- **Standard Output Files**  
  Generates `OUTCAR`, `OSZICAR`, and `CONTCAR` files in VASP-like formats for downstream compatibility and analysis.

- **Molecular Dynamics (MD) Simulations**  
  Enables MD simulations through ASE with GRACE potentials.

- **High-Performance Backend**  
  Utilizes the GRACE `tensorpotential` ASE calculator, with optional GPU acceleration via TensorFlow for efficient large-scale simulations.

## Installation

It is recommended to install **vasp-grace** within a dedicated Python environment:

```bash
uv pip install numpy tensorflow
uv pip install tensorpotential
uv pip install git+https://gitlab.com/ase/ase.git
