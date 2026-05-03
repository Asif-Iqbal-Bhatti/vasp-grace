#!/usr/bin/env python3
"""
vasp_grace.py

A standalone Python script to use GRACE Machine Learning Potentials
as a drop-in replacement for VASP. It supports:

- Single Point calculations
- Geometry Optimizations
- Molecular Dynamics
- Finite-displacement phonons (ASE backend) triggered by standard VASP tags:
    IBRION = 5 or 6

Design choice for phonons in this version:
- No new custom INCAR tags are introduced.
- The supplied POSCAR is treated as the actual working cell to displace.
  Therefore, if the user wants a supercell phonon calculation, they should
  provide a supercell POSCAR (consistent with finite-displacement workflows).
- IBRION=5 and IBRION=6 are treated identically in the ASE backend.
- NFREE=2 is the supported mode; other values warn and fall back conceptually
  to the ASE central-difference workflow.
"""

import os, warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

import sys
import argparse
import numpy as np
from itertools import groupby
import matplotlib.pyplot as plt

from ase import units
from ase.io import read, write
from ase.optimize import LBFGS, FIRE2
from ase.filters import FrechetCellFilter

# Molecular Dynamics
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.md.andersen import Andersen
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.md.nose_hoover_chain import MTKNPT
from ase.phonons import Phonons

from elastic import get_elementary_deformations, get_elastic_tensor
from elastic import get_cij_order, get_lattice_type

# GRACE / TensorPotential
from tensorpotential.calculator.foundation_models import grace_fm
from tensorpotential.calculator import TPCalculator

# Constant-Potential Electrochemistry
from constant_potential import run_constant_potential_md


# ==========================================
# 1. Helpers
# ==========================================

def parse_bool(val):
    """Parse VASP-like booleans."""
    return str(val).strip().upper() in {"T", ".TRUE.", "TRUE", "1", "YES", "Y"}

def safe_get_stress(atoms):
    """Safely fetch stress, returning zeros if unavailable."""
    try:
        stress = atoms.get_stress(voigt=True)
        return stress if stress is not None else np.zeros(6)
    except Exception:
        return np.zeros(6)

def get_calculator(model_identifier):
    """Load the appropriate GRACE ML potential."""
    if os.path.exists(model_identifier):
        print(f"Loading custom GRACE model from path: {model_identifier}")
        return TPCalculator(model_identifier)
    else:
        print(f"Loading GRACE foundation model: {model_identifier}")
        return grace_fm(model_identifier)

def generate_dummy_potcar(atoms):
    """Generate a dummy POTCAR to prevent downstream VASP parsers from failing."""
    if not os.path.exists("POTCAR"):
        symbols = list(dict.fromkeys(atoms.get_chemical_symbols()))
        with open("POTCAR", "w") as f:
            for elem in symbols:
                f.write(f" PAW_PBE {elem}\n")

def parse_mesh_file(filepath):
    """
    Try to parse a simple VASP-style uniform mesh from QPOINTS/KPOINTS.

    Supported format example:
        comment
        0
        Gamma
        20 20 20
        0 0 0

    Returns (nx, ny, nz) or None.
    """
    if not os.path.exists(filepath):
        return None

    try:
        with open(filepath, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        if len(lines) < 4:
            return None

        mode = lines[2].lower()
        if "gamma" in mode or "monkhorst" in mode or "auto" in mode:
            parts = lines[3].split()
            if len(parts) >= 3:
                mesh = tuple(int(float(x)) for x in parts[:3])
                if all(m > 0 for m in mesh):
                    return mesh
    except Exception:
        pass

    return None

def write_simple_phonon_outcar(atoms, energy, gamma_energies_eV):
    """
    Write a simple OUTCAR-like phonon section containing Gamma frequencies.
    """
    ev_to_thz = 241.79893
    ev_to_cm1 = 8065.54429

    with open("OUTCAR", "w") as f:
        f.write("--------------------------------------------------------------------------------\n")
        f.write(" ASE/GRACE finite-displacement phonon calculation (VASP-like wrapper)\n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write(f"  free  energy   TOTEN  = {energy:18.8f} eV\n\n")
        f.write(" Eigenvectors and eigenvalues of the dynamical matrix\n")
        f.write(" ----------------------------------------------------\n")
        f.write("  mode    energy (eV)        freq (THz)        freq (cm-1)\n")
        f.write(" ---------------------------------------------------------\n")

        gamma_energies_eV = np.array(gamma_energies_eV, dtype=float).ravel()
        for i, e in enumerate(gamma_energies_eV, start=1):
            if e >= 0:
                thz = e * ev_to_thz
                cm1 = e * ev_to_cm1
                f.write(f" {i:5d}   {e:14.8f}   {thz:14.6f}   {cm1:14.6f}\n")
            else:
                thz = abs(e) * ev_to_thz
                cm1 = abs(e) * ev_to_cm1
                f.write(f" {i:5d}   {e:14.8f}   f/i={thz:10.6f}   f/i={cm1:10.6f}\n")

        f.write("\n")

    write("CONTCAR", atoms, format="vasp")

def write_total_dos_file(dos, filename="phonon_dos.dat"):
    """Write DOS data to a text file."""
    energies = np.array(dos.get_energies(), dtype=float)
    weights = np.array(dos.get_weights(), dtype=float)

    with open(filename, "w") as f:
        f.write("# Energy(eV)    DOS\n")
        for e, w in zip(energies, weights):
            f.write(f"{e:16.8e} {w:16.8e}\n")

def make_band_plot(bs, filename="phonon_band.png"):
    """Plot and save phonon band structure if matplotlib is available."""
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    emax = 0.4
    bs.plot(ax=ax, emin=0.0, emax=emax)
    ax.set_title("Phonon band structure")
    fig.tight_layout()
    fig.savefig(filename, dpi=400)
    plt.close(fig)

def make_dos_plot(dos, filename="phonon_dos.png"):
    """Plot and save DOS if matplotlib is available."""
    energies = np.array(dos.get_energies(), dtype=float)
    weights = np.array(dos.get_weights(), dtype=float)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    ax.plot(energies, weights)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("DOS")
    ax.set_title("Phonon DOS")
    fig.tight_layout()
    fig.savefig(filename, dpi=400)
    plt.close(fig)


# ==========================================
# 2. Output Formatting (Mock VASP Files)
# ==========================================

def format_outcar_block(atoms, energy, forces, stress_voigt, step):
    """Format a single iteration block for the OUTCAR file mimicking VASP style."""
    eV_to_kB = 1602.1766208
    s_xx, s_yy, s_zz, s_yz, s_xz, s_xy = stress_voigt * eV_to_kB

    lines = [
        "--------------------------------------------------------------------------------",
        f" STEP {step}",
        "--------------------------------------------------------------------------------",
        "  FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)",
        "  ---------------------------------------------------",
        f"  free  energy   TOTEN  = {energy:18.8f} eV\n",
        " POSITION                                       TOTAL-FORCE (eV/Angst)",
        " -----------------------------------------------------------------------------------"
    ]

    for pos, force in zip(atoms.positions, forces):
        lines.append(
            f" {pos[0]:12.5f} {pos[1]:12.5f} {pos[2]:12.5f}    "
            f"{force[0]:12.5f} {force[1]:12.5f} {force[2]:12.5f}"
        )

    lines.append(" -----------------------------------------------------------------------------------\n")
    lines.append("  FORCE on cell =-STRESS in cart. coord.  units (eV):")
    lines.append("  Direction    XX          YY          ZZ          XY          YZ          ZX")
    lines.append("  --------------------------------------------------------------------------------------")
    lines.append(
        f"  in kB     {s_xx:10.5f}  {s_yy:10.5f}  {s_zz:10.5f}  "
        f"{s_xy:10.5f}  {s_yz:10.5f}  {s_xz:10.5f}\n"
    )

    return "\n".join(lines) + "\n"

class VaspWriterObserver:
    """Observer attached to ASE optimizers/MD to write VASP-like output iteratively."""
    def __init__(self, atoms, is_md=False):
        self.atoms = atoms
        self.step = 1
        self.is_md = is_md

        open("OSZICAR", "w").close()
        open("OUTCAR", "w").close()

        self._init_xdatcar()

    def _init_xdatcar(self):
        with open("XDATCAR", "w") as f:
            f.write("System from vasp-grace\n")
            f.write("  1.00000000000000\n")

            for row in self.atoms.get_cell():
                f.write(f"    {row[0]:12.8f}  {row[1]:12.8f}  {row[2]:12.8f}\n")

            symbols = self.atoms.get_chemical_symbols()
            grouped = [(k, len(list(g))) for k, g in groupby(symbols)]

            f.write("  " + "  ".join([g[0] for g in grouped]) + "\n")
            f.write("  " + "  ".join([str(g[1]) for g in grouped]) + "\n")

    def _append_xdatcar(self):
        with open("XDATCAR", "a") as f:
            f.write(f"Direct configuration={self.step:8d}\n")
            scaled_positions = self.atoms.get_scaled_positions(wrap=False)
            for pos in scaled_positions:
                f.write(f"  {pos[0]:12.8f}  {pos[1]:12.8f}  {pos[2]:12.8f}\n")

    def __call__(self):
        energy = self.atoms.get_potential_energy()
        forces = self.atoms.get_forces()
        stress = safe_get_stress(self.atoms)

        with open("OSZICAR", "a") as f:
            if self.is_md:
                temp = self.atoms.get_temperature()
                kin_e = self.atoms.get_kinetic_energy()
                tot_e = energy + kin_e
                f.write(
                    f"   {self.step:4d} T= {temp:8.2f} E= {tot_e:15.8E} "
                    f"F= {energy:15.8E} EK= {kin_e:15.8E}\n"
                )
            else:
                f.write(
                    f"   {self.step:4d} F= {energy:15.8E} E0= {energy:15.8E}  d E ={0.0:15.8E}\n"
                )

        with open("OUTCAR", "a") as f:
            f.write(format_outcar_block(self.atoms, energy, forces, stress, self.step))

        write("CONTCAR", self.atoms, format="vasp")
        self._append_xdatcar()
        self.step += 1

def write_vasp_single_point(atoms, energy, forces, stress_voigt):
    """Handle static output files for single-point calculations."""
    with open("OSZICAR", "w") as f:
        f.write(f"   1 F= {energy:15.8E} E0= {energy:15.8E}  d E ={0.0:15.8E}\n")

    with open("OUTCAR", "w") as f:
        f.write(format_outcar_block(atoms, energy, forces, stress_voigt, 1))

    write("CONTCAR", atoms, format="vasp")


# ==========================================
# 3. INCAR Parsing
# ==========================================

def parse_incar(filepath="INCAR"):
    """
    Read standard INCAR parameters plus GRACE_MODEL.
    Phonon logic uses only original VASP phonon tags:
    IBRION, NFREE, POTIM, LPHON_DISPERSION, PHON_DOS, PHON_NEDOS, PHON_SIGMA.

    Constant-Potential (CP) electrochemistry tags:
    CONSTANT_POTENTIAL, TARGET_POTENTIAL, ELECTRON_MASS, NELECT_INIT.
    """
    params = {
        # Basic run control
        "IBRION": -1,
        "NSW": 0,
        "ISIF": 3,
        "EDIFFG": -0.01,

        # GRACE model
        "GRACE_MODEL": "GRACE-2L-OAM",

        # MD / ionic params
        "POTIM": None,
        "TEBEG": 300.0,
        "MDALGO": 0,

        # VASP phonon-style tags
        "NFREE": 2,
        "LPHON_DISPERSION": False,
        "PHON_DOS": 0,
        "PHON_NEDOS": 1000,
        "PHON_SIGMA": 0.001,
        "PHON_NWRITE": 0,
        "LEPSILON": False,
        "LCALCEPS": False,
        "ISYM": 0,

        # Constant-Potential Electrochemistry
        "CONSTANT_POTENTIAL": False,
        "TARGET_POTENTIAL": -3.36,
        "ELECTRON_MASS": 660.74,
        "NELECT_INIT": None,
    }

    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            for line in f:
                line = line.split("#")[0].split("!")[0].strip()
                if "=" not in line:
                    continue

                key, val = [x.strip() for x in line.split("=", 1)]
                key = key.upper()

                if key not in params:
                    continue

                default = params[key]
                try:
                    if isinstance(default, bool):
                        params[key] = parse_bool(val)
                    elif isinstance(default, int):
                        params[key] = int(val)
                    elif isinstance(default, float):
                        params[key] = float(val)
                    elif default is None:
                        params[key] = float(val)
                    else:
                        params[key] = val
                except Exception:
                    # Keep default on parse failure
                    pass

    return params


def write_elastic_results(Cij, Bij, cryst, filename_cij="ELASTIC_Cij.dat", filename_bij="ELASTIC_Bij.dat"):
    """
    Save elastic outputs from jochym/Elastic.

    get_elastic_tensor returns:
        Cij, Bij
    where Bij is the full lstsq return tuple:
        (birch_coeffs, residuals, rank, singular_values)
    """
    Cij_arr = np.array(Cij, dtype=float)

    birch_coeffs = np.array(Bij[0], dtype=float)
    residuals = np.asarray(Bij[1], dtype=float)
    rank = int(Bij[2])
    singular_values = np.asarray(Bij[3], dtype=float)

    # Save Cij
    np.savetxt(
        filename_cij,
        np.atleast_2d(Cij_arr),
        header="Cij returned by elastic.get_elastic_tensor (ASE units)"
    )
    
    order = get_cij_order(cryst)
    values = np.array(Cij, dtype=float) / units.GPa

    with open("ELASTIC_Cij_GPa.dat", "w") as f:
        f.write("Elastic constants (GPa)\n")
        f.write("======================\n")
        for name, value in zip(order, values):
            f.write(f"{name:>5s} = {value:14.8f} GPa\n")

    # Save Birch coefficients only
    np.savetxt(
        filename_bij,
        np.atleast_2d(birch_coeffs),
        header="Birch coefficients Bij[0] from elastic.get_elastic_tensor (ASE units)"
    )
    np.savetxt(
        "ELASTIC_Bij_GPa.dat",
        np.atleast_2d(birch_coeffs / units.GPa),
        header="Birch coefficients in GPa"
    )

    # Save fit diagnostics safely
    with open("ELASTIC_fit_info.txt", "w") as f:
        f.write("jochym/Elastic least-squares fit diagnostics\n")
        f.write("===========================================\n")
        f.write(f"rank = {rank}\n")

        f.write("residuals = ")
        if residuals.size == 0:
            f.write("[]")
        else:
            f.write(" ".join(f"{x:.12e}" for x in np.atleast_1d(residuals).ravel()))
        f.write("\n")

        f.write("singular_values = ")
        if singular_values.size == 0:
            f.write("[]")
        else:
            f.write(" ".join(f"{x:.12e}" for x in np.atleast_1d(singular_values).ravel()))
        f.write("\n")
        
    
def run_elastic_tensor_with_jochym(atoms, incar):
    """
    Compute elastic constants using jochym/Elastic.

    Intended VASP-like trigger:
      IBRION = 5 or 6
      ISIF >= 3

    Workflow:
      1. Assume input structure is already reasonably relaxed (as in VASP practice).
      2. Generate elementary strained structures using Elastic.
      3. For each strained structure, relax internal coordinates only (fixed cell)
         using ASE optimizer + GRACE calculator.
      4. Evaluate stress on each strained structure.
      5. Pass all strained systems to Elastic to obtain Cij, Bij.
    """

    print("\n--- Elastic tensor mode (jochym/Elastic) ---")
    print("Trigger condition met: ISIF >= 3 together with IBRION = 5/6")

    # Reference structure
    cryst = atoms.copy()
    cryst.calc = get_calculator(incar["GRACE_MODEL"])

    # Check residual stress on the reference structure
    try:
        ref_stress = cryst.get_stress(voigt=True)
        max_ref_stress_gpa = np.max(np.abs(ref_stress / units.GPa))
        print(f"Reference max |stress| = {max_ref_stress_gpa:.4f} GPa")
        if max_ref_stress_gpa > 1.0:
            print("Warning: reference structure is not close to zero stress.")
            print("         Elastic constants are best computed from a pre-relaxed structure.")
    except Exception:
        print("Warning: could not evaluate reference stress before elastic workflow.")

    # n = number of points per elementary deformation
    # d = deformation magnitude parameter as used in Elastic examples
    systems = get_elementary_deformations(cryst, n=5, d=0.33)

    print(f"Generated {len(systems)} elementary deformations")

    # For VASP, the Elastic docs switch to internal degrees of freedom optimization
    # before the strained stress calculations. We emulate that by relaxing atomic
    # positions only (fixed cell) with ASE optimizer on each strained structure.
    #
    # Use a force threshold consistent
    ediffg = incar.get("EDIFFG", -0.01)
    fmax = abs(ediffg) if ediffg < 0 else 0.05

    relaxed_systems = []

    for i, s in enumerate(systems, start=1):
        s = s.copy()
        s.calc = get_calculator(incar["GRACE_MODEL"])

        print(f"  deformation {i:3d}/{len(systems):3d}: relaxing internal coordinates...")

        # Fixed-cell ionic relaxation only
        opt = FIRE2(s, logfile=None)
        opt.run(fmax=fmax, steps=max(incar.get("NSW", 50), 50))

        # Ensure stress is evaluated and cached
        _ = s.get_potential_energy()
        _ = s.get_stress(voigt=True)

        relaxed_systems.append(s)

    # Now fit elastic tensor from the strained systems
    Cij, Bij = get_elastic_tensor(cryst, systems=relaxed_systems)
    order = get_cij_order(cryst)
    write_elastic_results(Cij, Bij, cryst)

    print("Elastic tensor calculation completed.")
    print("Wrote:")
    print("  - ELASTIC_Cij.dat")
    print("  - ELASTIC_Bij.dat")
    print("  - ELASTIC_Cij_GPa.dat")
    print("  - ELASTIC_Bij_GPa.dat")
    print("  - ELASTIC_fit_info.txt")
    
    print("\nElastic summary:")
    print("Cij (GPa):")
    for name, value in zip(order, np.array(Cij, dtype=float) / units.GPa):
        print(f"{name} = {value:12.6f} GPa")
    
    print("Birch coefficients (GPa):")
    print(np.array(Bij[0], dtype=float) / units.GPa)
    
    print("Fit residuals:")
    print(np.array(Bij[1], dtype=float))
    
    print("Fit rank:")
    print(Bij[2])
    
    print("Singular values:")
    print(np.array(Bij[3], dtype=float))
    print()
    
# ==========================================
# 4. ASE Phonon Driver
# ==========================================

def run_ase_phonons(atoms, incar):
    """
    Run finite-displacement phonons using ASE.

    Trigger:
        IBRION = 5 or 6

    Interpretation:
        The provided POSCAR is the actual cell to be displaced.
        Therefore supercell=(1,1,1) is used inside ASE. If a larger phonon
        supercell is desired, the user should supply that supercell as POSCAR.
    """
    N = 3

    if atoms.cell.rank < 3:
        print("Error: ASE phonons require a proper 3D periodic cell.")
        sys.exit(1)

    nfree = incar.get("NFREE", 2)
    if nfree != 2:
        print(f"Warning: ASE Phonons is a central-difference finite-displacement workflow.")
        print(f"         Requested NFREE={nfree}; continuing with the ASE standard ±delta approach.")

    if incar["LEPSILON"] or incar["LCALCEPS"]:
        print("Warning: LEPSILON/LCALCEPS requested, but Born-charge / non-analytic")
        print("         correction is not used in this ASE wrapper.")

    if incar["IBRION"] == 6:
        print("Note: IBRION=6 requested.")
        print("      In the ASE backend, IBRION=5 and IBRION=6 are treated the same.")

    nsw = incar.get("NSW", 1)
    if nsw != 1:
        print(f"Warning: For VASP-style phonons, NSW is typically 1. You set NSW={nsw}.")
        print("         Continuing anyway.")

    # VASP default-like phonon displacement if POTIM missing or unreasonable
    potim = incar.get("POTIM", None)
    if potim is None or potim <= 0 or potim > 0.2:
        delta = 0.015
    else:
        delta = float(potim)
    
    print("\n--- ASE phonon mode (triggered by IBRION=5/6) ---")
    print(f"Using displacement delta = {delta:.6f} Å")
    print(f"Using ASE Phonons with supercell = {(N, N, N)}")
    print("Interpretation: the supplied POSCAR is the working phonon cell.\n")

    ph = Phonons(
        atoms,
        get_calculator(incar["GRACE_MODEL"]),
        supercell=(N, N, N),
        delta=delta,
        name="phonon",
    )

    # Run displaced calculations and build force constants
    ph.run()
    ph.read(method="Frederiksen", symmetrize=3, acoustic=True, cutoff=None, born=False)

    # Check equilibrium forces in the phonon reference structure
    try:
        fmin, fmax, i_min, i_max = ph.check_eq_forces()
        max_abs = max(abs(float(fmin)), abs(float(fmax)))
        print(f"Reference-cell max |force| before phonon postprocessing: {max_abs:.6e} eV/Å")
    except Exception:
        pass

    # Save force constants
    try:
        np.save("force_constants.npy", ph.get_force_constant())
        print("Wrote force_constants.npy")
    except Exception:
        pass

    # Reference energy on the unchanged structure
    atoms.calc = get_calculator(incar["GRACE_MODEL"])
    energy = atoms.get_potential_energy()

    # Gamma frequencies
    gamma_path = atoms.cell.bandpath(npoints=100)
    gamma_bs = ph.get_band_structure(gamma_path)
    gamma_energies = np.array(gamma_bs.energies)
    if gamma_energies.ndim == 3:
        gamma_energies = gamma_energies[0, 0, :]
    elif gamma_energies.ndim == 2:
        gamma_energies = gamma_energies[0, :]
    else:
        gamma_energies = gamma_energies.ravel()

    with open("OSZICAR", "w") as f:
        f.write(f"   1 F= {energy:15.8E} E0= {energy:15.8E}  d E ={0.0:15.8E}\n")

    write_simple_phonon_outcar(atoms, energy, gamma_energies)

    ev_to_thz = 241.79893
    ev_to_cm1 = 8065.54429
    with open("phonon_gamma.dat", "w") as f:
        f.write("# mode   energy_eV   freq_THz   freq_cm-1\n")
        for i, e in enumerate(gamma_energies, start=1):
            f.write(f"{i:4d} {e:14.8f} {e*ev_to_thz:14.6f} {e*ev_to_cm1:14.6f}\n")

    print("Wrote:")
    print("  - OUTCAR")
    print("  - OSZICAR")
    print("  - CONTCAR")
    print("  - phonon_gamma.dat")

    # Optional phonon band structure
    if incar.get("LPHON_DISPERSION", False):
        print("\nLPHON_DISPERSION = .TRUE. -> computing phonon band structure...")
        try:
            # Use ASE's automatic high-symmetry path for the provided cell
            path = atoms.cell.bandpath('XGN', npoints=200)
            bs = ph.get_band_structure(path)
            make_band_plot(bs, filename="phonon_band.png")
            print("Wrote:")
            print("  - phonon_band.png")
        except Exception as e:
            print(f"Warning: phonon band structure failed: {e}")

    # Optional DOS
    if int(incar.get("PHON_DOS", 0)) > 0:
        print("\nPHON_DOS > 0 -> computing phonon DOS...")

        mesh = (
            parse_mesh_file("QPOINTS")
            or parse_mesh_file("KPOINTS")
            or (20, 20, 20)
        )
        nedos = int(incar.get("PHON_NEDOS", 1000))
        sigma = float(incar.get("PHON_SIGMA", 0.001))

        print(f"Using q-mesh = {mesh}, PHON_NEDOS = {nedos}, PHON_SIGMA = {sigma}")

        try:
            dos = ph.get_dos(kpts=mesh).sample_grid(npts=nedos, width=sigma)
            write_total_dos_file(dos, filename="phonon_dos.dat")
            make_dos_plot(dos, filename="phonon_dos.png")
            print("Wrote:")
            print("  - phonon_dos.dat")
            print("  - phonon_dos.png")
        except Exception as e:
            print(f"Warning: phonon DOS failed: {e}")

    print("\nASE phonon calculation completed successfully.\n")


# ==========================================
# 5. Main Execution Routine
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description="Mock VASP execution using GRACE Machine Learning Potential."
    )
    parser.add_argument("--poscar", default="CONTCAR", help="Path to POSCAR file.")
    parser.add_argument("--incar", default="INCAR", help="Path to INCAR file.")
    args = parser.parse_args()

    if not os.path.exists(args.poscar):
        print(f"Error: {args.poscar} not found!")
        sys.exit(1)

    if not os.path.exists(args.incar):
        print(f"Error: {args.incar} not found!")
        sys.exit(1)

    print("--- vasp-grace initialized ---")

    atoms = read(args.poscar, format="vasp")
    incar = parse_incar(args.incar)
    generate_dummy_potcar(atoms)

    ibrion = incar["IBRION"]
    nsw = incar["NSW"]
    isif = incar["ISIF"]
    ediffg = incar["EDIFFG"]
    fmax = abs(ediffg) if ediffg < 0 else 0.05

    # ==========================================
    # ROUTINE 0: Phonons (VASP-style trigger)
    # ==========================================
    if ibrion in (5, 6):
        run_ase_phonons(atoms, incar)

        # VASP-like behavior: for finite differences, also compute elastic tensor
        # when ISIF >= 3.
        if incar["ISIF"] >= 3:
            run_elastic_tensor_with_jochym(atoms, incar)

        return

    if ibrion in (7, 8):
        print("Error: IBRION=7/8 correspond to DFPT phonons in VASP.")
        print("This wrapper currently implements only ASE finite-displacement phonons")
        print("for IBRION=5/6.")
        sys.exit(1)

    # Non-phonon workflows use attached calculator directly
    atoms.calc = get_calculator(incar["GRACE_MODEL"])

    # ==========================================
    # ROUTINE CP: Constant-Potential Electrochemistry
    # ==========================================
    if incar["CONSTANT_POTENTIAL"]:
        run_constant_potential_md(atoms, incar)
        return

    # ==========================================
    # ROUTINE A: Single Point
    # ==========================================
    if nsw <= 0 or ibrion == -1:
        print("Performing Single-Point Evaluation...")
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        stress = safe_get_stress(atoms)
        write_vasp_single_point(atoms, energy, forces, stress)
        print("Done. Wrote OUTCAR, OSZICAR, CONTCAR.")

    # ==========================================
    # ROUTINE B: Molecular Dynamics
    # ==========================================
    elif ibrion == 0:
        print(
            f"Performing Molecular Dynamics "
            f"(IBRION=0, TEBEG={incar['TEBEG']} K, POTIM={incar['POTIM']} fs)..."
        )

        if incar["TEBEG"] > 0:
            MaxwellBoltzmannDistribution(atoms, temperature_K=incar["TEBEG"])
            Stationary(atoms)
            ZeroRotation(atoms)

        dt = float(incar["POTIM"]) * units.fs if incar["POTIM"] is not None else 1.0 * units.fs
        temperature = incar["TEBEG"]

        if incar["ISIF"] >= 3:
            print("Using NPT Ensemble (ISIF>=3)...")
            target_pressure = 1.0 * units.bar
            md = MTKNPT(
                atoms,
                timestep=dt,
                temperature_K=temperature,
                pressure_au=target_pressure,
                tdamp=100 * dt,
                pdamp=1000 * dt,
                tchain=3,
                pchain=3,
                tloop=5,
                ploop=5,
            )
        else:
            if incar["MDALGO"] == 0:
                print("Using NVE Ensemble (Velocity Verlet, MDALGO=0)...")
                md = VelocityVerlet(atoms, timestep=dt)
            elif incar["MDALGO"] == 1:
                print("Using NVT Ensemble (Andersen Thermostat, MDALGO=1)...")
                md = Andersen(atoms, timestep=dt, temperature_K=temperature, andersen_prob=1e-4)
            elif incar["MDALGO"] == 2:
                print("Using NVT Ensemble (Berendsen Thermostat, MDALGO=2)...")
                md = NVTBerendsen(atoms, timestep=dt, temperature_K=temperature, taut=100 * units.fs)
            elif incar["MDALGO"] == 3:
                print("Using NVT Ensemble (Langevin Thermostat, MDALGO=3)...")
                md = Langevin(atoms, timestep=dt, temperature_K=temperature, friction=0.002)
            else:
                print("Fallback: Using NVE Ensemble (Velocity Verlet)...")
                md = VelocityVerlet(atoms, timestep=dt)

        observer = VaspWriterObserver(atoms, is_md=True)
        md.attach(observer, interval=1)
        observer()  # Write step 0
        md.run(nsw)
        print("MD successfully completed.")

    # ==========================================
    # ROUTINE C: Geometry Optimization
    # ==========================================
    else:
        print(f"Performing Geometry Optimization (IBRION={ibrion}, NSW={nsw}, ISIF={isif})...")

        if isif == 3:
            opt_target = FrechetCellFilter(atoms)
        elif isif == 4:
            opt_target = FrechetCellFilter(atoms, constant_volume=True)
        else:
            opt_target = atoms

        if ibrion == 1:
            optimizer = LBFGS(opt_target, trajectory="grace_opt.traj", logfile="grace_opt.log")
        else:
            optimizer = FIRE2(opt_target, trajectory="grace_opt.traj", logfile="grace_opt.log")

        observer = VaspWriterObserver(atoms, is_md=False)
        optimizer.attach(observer, interval=1)
        observer()  # Write step 0
        optimizer.run(fmax=fmax, steps=nsw)
        print("Optimization successfully completed.")


if __name__ == "__main__":
    main()