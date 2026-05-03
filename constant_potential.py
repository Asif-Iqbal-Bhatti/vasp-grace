#!/usr/bin/env python3
"""
constant_potential.py

Constant-Potential (CP) Electrochemistry Module for GRACE ML Potentials
Implements grand canonical (μVT) ensemble MD with electron dynamics feedback
Based on CP-MACE approach but adapted for GRACE.

Key Features:
- Electron count as dynamic variable
- Fermi level feedback loop
- Target electrode potential control
- Compatible with GRACE GRACE-2L-OAM model
"""

import numpy as np
from ase import units
from ase.md.md import MolecularDynamics
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)


class CPIntegrator(MolecularDynamics):
    """
    Constant-Potential Molecular Dynamics Integrator

    Implements Velocity Verlet with electron dynamics feedback to maintain
    a target Fermi level (electrode potential). Electron count is updated
    each MD step based on the difference between target and current Fermi level.

    Parameters
    ----------
    atoms : ASE Atoms object
        System to simulate
    timestep : float
        Time step in fs
    temperature_K : float
        Temperature in Kelvin for Maxwell-Boltzmann velocity initialization
    targetmu : float
        Target Fermi level / electrode potential in eV
    Mne : float
        Effective electron mass (controls electron response timescale)
        Typical range: 600-1100
    trajectory : str, optional
        Trajectory file name (ASE format)
    logfile : str, optional
        Log file name
    """

    def __init__(self, atoms, timestep, temperature_K, targetmu, Mne,
                 trajectory=None, logfile=None):
        MolecularDynamics.__init__(self, atoms, timestep=timestep,
                                   trajectory=trajectory, logfile=logfile)

        self.targetmu = targetmu
        self.Mne = Mne
        self.temperature_K = temperature_K

        # Electron dynamics state
        self.electron_velocity = 0.0
        self.nelect = atoms.info.get('electron', 0.0)

        # Ensure atoms have electron count initialized
        if 'electron' not in atoms.info:
            atoms.info['electron'] = self.nelect

        self.nsteps = 0
        self.observers = []

    def attach(self, function, interval=1):
        """Attach observer function to be called every interval steps."""
        self.observers.append((function, interval))

    def get_mu(self):
        """Get current Fermi level from calculator."""
        try:
            # Try to get potential from calculator results
            if 'potential' in self.atoms.calc.results:
                return self.atoms.calc.results['potential']
            else:
                # Fallback if potential not available
                print("Warning: Fermi level (potential) not in calculator results.")
                print("         Defaulting to 0.0 eV")
                return 0.0
        except Exception as e:
            print(f"Warning: Could not retrieve Fermi level: {e}")
            return 0.0

    def run(self, steps=0, fmax=None):
        """
        Run constant-potential MD for specified number of steps.

        Parameters
        ----------
        steps : int
            Number of MD steps to run
        fmax : float, optional
            Not used (for compatibility with ASE interface)
        """

        # Initialize velocities if needed
        if not hasattr(self.atoms, 'get_velocities') or \
           np.allclose(self.atoms.get_velocities(), 0.0):
            if self.temperature_K > 0:
                MaxwellBoltzmannDistribution(self.atoms,
                                            temperature_K=self.temperature_K)
                Stationary(self.atoms)
                ZeroRotation(self.atoms)

        print(f"\n--- Constant-Potential MD (CP-GRACE) ---")
        print(f"Target Fermi Level: {self.targetmu:.4f} eV")
        print(f"Electron Mass (Mne): {self.Mne:.2f}")
        print(f"Temperature: {self.temperature_K:.1f} K")
        print(f"Timestep: {self.dt / units.fs:.4f} fs")
        print(f"Total steps: {steps}\n")

        # Main MD loop
        for step in range(steps):
            # Get forces and energies from calculator
            forces = self.atoms.get_forces()
            energy = self.atoms.get_potential_energy()

            # Get Fermi level (potential) from calculator
            mu = self.get_mu()

            # Current velocity and positions
            velocities = self.atoms.get_velocities()
            positions = self.atoms.get_positions()
            masses = self.atoms.get_masses()

            # Velocity Verlet: update positions
            new_positions = (positions +
                           velocities * self.dt +
                           0.5 * forces / masses[:, np.newaxis] * self.dt**2)

            self.atoms.set_positions(new_positions)

            # Get new forces at updated positions
            new_forces = self.atoms.get_forces()

            # Velocity Verlet: update velocities
            new_velocities = (velocities +
                            0.5 * (forces + new_forces) / masses[:, np.newaxis] * self.dt)

            self.atoms.set_velocities(new_velocities)

            # ============================================
            # Electron Dynamics (CRITICAL PART)
            # ============================================
            # Update electron velocity based on Fermi level feedback
            # Equation: v_e += (targetmu - mu) * dt / (2 * Mne)
            self.electron_velocity += (self.targetmu - mu) * self.dt / (2.0 * self.Mne)

            # Update electron count
            self.nelect += self.electron_velocity * self.dt

            # Store in atoms.info for next calculator call
            self.atoms.info['electron'] = self.nelect

            # ============================================
            # Call observer functions
            # ============================================
            self.nsteps += 1
            for func, interval in self.observers:
                if self.nsteps % interval == 0:
                    func()

            # Print progress
            if (step + 1) % max(1, steps // 10) == 0:
                print(f"  Step {step + 1:4d}/{steps}: E = {energy:12.6f} eV, "
                      f"μ = {mu:8.4f} eV, N_e = {self.nelect:8.2f}, "
                      f"ΔV_e = {self.electron_velocity:.6f}")

        print(f"\nConstant-Potential MD completed successfully.\n")


def run_constant_potential_md(atoms, incar):
    """
    Run constant-potential electrochemical MD with GRACE.

    This function orchestrates the CP-GRACE simulation:
    1. Validates GRACE calculator can output Fermi level
    2. Initializes electron count
    3. Creates CPIntegrator with feedback loop
    4. Attaches output observer
    5. Runs MD and generates VASP-like output files

    Parameters
    ----------
    atoms : ASE Atoms object
        Structure to simulate
    incar : dict
        Parsed INCAR parameters
    """
    from elastic import get_elementary_deformations, get_elastic_tensor
    from elastic import get_cij_order, get_lattice_type

    print("\n--- Constant-Potential GRACE MD ---")
    print(f"Target Potential: {incar['TARGET_POTENTIAL']:.4f} eV")
    print(f"Electron Mass: {incar['ELECTRON_MASS']:.2f}")

    # Validate calculator outputs Fermi level
    energy = atoms.get_potential_energy()
    if 'potential' not in atoms.calc.results:
        print("Error: GRACE calculator does not output Fermi level ('potential' key).")
        print("Your GRACE model must be trained to predict Fermi level.")
        print("See CP-MACE documentation for training instructions.")
        import sys
        sys.exit(1)

    # Initialize electron count
    nelect_init = incar.get("NELECT_INIT", None)
    if nelect_init is None:
        # Try to infer from atoms.info or use a default
        if 'electron' in atoms.info:
            nelect_init = atoms.info['electron']
        else:
            print("Warning: NELECT_INIT not specified and not in atoms.info")
            print("         Defaulting to number of valence electrons (rough estimate)")
            nelect_init = len(atoms) * 8  # Rough estimate for metals

    atoms.info['electron'] = nelect_init

    # Parse MD parameters
    dt = float(incar["POTIM"]) * units.fs if incar["POTIM"] is not None else 1.0 * units.fs
    nsw = incar.get("NSW", 1)
    temperature = incar.get("TEBEG", 300.0)

    # Create CP integrator
    cp_md = CPIntegrator(
        atoms,
        timestep=dt,
        temperature_K=temperature,
        targetmu=incar['TARGET_POTENTIAL'],
        Mne=incar['ELECTRON_MASS']
    )

    # Import observer for VASP-like output writing
    # This is defined in main.py, but we'll create it here
    class VaspWriterObserverCP:
        """Observer for writing CP-GRACE output in VASP format."""
        def __init__(self, atoms, is_md=False):
            self.atoms = atoms
            self.step = 1
            self.is_md = is_md

            from ase.io import write

            open("OSZICAR", "w").close()
            open("OUTCAR", "w").close()
            self._init_xdatcar()

        def _init_xdatcar(self):
            """Initialize XDATCAR file with header."""
            from itertools import groupby
            with open("XDATCAR", "w") as f:
                f.write("System from CP-GRACE\n")
                f.write("  1.00000000000000\n")
                for row in self.atoms.get_cell():
                    f.write(f"    {row[0]:12.8f}  {row[1]:12.8f}  {row[2]:12.8f}\n")

                symbols = self.atoms.get_chemical_symbols()
                grouped = [(k, len(list(g))) for k, g in groupby(symbols)]
                f.write("  " + "  ".join([g[0] for g in grouped]) + "\n")
                f.write("  " + "  ".join([str(g[1]) for g in grouped]) + "\n")

        def _append_xdatcar(self):
            """Append current configuration to XDATCAR."""
            with open("XDATCAR", "a") as f:
                f.write(f"Direct configuration={self.step:8d}\n")
                scaled_positions = self.atoms.get_scaled_positions(wrap=False)
                for pos in scaled_positions:
                    f.write(f"  {pos[0]:12.8f}  {pos[1]:12.8f}  {pos[2]:12.8f}\n")

        def __call__(self):
            """Called each MD step to write output."""
            from ase.io import write

            energy = self.atoms.get_potential_energy()
            forces = self.atoms.get_forces()

            # Get stress safely
            try:
                stress = self.atoms.get_stress(voigt=True)
            except Exception:
                stress = np.zeros(6)

            # Write OSZICAR line
            with open("OSZICAR", "a") as f:
                if self.is_md:
                    try:
                        temp = self.atoms.get_temperature()
                        kin_e = self.atoms.get_kinetic_energy()
                    except:
                        temp = 0.0
                        kin_e = 0.0
                    tot_e = energy + kin_e
                    mu = self.atoms.calc.results.get('potential', 0.0)
                    f.write(
                        f"   {self.step:4d} T={temp:8.2f} E={tot_e:15.8E} "
                        f"F={energy:15.8E} EK={kin_e:15.8E} μ={mu:8.4f} "
                        f"N_e={self.atoms.info.get('electron', 0.0):8.2f}\n"
                    )

            # Write OUTCAR block
            self._format_and_write_outcar(energy, forces, stress)

            # Write CONTCAR
            write("CONTCAR", self.atoms, format="vasp")

            # Append to XDATCAR
            self._append_xdatcar()

            self.step += 1

        def _format_and_write_outcar(self, energy, forces, stress_voigt):
            """Format and write OUTCAR block."""
            eV_to_kB = 1602.1766208
            s_xx, s_yy, s_zz, s_yz, s_xz, s_xy = stress_voigt * eV_to_kB

            lines = [
                "--------------------------------------------------------------------------------",
                f" STEP {self.step}",
                "--------------------------------------------------------------------------------",
                "  FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)",
                "  ---------------------------------------------------",
                f"  free  energy   TOTEN  = {energy:18.8f} eV\n",
                " POSITION                                       TOTAL-FORCE (eV/Angst)",
                " -----------------------------------------------------------------------------------"
            ]

            for pos, force in zip(self.atoms.positions, forces):
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

            with open("OUTCAR", "a") as f:
                f.write("\n".join(lines))

    # Attach observer
    observer = VaspWriterObserverCP(atoms, is_md=True)
    cp_md.attach(observer, interval=1)
    observer()  # Write initial step

    # Run CP-GRACE MD
    cp_md.run(nsw)

    print("CP-GRACE MD completed.")
    print("Wrote OUTCAR, OSZICAR, CONTCAR, XDATCAR")
    print(f"Final electron count: {atoms.info.get('electron', 0.0):.2f}")
    print(f"Final Fermi level: {atoms.calc.results.get('potential', 0.0):.4f} eV")
