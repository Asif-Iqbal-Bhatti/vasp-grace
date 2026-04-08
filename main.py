import os
import sys
import argparse
from ase.io import read
from ase.optimize import LBFGS, FIRE2
from ase.filters import FrechetCellFilter

from tensorpotential.calculator.foundation_models import grace_fm
from tensorpotential.calculator import TPCalculator

from writer import VaspWriterObserver, write_vasp_single_point, safe_get_stress

def parse_incar(filepath="INCAR"):
    """Parse the VASP INCAR file for relaxation params & GRACE configurations."""
    params = {
        "IBRION": -1,
        "NSW": 0,
        "ISIF": 2,
        "EDIFFG": -0.01,
        "GRACE_MODEL": "GRACE-2L-OMAT" # Default GRACE model fallback
    }
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            for line in f:
                # Remove comments
                line = line.split("#")[0].split("!")[0].strip()
                if "=" in line:
                    key, val = [x.strip() for x in line.split("=", 1)]
                    key = key.upper()
                    if key in params:
                        # Auto-cast logic
                        if isinstance(params[key], int):
                            params[key] = int(val)
                        elif isinstance(params[key], float):
                            params[key] = float(val)
                        else:
                            params[key] = val
    return params


def get_calculator(model_identifier):
    """Initialize the GRACE potential calculator via tensorpotential."""
    if os.path.exists(model_identifier):
        print(f"Loading custom GRACE model from path: {model_identifier}")
        return TPCalculator(model_identifier)
    else:
        print(f"Loading GRACE foundation model: {model_identifier}")
        return grace_fm(model_identifier)


def generate_dummy_potcar(atoms):
    """Generate a dummy POTCAR if missing (solves Pymatgen parsing errors)."""
    if not os.path.exists("POTCAR"):
        symbols = set(atoms.get_chemical_symbols())
        with open("POTCAR", "w") as f:
            for elem in symbols:
                f.write(f" PAW_PBE {elem} \n")


def main():
    parser = argparse.ArgumentParser(description="Mock VASP execution using GRACE Machine Learning Potential.")
    parser.add_argument("--poscar", default="POSCAR", help="Path to POSCAR file.")
    parser.add_argument("--incar", default="INCAR", help="Path to INCAR file.")
    args = parser.parse_args()

    if not os.path.exists(args.poscar):
        print(f"Error: {args.poscar} not found! Run from a VASP working directory.")
        sys.exit(1)

    print("vasp-grace: Mock VASP initialized.")

    # 1. Read configuration & Atoms
    atoms = read(args.poscar, format="vasp")
    incar = parse_incar(args.incar)
    generate_dummy_potcar(atoms)
    
    # 2. Attach GRACE Calculator
    calc = get_calculator(incar["GRACE_MODEL"])
    atoms.calc = calc
    
    ibrion = incar["IBRION"]
    nsw = incar["NSW"]
    isif = incar["ISIF"]
    ediffg = incar["EDIFFG"]
    
    # Convert VASP EDIFFG to ASE fmax
    fmax = abs(ediffg) if ediffg < 0 else 0.05
    
    # 3. Execution logic
    if nsw <= 0 or ibrion == -1:
        print("Performing Single-Point Evaluation...")
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        stress = safe_get_stress(atoms)
        write_vasp_single_point(atoms, energy, forces, stress)
        print("Done. Wrote OUTCAR, OSZICAR, CONTCAR.")
        
    else:
        print(f"Performing Geometry Optimization (IBRION={ibrion}, NSW={nsw}, ISIF={isif})...")
        
        # Determine Cell Filtering based on ISIF
        if isif == 3:
            opt_target = FrechetCellFilter(atoms)
        elif isif == 4:
            opt_target = FrechetCellFilter(atoms, constant_volume=True)
        else: # ISIF = 2
            opt_target = atoms
            
        # Select optimizer based on IBRION
        if ibrion == 1:
            optimizer = LBFGS(opt_target, trajectory='grace_opt.traj', logfile='grace_opt.log')
        else:
            optimizer = FIRE2(opt_target, trajectory='grace_opt.traj', logfile='grace_opt.log')
            
        # Attach observer to mock VASP file generation sequentially 
        observer = VaspWriterObserver(atoms)
        optimizer.attach(observer, interval=1)
        
        # Write 0th step output
        observer()
        
        # Run optimization
        optimizer.run(fmax=fmax, steps=nsw)
        print("Optimization successfully completed.")

if __name__ == "__main__":
    main()
