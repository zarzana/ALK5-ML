from multiprocessing import freeze_support
from rdkit import Chem
from mordred import Calculator, descriptors

import numpy as np

if __name__ == '__main__':

    with open('smiles.smi') as file:
        smiles = file.read().splitlines()

    freeze_support()
    calc = Calculator(descriptors, ignore_3D=True)
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    calculated_descriptors = calc.pandas(mols)

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    calculated_descriptors = calculated_descriptors.select_dtypes(include=numerics)
    calculated_descriptors.to_csv('all_descriptors.csv', index=False)
