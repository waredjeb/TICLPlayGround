import uproot, numpy as np, awkward as ak
from glob import glob
from tqdm import tqdm
from collections import defaultdict
import awkward as ak
import os, pathlib
import ROOT
from utils import *


inputFolder = "/eos/cms/store/group/dpg_hgcal/comm_hgcal/wredjeb/ForGNN/MultiParticle_5_10_0PU/CMSSW_15_1_X/D110/CloseByMP_ForGNN_0PU/histo"
fileList    = glob(f"{inputFolder}/*.root")
Nfiles = 10
print("Found", len(fileList), f"ROOT files, will process {Nfiles}")

tmp_store = defaultdict(list)

for f in tqdm(fileList[:Nfiles], desc="reading"):
    with uproot.open(f)["ticlDumper"] as tdir:
        basenames = {key.rsplit(";", 1)[0] for key in tdir.keys()}
        for base in basenames:
            tree = tdir[base] 
            tmp_store[base].append(tree.arrays(library="ak"))

dump_data = {name: ak.concatenate(arr_list) for name, arr_list in tmp_store.items()}
print("Trees read:", list(dump_data))


outdir = pathlib.Path("parquet_out")
outdir.mkdir(exist_ok=True)

for name, arr in dump_data.items():
    path = outdir / f"{name}.parquet"
    ak.to_parquet(arr, path, compression=None)
    print("Wrote", path)
