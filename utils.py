import numpy as np
import awkward as ak
import ROOT
import matplotlib.pyplot as plt
import mplhep as hep
import functools, random
from copy import deepcopy
from typing import List, Dict, Tuple
hep.style.use(hep.style.CMS)


def collect_pass_fail_efficiency(
    simTrackstersCP,
    associations,
    shared_fraction_threshold: float,
    var: str = "energy",   
):
    """
    Parameters
    ----------
    simTrackstersCP   : Awkward Array (len = #events)
    associations      : Awkward Array (same length)
    shared_fraction_threshold : float (e.g. 0.5)
    var               : which variable to plot (energy/pt/eta/phi)

    Returns
    -------
    values : numpy.ndarray (float64)
    passed : numpy.ndarray (bool)
    """
    values, passed = [], []

    for ev in range(len(simTrackstersCP)):
        sim_ev     = simTrackstersCP[ev]
        assoc_ev   = associations[ev]
        sharedE_ev = assoc_ev.ticlCandidate_simToReco_CP_sharedE

        for iSim, E in enumerate(sim_ev.raw_energy):
            if   var == "energy":
                value = E
            elif var == "pt":
                value = sim_ev.regressed_pt[iSim]
            elif var == "eta":
                value = abs(sim_ev.barycenter_eta[iSim])        # <-- fixed here

            elif var == "phi":
                value = sim_ev.barycenter_phi[iSim]
            else:
                raise ValueError(f"Unknown var='{var}'")

            values.append(value)
            if len(sharedE_ev[iSim]) == 0:
                passed.append(False)
                continue

            shared_max = sharedE_ev[iSim][ak.argmax(sharedE_ev[iSim])]
            passed.append(shared_max / E > shared_fraction_threshold)

    return np.asarray(values, dtype="f8"), np.asarray(passed, dtype=bool)

def collect_pass_fail_merge(
    recoTracksters,
    associations,
    shared_fraction_threshold: float,
    var: str = "energy",          # choose: energy, pt, eta, phi
):
    """
    For every *Sim* trackster decide whether it is MERGED
    (i.e. overlaps above threshold with > 1 Reco trackster).

    Returns
    -------
    values : np.ndarray  – variable chosen by `var`
    passed : np.ndarray  – True if "merged", else False
    """
    values, passed = [], []

    for ev in range(len(recoTracksters)):
        reco_ev   = recoTracksters[ev]
        assoc_ev = associations[ev]

        # shortcuts to sim→reco match arrays
        sim_idx_ev   = assoc_ev.ticlCandidate_recoToSimCP
        sharedE_ev    = assoc_ev.ticlCandidate_recoToSimCP_sharedE
        score_ev    = assoc_ev.ticlCandidate_recoToSimCP_score

        for iReco, E in enumerate(reco_ev.raw_energy):
            # -------- X-axis variable ----------------------------------
            if   var == "energy":
                value = E
            elif var == "pt":
                value = reco_ev.regressed_pt[iSim]
            elif var == "eta":
                value = abs(reco_ev.barycenter_eta[iSim])
            elif var == "phi":
                value = reco_ev.barycenter_phi[iSim]
            else:
                raise ValueError(f"Unknown var='{var}'")
            values.append(value)

            # -------- merge criterion ---------------------------------
            # Count how many Reco trksters pass the shared-fraction cut
            n_high = 0
            for score in score_ev[iReco]:
                if score < shared_fraction_threshold:
                    n_high += 1
                if n_high > 1:            # early exit: already merged
                    break

            passed.append(n_high > 1)

    return np.asarray(values, dtype="f8"), np.asarray(passed, dtype=bool)

def collect_pass_fail_fake(
    recoTracksters,
    associations,
    shared_fraction_threshold: float,
    var: str = "energy",          # choose: energy, pt, eta, phi
):
    """
    For every *Sim* trackster decide whether it is MERGED
    (i.e. overlaps above threshold with > 1 Reco trackster).

    Returns
    -------
    values : np.ndarray  – variable chosen by `var`
    passed : np.ndarray  – True if "merged", else False
    """
    values, passed = [], []

    for ev in range(len(recoTracksters)):
        reco_ev   = recoTracksters[ev]
        assoc_ev = associations[ev]

        # shortcuts to sim→reco match arrays
        sim_idx_ev   = assoc_ev.ticlCandidate_recoToSimCP
        sharedE_ev    = assoc_ev.ticlCandidate_recoToSimCP_sharedE
        score_ev    = assoc_ev.ticlCandidate_recoToSimCP_score

        for iReco, E in enumerate(reco_ev.raw_energy):
            # -------- X-axis variable ----------------------------------
            if   var == "energy":
                value = E
            elif var == "pt":
                value = reco_ev.regressed_pt[iSim]
            elif var == "eta":
                value = abs(reco_ev.barycenter_eta[iSim])
            elif var == "phi":
                value = reco_ev.barycenter_phi[iSim]
            else:
                raise ValueError(f"Unknown var='{var}'")
            values.append(value)

            # -------- merge criterion ---------------------------------
            # Count how many Reco trksters pass the shared-fraction cut
            n_high = 0
            for score in score_ev[iReco]:
                if score > shared_fraction_threshold:
                    n_high += 1

            passed.append(n_high == 0)

    return np.asarray(values, dtype="f8"), np.asarray(passed, dtype=bool)


def _make_tefficiency(values, passed, bins):
    if isinstance(bins, int):
        edges = np.linspace(values.min(), values.max(), bins + 1, dtype="f8")
    else:
        edges = np.asarray(bins, dtype="f8")
    nb = len(edges) - 1

    h_tot  = ROOT.TH1F("h_tot",  "", nb, edges)
    h_pass = ROOT.TH1F("h_pass", "", nb, edges)

    for v, ok in zip(values, passed):
        h_tot.Fill(v)
        if ok:
            h_pass.Fill(v)

    teff = ROOT.TEfficiency(h_pass, h_tot)
    teff.SetStatisticOption(ROOT.TEfficiency.kFCP)
    return teff

def plot_efficiency_mpl(
    values,
    passed,
    bins, 
    x_label,
    y_label,
    label,
    title="Efficiency",
    fmt="o",
    ax=None,
    return_teff=False,
):
    """
    Make a Clopper–Pearson efficiency plot with matplotlib.
    """
    teff = _make_tefficiency(values, passed, bins)

    nb = teff.GetPassedHistogram().GetNbinsX()
    x, y, yerr_low, yerr_up = [], [], [], []
    for i in range(1, nb + 1):
        x.append(teff.GetTotalHistogram().GetBinCenter(i))
        y.append(teff.GetEfficiency(i))
        yerr_low.append(teff.GetEfficiencyErrorLow(i))
        yerr_up .append(teff.GetEfficiencyErrorUp(i))

    x, y = np.asarray(x), np.asarray(y)
    yerr = np.vstack([yerr_low, yerr_up])

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))

    ax.errorbar(x, y, yerr=yerr, fmt=fmt, capsize=3, linestyle="",
                label=label)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_ylim(0, 1.05)
    
    # ax.set_title(title)
    ax.grid(True)
    ax.legend()

    if return_teff:
        return ax, teff
    return ax


# ============================================================
#  Trackster, merging helpers and scoring  (copy–paste once)
# ============================================================
import awkward as ak
import functools, random
from copy import deepcopy
from typing import List, Dict, Tuple


class Trackster:
    # --------------------------------------------------------
    # constructor (core aliases made once)
    # --------------------------------------------------------
    def __init__(self, x, y, z, energy, lc_indices):
        self.x = self.barycenter_x = float(x)
        self.y = self.barycenter_y = float(y)
        self.z = self.barycenter_z = float(z)

        self.raw_energy = self.energy = float(energy)

        self.vertices_indexes      = list(lc_indices)
        self.lc_indices            = self.vertices_indexes
        self.vertices_multiplicity = []
        self.vertices_energy       = []

    # --------------------------------------------------------
    # build from event record + index, copying *all* fields
    # --------------------------------------------------------
    @classmethod
    def from_event_all(cls, event_rec, idx,
                       lc_key="vertices_indexes",
                       energy_key="raw_energy",
                       default_xyz=("barycenter_x",
                                    "barycenter_y",
                                    "barycenter_z")):
        if not isinstance(event_rec, ak.highlevel.Record):
            raise TypeError("event_rec must be an Awkward Record")

        x, y, z = (event_rec[f][idx] for f in default_xyz)
        energy  = event_rec[energy_key][idx]
        lc_idx  = event_rec[lc_key][idx]

        self = cls(x, y, z, energy, lc_idx)

        # fields already copied
        skip = set(default_xyz) | {energy_key, lc_key}

        for name in event_rec.fields:
            if name in skip:
                continue
            field = event_rec[name]
            if isinstance(field, ak.highlevel.Array):
                value = field[idx] if len(field) > idx else None
                # variable-length sub-arrays → plain list
                value = ak.to_list(value) if isinstance(value, ak.highlevel.Array) else value
                setattr(self, name, value)
            else:
                setattr(self, name, field)
        return self

    # --------------------------------------------------------
    # merge (returns NEW Trackster)
    # --------------------------------------------------------
    @staticmethod
    def _ewmean(a1, w1, a2, w2):
        s = w1 + w2
        return 0.0 if s == 0 else (a1 * w1 + a2 * w2) / s

    def merge(self, other):
        if not isinstance(other, Trackster):
            raise TypeError("other must be a Trackster")

        merged = deepcopy(self)

        merged.raw_energy = self.raw_energy + other.raw_energy
        merged.energy     = merged.raw_energy

        merged.barycenter_x = self._ewmean(self.barycenter_x, self.raw_energy,
                                           other.barycenter_x, other.raw_energy)
        merged.barycenter_y = self._ewmean(self.barycenter_y, self.raw_energy,
                                           other.barycenter_y, other.raw_energy)
        merged.barycenter_z = self._ewmean(self.barycenter_z, self.raw_energy,
                                           other.barycenter_z, other.raw_energy)
        merged.x, merged.y, merged.z = (merged.barycenter_x,
                                        merged.barycenter_y,
                                        merged.barycenter_z)

        # ---- safe list concatenation -----------------------------------
        merged.vertices_indexes = list(self.vertices_indexes) + list(other.vertices_indexes)
        merged.vertices_multiplicity = (
            list(getattr(self,  "vertices_multiplicity", [])) +
            list(getattr(other, "vertices_multiplicity", []))
        )
        merged.lc_indices = merged.vertices_indexes
        return merged

    def __repr__(self):
        return (f"Trackster(E={self.energy:.2f} GeV, "
                f"xyz=({self.x:.2f},{self.y:.2f},{self.z:.2f}), "
                f"nlc={len(self.lc_indices)})")



# ------------------------------------------------------------
# Sim ↔ Reco score and aggregator
# ------------------------------------------------------------
def compute_scores(sim_trk, reco_trk) -> Tuple[float, float]:
    if not all(hasattr(sim_trk, f) for f in ("vertices_indexes",
                                             "vertices_multiplicity",
                                             "vertices_energy")):
        raise AttributeError("sim_trk missing required vertices_* fields")

    num = den = shared = 0.0
    reco_set = set(reco_trk.vertices_indexes)

    for lc, mult, E in zip(sim_trk.vertices_indexes,
                           sim_trk.vertices_multiplicity,
                           sim_trk.vertices_energy):
        f_s = 1.0 / mult if mult else 0.0
        f_r = 1.0 if lc in reco_set else 0.0
        num += min((f_r - f_s)**2, f_s**2) * E**2
        den += (f_s**2) * E**2
        shared += f_r * f_s * E

    return round(num / den if den else 0.0, 8), round(shared, 8)


def compute_sim_to_reco_scores(sim_ev, merged_reco, *, best_only=False) -> List[Dict]:
    results: List[Dict] = []
    for i_sim in range(len(sim_ev.raw_energy)):
        sim_trk = Trackster.from_event_all(sim_ev, i_sim)
        best = None
        for j_reco, reco_trk in enumerate(merged_reco):
            score, shared = compute_scores(sim_trk, reco_trk)
            row = dict(sim_idx=i_sim, reco_idx=j_reco,
                       score=score, sharedE=shared)
            if best_only:
                if best is None or score < best["score"]:
                    best = row
            else:
                results.append(row)
        if best_only and best:
            results.append(best)
    return results

def compute_reco_to_sim_scores(sim_ev, merged_reco, *, best_only=False) -> List[Dict]:
    results: List[Dict] = []
    for i_reco in range(len(merged_reco)):
        reco_trk = merged_reco[i_reco]
        best = None
        for j_sim in range(len(sim_ev.raw_energy)):
            sim_trk = Trackster.from_event_all(sim_ev, j_sim)
            score, shared = compute_scores(reco_trk, sim_trk)
            row = dict(reco_idx=i_reco, sim_idx=j_sim,
                       score=score, sharedE=shared)
            if best_only:
                if best is None or score < best["score"]:
                    best = row
            else:
                results.append(row)
        if best_only and best:
            results.append(best)
    return results

# ----------------------------------------------------------------------
# E.  Collect ONE big (values, passed) pair for *each* variable
# ----------------------------------------------------------------------
def aggregate_for_efficiency(
    simTrackstersCP,
    score_tables,                   # list[list[dict]]  (same len as above)
    score_threshold: float = 0.5,
    vars=("energy", "pt", "eta", "phi"),
):
    """
    Build aggregated arrays for every variable in `vars`.

    Returns
    -------
    out : dict
        { "energy": (values_E, passed_E),
          "pt"    : (values_pt, passed_pt),
          "eta"   : (values_eta, passed_eta),
          "phi"   : (values_phi, passed_phi) }
    """
    # initialise empty lists for each variable
    out = {v: ([], []) for v in vars}   # maps var -> (values_list, passed_list)

    for ev, rows in enumerate(score_tables):
        if len(rows) == 0:
            continue
        sim_ev = simTrackstersCP[ev]

        for row in rows:
            iSim = row["sim_idx"]
            E    = sim_ev.raw_energy[iSim]

            # quick helpers
            getter = {
                "energy": lambda: E,
                "pt"    : lambda: sim_ev.regressed_pt[iSim],
                "eta"   : lambda: abs(sim_ev.barycenter_eta[iSim]),
                "phi"   : lambda: sim_ev.barycenter_phi[iSim],
            }

            passed_flag = (row["sharedE"] / E) > score_threshold if E else False

            for v in vars:
                out[v][0].append(getter[v]())     # value
                out[v][1].append(passed_flag)     # pass/fail

    # convert lists → NumPy arrays
    for v in vars:
        values, ok = out[v]
        out[v] = (np.asarray(values, dtype="f8"),
                  np.asarray(ok,     dtype=bool))
    return out

from typing import List, Dict, Tuple, Sequence

def aggregate_for_merge(
    merged_tracksters_per_event: Sequence[Sequence[Trackster]],
    score_tables:                Sequence[List[Dict]],
    *,
    score_threshold: float = 0.6,
    vars: Tuple[str, ...] = ("energy", "pt", "eta", "phi"),
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Parameters
    ----------
    merged_tracksters_per_event : list[list[Trackster]]
        One list of (possibly merged) Reco tracksters for every event.
        Length must equal `score_tables`.
    score_tables : list[list[dict]]
        Output of `compute_reco_to_sim_scores(best_only=False)` for every event.
    score_threshold : float
        'High overlap' threshold (default 0.6).  A Reco passes if it exceeds
        this score for *more than one* Sim trackster.
    vars : which variables to accumulate.

    Returns
    -------
    out : dict
        {var: (values, passed)}  with NumPy arrays for each `var` in `vars`.
    """
    out = {v: ([], []) for v in vars}   # var -> (values_list, pass_list)

    for reco_list, rows in zip(merged_tracksters_per_event, score_tables):
        if not rows:
            continue

        # --------------------------------------------------------------
        # 1) For this event, count how many Sim's give score>thr to each Reco
        # --------------------------------------------------------------
        high_overlap_count = {}
        for r in rows:
            if r["score"] < score_threshold:
                high_overlap_count[r["reco_idx"]] = \
                    high_overlap_count.get(r["reco_idx"], 0) + 1

        # --------------------------------------------------------------
        # 2) Flag each Reco as passed (True) / not (False)
        # --------------------------------------------------------------
        passed_map = {idx: (cnt > 1) for idx, cnt in high_overlap_count.items()}

        # --------------------------------------------------------------
        # 3) Append variable values + pass flag to the global lists
        # --------------------------------------------------------------
        for idx, trk in enumerate(reco_list):
            passed_flag = passed_map.get(idx, False)

            value_dict = {
                "energy": trk.raw_energy,
                "pt":     getattr(trk, "raw_pt",
                         getattr(trk, "regressed_pt", np.nan)),
                "eta":    abs(trk.barycenter_eta),
                "phi":    trk.barycenter_phi,
            }

            for v in vars:
                out[v][0].append(value_dict[v])
                out[v][1].append(passed_flag)

    # convert to NumPy arrays
    for v in vars:
        values, ok = out[v]
        out[v] = (np.asarray(values, dtype="f8"),
                  np.asarray(ok,     dtype=bool))
    return out

def aggregate_for_fake(
    merged_tracksters_per_event: Sequence[Sequence[Trackster]],
    score_tables:                Sequence[List[Dict]],
    *,
    score_threshold: float = 0.6,
    vars: Tuple[str, ...] = ("energy", "pt", "eta", "phi"),
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Parameters
    ----------
    merged_tracksters_per_event : list[list[Trackster]]
        One list of (possibly merged) Reco tracksters for every event.
        Length must equal `score_tables`.
    score_tables : list[list[dict]]
        Output of `compute_reco_to_sim_scores(best_only=False)` for every event.
    score_threshold : float
        'High overlap' threshold (default 0.6).  A Reco passes if it exceeds
        this score for *more than one* Sim trackster.
    vars : which variables to accumulate.

    Returns
    -------
    out : dict
        {var: (values, passed)}  with NumPy arrays for each `var` in `vars`.
    """
    out = {v: ([], []) for v in vars}   # var -> (values_list, pass_list)

    for reco_list, rows in zip(merged_tracksters_per_event, score_tables):
        if not rows:
            continue

        # --------------------------------------------------------------
        # 1) For this event, count how many Sim's give score>thr to each Reco
        # --------------------------------------------------------------
        high_overlap_count = {}
        for r in rows:
            if r["score"] > score_threshold:
                high_overlap_count[r["reco_idx"]] = \
                    high_overlap_count.get(r["reco_idx"], 0) + 1

        # --------------------------------------------------------------
        # 2) Flag each Reco as passed (True) / not (False)
        # --------------------------------------------------------------
        passed_map = {idx: (cnt == 0) for idx, cnt in high_overlap_count.items()}

        # --------------------------------------------------------------
        # 3) Append variable values + pass flag to the global lists
        # --------------------------------------------------------------
        for idx, trk in enumerate(reco_list):
            passed_flag = passed_map.get(idx, False)

            value_dict = {
                "energy": trk.raw_energy,
                "pt":     getattr(trk, "raw_pt",
                         getattr(trk, "regressed_pt", np.nan)),
                "eta":    abs(trk.barycenter_eta),
                "phi":    trk.barycenter_phi,
            }

            for v in vars:
                out[v][0].append(value_dict[v])
                out[v][1].append(passed_flag)

    # convert to NumPy arrays
    for v in vars:
        values, ok = out[v]
        out[v] = (np.asarray(values, dtype="f8"),
                  np.asarray(ok,     dtype=bool))
    return out


# ------------------------------------------------------------
#  Plot a 3 × 4 grid:  (efficiency | merge | fake) × (E, pT, η, φ)
# ------------------------------------------------------------
def plot_all_metrics(
    data_efficiency: dict,
    data_merged: dict,
    data_fake: dict,
    *,
    out_file: str | None = None,
):
    """
    Parameters
    ----------
    data_efficiency, data_merged, data_fake : dict
        Output from `aggregate_for_efficiency`, `aggregate_for_merge`,
        `aggregate_for_fake` respectively.  Each maps
           {var: (values, passed)}
    out_file : str or None
        If given, save the figure (png/pdf/…).
    """
    # ----- configuration -------------------------------------------------
    vars_cfg = [
        ("energy", np.linspace(5, 600, 50),  "Energy [GeV]"),
        ("pt",     np.linspace(2, 150, 25),  r"$p_{T}$  [GeV]"),
        ("eta",    np.linspace(0, 3.2, 25),  r"|η|"),
        ("phi",    np.linspace(-np.pi, np.pi, 25),  r"φ"),
    ]

    metrics = [
        ("Efficiency", data_efficiency, {"fmt": "o", "label": "Eff"}),
        ("Merged Rate", data_merged,    {"fmt": "s", "label": "Merged"}),
        ("Fake Reco",   data_fake,      {"fmt": "D", "label": "Fake"}),
    ]

    # ----- figure layout -------------------------------------------------
    fig, axes = plt.subplots(
        nrows=len(metrics), ncols=len(vars_cfg),
        figsize=(20, 12), sharey="row"
    )

    for row_idx, (metric_title, data_dict, style) in enumerate(metrics):
        for col_idx, (var, bins, xlabel) in enumerate(vars_cfg):
            ax = axes[row_idx, col_idx]
            vals, passed = data_dict[var]

            plot_efficiency_mpl(
                vals,
                passed,
                bins,
                x_label=xlabel,
                y_label = metric_title,
                label=style["label"],
                fmt=style["fmt"],
                ax=ax,
            )

            if row_idx == 0:
                ax.set_title(xlabel)

            if col_idx == 0:
                ax.set_ylabel(metric_title)

    # neat layout & (optional) save
    plt.tight_layout()
    if out_file:
        fig.savefig(out_file)
    plt.show()





