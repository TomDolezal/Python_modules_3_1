"""
cell_model_io
==============

Loads a battery cell model from the XLSX format described by the
B&ESS-CTU group, and converts it into the dict format expected by
_pg_helpers.make_sim_cell_model / profile_generation.py:

    {'SOC': ..., 'TEMP': ..., 'Q': ..., 'OCV0': ..., 'OCVrel': ...,
     'R0': ..., 'eta': ...}

Only the sheets needed for the OCV + R0 equivalent-circuit model are
read: "Meta" (for TEMP_C and optional datasheet fields), "OCV", "R0",
and "CapacityEta". The "R1"/"R2"/"C1"/"C2" sheets (which would imply a
1RC/2RC model) are intentionally NOT read -- profile_generation.py only
implements the simple OCV+R0 model, so pulling in RC-pair data would
just be unused overhead. If the model is extended later, those sheets
can be added back in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import openpyxl

from _pg_helpers import apply_default_datasheet


REQUIRED_SHEETS = ("Meta", "OCV", "R0", "CapacityEta")

# Optional Meta-sheet fields that, if present, are pulled out into a
# ready-to-use `datasheet` dict for profile_generation (rather than left
# sitting in the generic datasheet_meta bag with weight_g, type, etc.).
DATASHEET_FIELDS = ("max_current_charge_A", "max_current_discharge_A")


@dataclass
class CellModelFromExcel:
    """Everything loaded from the workbook.

    ``cell_model`` is ready to hand straight to make_sim_cell_model /
    profile_generation.

    ``datasheet`` is ready to hand straight to profile_generation's
    `datasheet` argument: it holds max_current_charge_A /
    max_current_discharge_A if those optional rows were present on the
    Meta sheet, with the usual profile_generation default (unconstrained,
    i.e. infinity) applied to whichever one is missing.

    ``datasheet_meta`` holds every OTHER optional datasheet field found
    on the Meta sheet (voltage_V, capacity_Ah, weight_g, type, ...) for
    reference -- these aren't consumed by profile_generation at all.
    """
    cell_model: dict
    datasheet: dict = field(default_factory=dict)
    datasheet_meta: dict = field(default_factory=dict)
    source_path: str = ""


def _to_float(value) -> float:
    """Defensively parse a cell value to float, tolerating comma-decimal
    strings (e.g. a value hand-typed in a Czech-locale Excel that ended
    up stored as text instead of a number)."""
    if value is None:
        raise ValueError("Empty cell where a numeric value was expected.")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip().replace(",", ".")
        try:
            return float(s)
        except ValueError:
            raise ValueError(f"Could not parse '{value}' as a number.")
    raise ValueError(f"Unexpected cell type for numeric value: {type(value)}")


def _read_meta_sheet(ws) -> tuple[np.ndarray, dict]:
    """Reads the Meta sheet: the required TEMP_C row, plus any optional
    label/value datasheet rows below it.
    """
    rows = list(ws.iter_rows(values_only=True))

    temp_c = None
    datasheet_meta = {}

    for row in rows:
        if row[0] is None:
            continue
        label = str(row[0]).strip()
        if label.upper() == "TEMP_C":
            values = [v for v in row[1:] if v is not None]
            if not values:
                raise ValueError("Meta sheet: TEMP_C row has no temperature values.")
            temp_c = np.array([_to_float(v) for v in values])
        elif label == "Var1":
            continue  # header row
        else:
            # optional datasheet field: label in col A, value in col B
            if len(row) > 1 and row[1] is not None:
                val = row[1]
                # keep 'type' (and any other non-numeric field) as string as-is
                try:
                    val = _to_float(val)
                except ValueError:
                    val = str(val).strip()
                datasheet_meta[label] = val

    if temp_c is None:
        raise ValueError(
            "Meta sheet is missing the required TEMP_C row. "
            "Import cannot continue without a temperature grid."
        )

    return temp_c, datasheet_meta


def _read_ocv_sheet(ws) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = list(ws.iter_rows(values_only=True))
    header = [str(h).strip() if h is not None else "" for h in rows[0]]

    required = {"SOC", "OCV0_V", "OCVrel_V"}
    if not required.issubset(set(header)):
        raise ValueError(
            f"OCV sheet must contain columns {sorted(required)}, "
            f"found {header}."
        )

    idx = {name: header.index(name) for name in required}
    data_rows = [r for r in rows[1:] if r[idx["SOC"]] is not None]

    soc = np.array([_to_float(r[idx["SOC"]]) for r in data_rows])
    ocv0 = np.array([_to_float(r[idx["OCV0_V"]]) for r in data_rows])
    ocvrel = np.array([_to_float(r[idx["OCVrel_V"]]) for r in data_rows])

    if not (np.all(soc >= 0) and np.all(soc <= 1)):
        raise ValueError("OCV sheet: SOC values must be in the interval [0, 1].")
    if not np.all(np.diff(soc) > 0):
        raise ValueError("OCV sheet: SOC values must be strictly increasing.")

    return soc, ocv0, ocvrel


def _read_r0_sheet(ws, expected_soc: np.ndarray, n_temp: int) -> np.ndarray:
    rows = list(ws.iter_rows(values_only=True))
    header = [str(h).strip() if h is not None else "" for h in rows[0]]

    if not header or header[0] != "SOC":
        raise ValueError(f"R0 sheet: first column must be 'SOC', found header {header}.")

    r0_cols = [h for h in header[1:] if h]  # drop trailing empty/None columns
    if len(r0_cols) != n_temp:
        raise ValueError(
            f"R0 sheet has {len(r0_cols)} temperature columns "
            f"({r0_cols}), but Meta.TEMP_C defines {n_temp} temperatures. "
            f"These must match."
        )

    data_rows = [r for r in rows[1:] if r[0] is not None]
    if len(data_rows) != len(expected_soc):
        raise ValueError(
            f"R0 sheet has {len(data_rows)} SOC rows, but the OCV sheet "
            f"defines {len(expected_soc)} SOC points. These must match."
        )

    soc_r0 = np.array([_to_float(r[0]) for r in data_rows])
    if not np.allclose(soc_r0, expected_soc):
        raise ValueError(
            "R0 sheet: SOC column does not match the SOC column on the OCV sheet."
        )

    r0_matrix = np.array([
        [_to_float(r[j + 1]) for j in range(len(r0_cols))]
        for r in data_rows
    ])
    return r0_matrix  # shape (len(SOC), n_temp), same orientation ndgrid(SOC,TEMP) expects


def _read_capacity_eta_sheet(ws, expected_temp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = list(ws.iter_rows(values_only=True))
    header = [str(h).strip() if h is not None else "" for h in rows[0]]

    required = {"TEMP_C", "Q_Ah", "eta"}
    if not required.issubset(set(header)):
        raise ValueError(
            f"CapacityEta sheet must contain columns {sorted(required)}, "
            f"found {header}."
        )
    idx = {name: header.index(name) for name in required}
    data_rows = [r for r in rows[1:] if r[idx['TEMP_C']] is not None]

    temp_c = np.array([_to_float(r[idx["TEMP_C"]]) for r in data_rows])
    q_ah = np.array([_to_float(r[idx["Q_Ah"]]) for r in data_rows])
    eta = np.array([_to_float(r[idx["eta"]]) for r in data_rows])

    if not np.allclose(np.sort(temp_c), np.sort(expected_temp)):
        raise ValueError(
            "CapacityEta sheet: TEMP_C values do not match Meta.TEMP_C "
            f"(got {temp_c.tolist()}, expected {expected_temp.tolist()})."
        )

    # Reorder to match Meta.TEMP_C order exactly, in case the sheet listed
    # temperatures in a different order than Meta did.
    order = np.array([np.where(temp_c == t)[0][0] for t in expected_temp])
    return q_ah[order], eta[order]


def load_cell_model_xlsx(path: str) -> CellModelFromExcel:
    """Loads a cell model XLSX file and returns a CellModelFromExcel with
    a ready-to-use ``cell_model`` dict for make_sim_cell_model /
    profile_generation.

    Raises ValueError with a descriptive message if any required sheet
    or field is missing or inconsistent.
    """
    wb = openpyxl.load_workbook(path, data_only=True)

    missing = [s for s in REQUIRED_SHEETS if s not in wb.sheetnames]
    if missing:
        raise ValueError(
            f"Workbook is missing required sheet(s): {missing}. "
            f"Found sheets: {wb.sheetnames}."
        )

    temp_c, datasheet_meta = _read_meta_sheet(wb["Meta"])
    soc, ocv0, ocvrel = _read_ocv_sheet(wb["OCV"])
    r0_matrix = _read_r0_sheet(wb["R0"], expected_soc=soc, n_temp=len(temp_c))
    q_ah, eta = _read_capacity_eta_sheet(wb["CapacityEta"], expected_temp=temp_c)

    cell_model = {
        "SOC": soc,          # normalized 0..1, matches make_sim_cell_model's expectation
        "TEMP": temp_c,      # degC
        "Q": q_ah,           # Ah, aligned with TEMP order
        "OCV0": ocv0,        # V
        "OCVrel": ocvrel,    # V/degC
        "R0": r0_matrix,     # ohm, shape (len(SOC), len(TEMP))
        "eta": eta,          # aligned with TEMP order
    }

    # Pull the recognized current-limit fields out of the generic
    # datasheet_meta bag into their own dict, defaulting whichever one is
    # absent to unconstrained (inf), matching profile_generation's own
    # apply_default_datasheet behaviour.
    datasheet_raw = {k: datasheet_meta.pop(k) for k in DATASHEET_FIELDS if k in datasheet_meta}
    datasheet = apply_default_datasheet(datasheet_raw)

    return CellModelFromExcel(
        cell_model=cell_model,
        datasheet=datasheet,
        datasheet_meta=datasheet_meta,
        source_path=path,
    )
