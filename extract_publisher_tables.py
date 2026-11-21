"""Extract and normalize selected public PDF tables; preserve page-level provenance."""
import hashlib
import json
import re

import pdfplumber
from benchmark import ROOT

SOURCES = {
    "wmrn": ("West Midlands Radiotherapy Network", "Lung SABR Protocol", "v4.0",
             "https://westmidlandsradiotherapynetwork.nhs.uk/wp-content/uploads/2025/07/West-Midlands-RT-Network-SABR-lung-V4.0-RATIFIED-BY-EAG-100725.pdf"),
    "wfb": ("Wake Forest Baptist Comprehensive Cancer Center", "WFBCCC 62718 / NCT03867175",
            "2022-04-01", "https://cdn.clinicaltrials.gov/large-docs/75/NCT03867175/Prot_SAP_000.pdf"),
}
ORGAN_KEYS = {
    "BrachialPlex": "brachial plexus", "SpinalCanal": "spinal cord",
    "Heart+A_Pulm": "heart", "Bronchus_Prox": "bronchus", "Trachea": "trachea",
    "Lungs-GTV": "lung", "ChestWall": "chest wall", "GreatVes": "great vessels",
    "Oesophagus": "oesophagus", "Liver": "liver", "Skin": "skin",
    "Lung": "lung", "Esophagus": "oesophagus", "Heart": "heart",
    "Brachial Plexus": "brachial plexus", "Spinal Cord": "spinal cord",
}
TIERS = ((3, "optimal"), (3, "mandatory"), (5, "optimal"), (5, "mandatory"),
         (8, "optimal"), (8, "mandatory"))
WFB_CONDITION = "每次分割 <3 Gy；原表为建议约束"


def compact(value):
    return " ".join((value or "").split())


def origin(protocol, page, row_id, section, text, header_page=None):
    publisher, title, version, url = SOURCES[protocol]
    row = dict(id=row_id, protocol=protocol, publisher=publisher, title=title,
               version=version, page=page, section=section, url=f"{url}#page={page}", text=text)
    if header_page and header_page != page:
        row["header_url"] = f"{url}#page={header_page}"
    return row


def metric_key(text):
    value = compact(text).lower().replace(" ", "")
    core = value.split("(", 1)[0]
    if core in {"dmean", "mean"}:
        return "mean"
    if core in {"dmax", "maxdose"}:
        return value if value.startswith("dmax(") else "max"
    return re.sub(r"gy$", "", core)


def cell_value(raw):
    if raw == "-":
        return dict(comparator=None, value=None, unit=None, listed=False, footnote=None)
    m = re.fullmatch(r"([<>≤≥]?)\s*(\d+(?:\.\d+)?)\s*(Gy|%)?\s*(\*+)?", raw)
    if not m:
        raise ValueError(f"Unrecognized numeric cell: {raw}")
    return dict(comparator=m[1] or None, value=float(m[2]), unit=m[3],
                listed=True, footnote=m[4] or None)


# Reviewed mappings for the selected contour rows. Expected anchors verify each concept
# against the extracted row and prevent use if the PDF layout or wording changes.
CONTOURS = {
    "ChestWall": ("chest wall",
        ("2cm rind", "intercostal ribs, nerves and muscles", "excludes vertebral bodies, sternum, and skin",
         "lateral edge of the sternum", "lateral aspect of the vertebral body"),
        {"boundary":"前内侧至胸骨外缘；后内侧至椎体外侧",
         "components":"包括肋骨、肋间神经及肌肉", "excluded":"不包括椎体、胸骨和皮肤",
         "radial":"胸腔外侧半胸的 2 cm 环形层"}),
    "SpinalCanal": ("spinal cord",
        ("inner limits of the spinal canal", "bone windowing", "tip of the dens of the C2",
         "most caudal slice", "L5-S1 vertebral bodies"),
        {"boundary":"头侧到 C2 齿突尖端；尾侧到椎管最后可见的层面，通常在 L5–S1 水平",
         "description":"骨窗下按椎管内缘勾画"}),
    "Oesophagus": ("oesophagus",
        ("all muscle layers out to the fatty adventitia", "lower edge of the cricoid cartilage",
         "oesophageal junction"),
        {"boundary":"从环状软骨下缘到胃食管交界处",
         "components":"纵隔窗下包括全部肌层至脂肪外膜"}),
    "Skin": ("skin", ("5 mm inner rind of the external body contour",),
        {"description":"体表外轮廓向内取 5 mm 环形层"}),
    "Heart+A_Pulm": ("heart",
        ("include the pericardial sac", "cranial aspect of the pulmonary artery", "inferior vena cava should be excluded"),
        {"boundary":"头侧到肺动脉头侧缘；尾侧到左心室与膈肌相接的心尖",
         "components":"纵隔窗下包含心包囊", "excluded":"排除大血管（包括下腔静脉）；主支气管下方的肺动脉排除"}),
}


def wmrn_extract(pdf, rows, cells, contours):
    lung_source = "wmrn-p10-lungs-gtv-definition"
    for page in (9, 10):
        tables = [t for t in pdf.pages[page - 1].extract_tables()
                  if t and len(t[0]) == 2]
        assert len(tables) == 1
        for structure, note in tables[0]:
            if "Lungs - GTV" in (structure or "").splitlines():
                definition = compact(note)
                if "Lungs – GTV is lungs minus the motion adapted GTV" not in definition:
                    raise ValueError("Missing Lungs–GTV definition")
                rows.append(origin("wmrn", page, lung_source, "§6.2 Organs at Risk",
                                   f"{compact(structure)} | {definition}"))
            if structure not in CONTOURS:
                continue
            organ, expected, fields = CONTOURS[structure]
            en = compact(note)
            for match in expected:
                if match.casefold() not in en.casefold():
                    raise ValueError((structure, match, en))
            key = f"wmrn-p{page}-contour-{organ.replace(' ', '-')}"
            rows.append(origin("wmrn", page, key, "§6.2 Organs at Risk", f"{structure} | {en}"))
            contours.append(dict(source_id=key, organ=organ, **fields))

    tables = {}
    for page in (13, 14):
        t = [x for x in pdf.pages[page - 1].extract_tables() if len(x[0]) == 8]
        assert len(t) == 1
        tables[page] = t[0]
    h, tierline = tables[13][:2]
    assert h[2:8:2] == ["3 Fractions", "5 Fractions", "8 Fractions"]
    assert tierline[2:] == ["Optimal", "Mandatory"] * 3
    header = "Description | Constraint | 3 Fractions Optimal | 3 Fractions Mandatory | " \
             "5 Fractions Optimal | 5 Fractions Mandatory | 8 Fractions Optimal | 8 Fractions Mandatory"
    rows.append(origin("wmrn", 13, "wmrn-p13-dose-header", "§8.2 OAR Dose Constraints", header))
    page14 = compact(pdf.pages[13].extract_text())
    note_spinal = re.search(r"\*Whilst the dose constraint is strictly to the Spinal Canal,.*?(?=\*\* If)",
                            page14).group().strip()
    note_heart = re.search(r"\*\* If the eight-fraction cardiac mandatory constraint.*?(?=Page 14 of 17)",
                           page14).group().strip()
    for label, note in (("spinal", note_spinal), ("heart", note_heart)):
        rows.append(origin("wmrn", 14, f"wmrn-p14-note-{label}", "§8.2 Table footnotes", note))
    for page, table in tables.items():
        prior = None
        for i, row in enumerate(table[2:] if page == 13 else table, 1):
            prior = compact(row[0]) or prior
            assert prior and len(row) == 8
            metric = compact(row[1])
            source_id = f"wmrn-p{page}-oar-{i:02d}"
            rows.append(origin("wmrn", page, source_id, "§8.2 OAR Dose Constraints",
                               header + "\n" + " | ".join([prior, metric, *[compact(v) for v in row[2:]]]), 13))
            for j, ((fractions, tier), v) in enumerate(zip(TIERS, row[2:]), 1):
                raw = compact(v)
                denom = "双肺减运动适配 GTV" if prior == "Lungs-GTV" else None
                note_id = ("wmrn-p14-note-spinal" if prior == "SpinalCanal*" else
                           "wmrn-p14-note-heart" if raw.endswith("**") else
                           lung_source if denom else None)
                cells.append(dict(id=f"{source_id}-C{j}", source_id=source_id,
                                  organ=ORGAN_KEYS[prior.rstrip("*")], metric=metric_key(metric),
                                  organ_en=prior, metric_en=metric, fractions=fractions, tier=tier,
                                  condition=f"{fractions} 次分割", denominator=denom, raw=raw,
                                  note_source_id=note_id,
                                  **cell_value(raw)))


def wfb_extract(pdf, rows, cells):
    page = 22
    tables = pdf.pages[page - 1].extract_tables()
    matches = [t for t in tables if len(t[0]) == 3 and t[0][0]
               and t[0][0].startswith("Recommended dose constraints")]
    assert len(matches) == 1
    table = matches[0]
    heading = compact(table[0][0])
    assert heading == "Recommended dose constraints for fractionated radiation < 3 Gy per fraction:"
    notes = compact(pdf.pages[20].extract_text())
    note = "Meeting these constraints are strongly recommended, but not required for inclusion on the protocol."
    assert note in notes
    rows.append(origin("wfb", 21, "wfb-p21-oar-policy", "§7.6 Dosimetric Guidelines", note))
    inventory = [heading]
    prior = None
    for i, (organ, label, v) in enumerate(table[1:], 1):
        prior = compact(organ) or prior
        name, raw = compact(label), compact(v)
        key = f"wfb-p22-oar-{i:02d}"
        inventory.append(f"{prior} | {name} | {raw}")
        rows.append(origin("wfb", page, key, "§7.6 Recommended conventional constraints", 
                           f"{heading}\n{prior} | {name} | {raw}"))
        denom = ("双肺减 PTV" if "minus PTV" in name else
                 "双肺减 GTV" if "minus GTV" in name else None)
        cells.append(dict(id=f"{key}-C1", source_id=key, organ=ORGAN_KEYS[prior], organ_en=prior,
                          metric=metric_key(name), metric_en=name, fractions=None, tier="recommended",
                          condition=WFB_CONDITION, denominator=denom, raw=raw,
                          note_source_id="wfb-p21-oar-policy", **cell_value(raw)))
    rows.append(origin("wfb", 22, "wfb-p22-oar-inventory", "§7.6 Recommended conventional constraints",
                       "\n".join(inventory)))


def build():
    root = ROOT / "data"
    rows, cells, contours, manifest = [], [], [], []
    for key, (publisher, title, version, url) in SOURCES.items():
        path = root / "raw" / f"{key}.pdf"
        with pdfplumber.open(path) as pdf:
            if key == "wmrn":
                wmrn_extract(pdf, rows, cells, contours)
            else:
                wfb_extract(pdf, rows, cells)
        manifest.append(dict(protocol=key, publisher=publisher, title=title, version=version,
                             url=url, path=f"raw/{key}.pdf",
                             sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    data = dict(sources=manifest, rows=rows, cells=cells, contours=contours)
    (root / "publisher_tables.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(cells)} dose cells ({sum(c['listed'] for c in cells)} listed), {len(contours)} contour rows")


if __name__ == "__main__":
    build()

