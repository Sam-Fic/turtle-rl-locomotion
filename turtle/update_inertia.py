import xml.etree.ElementTree as ET

XML_PATH = "turtle_position.xml"
OUT_PATH = "turtle_position.xml"

NEW_MASSES = {
    "base": 2.57,
    "link1": 0.13,
    "link2": 0.17,
    "link3": 0.10,
    "link4": 0.13,
    "link5": 0.17,
    "link6": 0.10,
    "link7": 0.125,
    "link8": 0.09,
    "link9": 0.125,
    "link10": 0.09,
}

tree = ET.parse(XML_PATH)
root = tree.getroot()

for body in root.iter("body"):
    name = body.get("name")
    if name not in NEW_MASSES:
        continue

    inertial = body.find("inertial")
    if inertial is None:
        print(f"  [skip] {name}: no <inertial> element found")
        continue

    old_mass = float(inertial.get("mass"))
    new_mass = NEW_MASSES[name]
    scale = new_mass / old_mass

    old_inertia = list(map(float, inertial.get("diaginertia").split()))
    new_inertia = [v * scale for v in old_inertia]

    inertial.set("mass", f"{new_mass}")
    inertial.set("diaginertia", f"{new_inertia[0]:.10e} {new_inertia[1]:.10e} {new_inertia[2]:.10e}")

    print(f"  [ok] {name}: {old_mass} -> {new_mass} kg  (scale={scale:.4f})")
    print(f"       diaginertia: {old_inertia} -> {[f'{v:.6e}' for v in new_inertia]}")

tree.write(OUT_PATH, encoding="utf-8", xml_declaration=True)
print(f"\nDone! Updated file written to {OUT_PATH}")