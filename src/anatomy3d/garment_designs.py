"""Basic garments as they are made, seen from every side: what a person
would see on each from the front, the side and the back. The photo shows the
front; the rest is how the garment is cut and sewn, the same on anyone, so
the builder can put it on the sides and back the camera doesn't see
(body_first._garment_views).

Sizes are the garment's own, in metres on the person (tailoring's usual
proportions), scaled to the body the photo gives; a figure shows them up
(SHOW) the way it shows the clothes' layers.

Each garment:
  views     what is on it from the front, the side and the back (words, for
            people and for the reports)
  collar    what is round the neck at the back and sides:
              ("notched", h)  a jacket's collar standing h up the back of the neck
              ("shirt", h)    a shirt's collar band, h up the neck
              ("rib", h)      a knitted neckband h deep, sitting at the neck's base
              ("hood", h)     a hood lying folded at the back of the neck, h deep
              None            an open neckline
  seams     the seams that show off the front: "side" (under each arm, down
            the side), "back" (down the middle of the back), "shoulder"
            (along the top of each shoulder, neck to shoulder point), "yoke"
            (across the upper back), "armhole" (round where the sleeve is set)
  hem       where it ends: "hip", "waist", "thigh", "knee", "ankle"; a step
            round the body there where the body reaches that far
  bands     knitted bands: "hem", "cuff"
  waistband a band round the waist (trousers, skirts) h deep
  outer     how far out it lies (the outermost one on a part shows its seams)
"""

SHOW = 1.6  # sizes shown up this much on a figure (as body_first._FIT_SHOW)
SEAM_DEPTH = 0.003  # a seam's groove on a figure, m (shown up)
SEAM_WIDTH = 0.007  # ... and its width
COLLAR_THICK = {"notched": 0.008, "shirt": 0.004, "rib": 0.004, "hood": 0.03}  # each collar's own thickness
SHIRT_ABOVE_JACKET = 0.012  # a shirt's collar shows this far above a jacket's at the back of the neck
HEM_STEP = 0.003  # a hem's step

DESIGNS = {
    "suit": {
        "views": {"front": "lapels along the opening, buttons, breast pocket, hip pockets, sleeves to the wrist with the shirt's cuff showing",
                  "side": "side seam under the arm, sleeve set into the armhole, the collar standing behind the neck",
                  "back": "collar round the back of the neck with the shirt's collar showing above it, a seam down the middle, "
                          "shoulder seams, often one or two vents at the hem"},
        "collar": ("notched", 0.035), "seams": ("side", "back", "shoulder", "armhole"), "hem": "hip", "outer": 3},
    "jacket": {
        "views": {"front": "opening with lapels or a zip, pockets", "side": "side seam, set-in sleeve",
                  "back": "collar round the neck, a seam down the middle, shoulder seams"},
        "collar": ("notched", 0.035), "seams": ("side", "back", "shoulder", "armhole"), "hem": "hip", "outer": 3},
    "coat": {
        "views": {"front": "wide lapels or a buttoned front, pockets", "side": "side seam, set-in sleeve, long skirt of the coat",
                  "back": "collar standing high round the neck, a seam down the middle, a vent"},
        "collar": ("notched", 0.05), "seams": ("side", "back", "shoulder", "armhole"), "hem": "knee", "outer": 4},
    "shirt": {
        "views": {"front": "collar points, a button placket down the middle, cuffs", "side": "side seam, sleeve",
                  "back": "collar band round the neck, a yoke seam across the shoulders"},
        "collar": ("shirt", 0.03), "seams": ("side", "shoulder", "yoke", "armhole"), "hem": "hip", "outer": 1},
    "tshirt": {
        "views": {"front": "round or V neck with a rib band, short sleeves", "side": "side seam, short sleeve's hem",
                  "back": "the neck band higher than at the front, shoulder seams"},
        "collar": ("rib", 0.015), "seams": ("side", "shoulder"), "hem": "hip", "outer": 1},
    "sweater": {
        "views": {"front": "rib neckband, rib cuffs and hem", "side": "side seam", "back": "the neck band round the back"},
        "collar": ("rib", 0.025), "seams": ("side", "shoulder"), "hem": "hip", "bands": ("hem", "cuff"), "outer": 2},
    "hoodie": {
        "views": {"front": "drawstrings, a front pocket, rib cuffs and hem", "side": "side seam, the hood behind the neck",
                  "back": "the hood lying folded over the upper back"},
        "collar": ("hood", 0.14), "seams": ("side", "shoulder"), "hem": "hip", "bands": ("hem", "cuff"), "outer": 2.5},
    "dress": {
        "views": {"front": "neckline, bodice, skirt", "side": "side seam", "back": "a seam or zip down the middle, the neckline higher"},
        "collar": None, "seams": ("side", "back"), "hem": "knee", "outer": 1},
    "kimono": {
        "views": {"front": "crossed collar band left over right, the obi at the waist", "side": "wide sleeves",
                  "back": "the collar band standing off the back of the neck, the obi's knot"},
        "collar": ("rib", 0.04), "seams": ("back",), "hem": "ankle", "outer": 1},
    "trousers": {
        "views": {"front": "waistband, fly, creases", "side": "side seam, pocket", "back": "waistband with belt loops, back pockets"},
        "waistband": 0.04, "seams": ("side",), "outer": 1},
    "jeans": {
        "views": {"front": "waistband, fly, front pockets", "side": "side seam", "back": "yoke, back pockets"},
        "waistband": 0.04, "seams": ("side",), "outer": 1},
    "skirt": {
        "views": {"front": "waistband", "side": "side seam", "back": "a zip down the middle"},
        "waistband": 0.03, "seams": ("side",), "outer": 1.5},
    "tie": {
        "views": {"front": "knot under the collar, blade down the shirt's front", "side": "lies on the shirt",
                  "back": "goes under the shirt's collar round the neck: hidden"},
        "outer": 2.2},
}

# the garments that go round the torso, outermost first when several are worn
TORSO = ("coat", "suit", "jacket", "hoodie", "sweater", "kimono", "dress", "shirt", "tshirt")


def worn_designs(found) -> dict:
    """What the builder puts round the body for the garments read: the
    outermost torso garment's seams and hem, every worn collar (a jacket's
    and the shirt's under it), waistbands. {"outer": name, "collars":
    [(name, kind, height)], "seams": (...), "hem": ..., "waistband": h,
    "views": {name: views}}."""
    found = set(found or ())
    torso = [g for g in TORSO if g in found]
    out = {"outer": torso[0] if torso else None, "collars": [], "seams": (), "hem": None, "waistband": None, "views": {}}
    if torso:
        d = DESIGNS[torso[0]]
        out["seams"] = d.get("seams", ())
        out["hem"] = d.get("hem")
    for g in torso:
        c = DESIGNS[g].get("collar")
        if c:
            out["collars"].append((g, c[0], c[1]))
    for g in ("trousers", "jeans", "skirt"):
        if g in found and DESIGNS[g].get("waistband"):
            out["waistband"] = DESIGNS[g]["waistband"]
    out["views"] = {g: DESIGNS[g]["views"] for g in found if g in DESIGNS}
    return out
