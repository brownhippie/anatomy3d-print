"""Everything the builder is being taught, as a checklist the server shows
(GET /api/study/checklist, the Study page): each part of the face, the
head, what's worn, how things overlap, and all 206 bones. For each item:

  read     the photo is read for it (detected, measured)
  sculpt   the builder shapes it from that reading
  knobs    its numbers are in knobs.py, for studies to tune
  critic   a score measures it (study.py), so a study can tell better from worse

A value is True (done), False (not yet) or "n/a" with the reason in `note`.
Statuses here are kept true to the code: an item is ticked when the code
that does it is in. Whether a study has tuned it ("studied") is worked out
by the server from its studies.
"""
from dataclasses import dataclass, field
from typing import Union

Flag = Union[bool, str]


@dataclass
class Item:
    id: str
    group: str
    name: str
    read: Flag
    sculpt: Flag
    knobs: list = field(default_factory=list)  # knobs.py keys (module._NAME)
    critic: Flag = False
    note: str = ""
    rig: str = ""  # bones: where it lives in the reference body's rig

    def done(self) -> bool:
        flags = all(v is True or v == "n/a" for v in (self.read, self.sculpt, self.critic))
        # a bone shaped by the reference body has no numbers of its own to tune
        return flags and (bool(self.knobs) or self.sculpt == "n/a" or self.group.startswith("Bones"))


SR = "silhouette_relief."
BR = "body_reference."
NA = "n/a"


def _face() -> list:
    k = lambda *names: [SR + n for n in names]  # noqa: E731
    return [
        Item("face.shape", "Face", "Face shape: jaw, cheeks, chin, forehead", True, True,
             k("_FACE_RELIEF_FADE_FRAC", "_FACE_RELIEF_SMOOTH_FRAC", "_FACE_RELIEF_WITH_DEPTH"), True,
             "468 face landmarks; the face's points found on the model score it, the jaw's on their own"),
        Item("face.nose", "Face", "Nose: bridge, tip, wings", True, True, k("_FACE_RELIEF_WITH_DEPTH"), True,
             "landmark shape; the nose's own points score it"),
        Item("face.nostrils", "Face", "Nostrils", True, True, k("_NOSTRIL_DEPTH"), True, "dark blobs under the tip; the nose's points score it"),
        Item("face.eyes", "Face", "Eyes: lids, eyeballs set behind them", True, True, k("_EYE_INSET", "_LID_CREASE"), True,
             "the eyes' own points score it"),
        Item("face.eyes_open", "Face", "Eyes open or closed", True, True, k("_LID_LINE"), True,
             "lid gap and blink scores; 'as open as the photo' scores it"),
        Item("face.irises", "Face", "Irises", True, True, k("_IRIS_DEPTH"), True, "from the iris landmarks; the eyes' points score it"),
        Item("face.brows", "Face", "Eyebrows (the hair)", True, True, k("_BROW_HEIGHT"), True,
             "the dark hair between the brow lines; the brows' points score it"),
        Item("face.lips", "Face", "Lips and their edge", True, True, k("_LIP_RIM", "_MOUTH_LINE"), True, "the mouth's points score it"),
        Item("face.mouth_open", "Face", "Mouth open (its opening)", True, True, k("_MOUTH_DEPTH"), True,
             "inner-lip gap and jawOpen; 'as open as the photo' scores it"),
        Item("face.teeth", "Face", "Teeth", True, True, k("_TEETH_INSET"), True, "bright, colourless inside the open mouth"),
        Item("face.tongue", "Face", "Tongue", True, True, k("_TONGUE_INSET"), True, "a red mass low in the mouth, below the teeth"),
        Item("face.scars", "Face", "Scars", True, True, k("_SCAR_DEPTH"), True,
             "long thin clear lines on visible skin, wrinkles and smile lines excluded; none on our test faces"),
        Item("face.beard", "Face", "Beard", True, True, k("_BEARD_THICK"), True, "hair on the lower face; bumps and the jaw's points score it"),
        Item("face.moustache", "Face", "Moustache", True, True, k("_BEARD_THICK"), True, "hair over the lip"),
        Item("face.stubble", "Face", "Stubble", True, True, k("_STUBBLE_DEPTH"), True, "the jaw darker and rougher than the cheeks"),
        Item("face.creases", "Face", "Creases: smile lines, forehead lines", True, True, k("_REF_RELIEF_MAX"), True,
             "photo depth where the face faces the camera"),
        Item("face.ears", "Face", "Ears: size, place, swept back, under hair", True, True,
             [SR + "_REF_EAR_HAIR", SR + "_REF_EAR_SINK", SR + "_REF_EAR_THICK", BR + "_EAR_DROP", BR + "_EAR_KINK", BR + "_EAR_BEND",
              BR + "_EAR_UNDER_HAIR"], True, "3D ears measured from the photo; the ears' reach beside the head scores them"),
    ]


def _head() -> list:
    return [
        Item("hair.volume", "Hair", "Hair volume round the skull, buns, long hair", True, True,
             [SR + k for k in ("_HAIR_SHELL_MAX", "_HAIR_SMOOTH", "_HAIR_THIN", "_HAIR_FACE_MEET", "_HAIR_LAYER", "_HAIR_HANG", "_REF_HAIR")],
             True, "bumps score its sides and back"),
        Item("hair.back_volume", "Hair", "Hair volume round the back, from what the front shows at the sides", True, True,
             [SR + "_HAIR_BACK_FROM_SIDES", SR + "_HAIR_SHELL_MAX"], True,
             "the side reach past the skull, row by row, carried round the back; bumps score the back"),
        Item("hair.pattern", "Hair", "Hair pattern: strands, waves, curls, coils", True, True,
             [SR + k for k in ("_HAIR_STRAND_W", "_HAIR_STRAND_LEN", "_HAIR_STRAND_DEPTH", "_HAIR_WAVE_DEPTH", "_HAIR_WAVE_LEN",
                               "_HAIR_CURL", "_HAIR_CURL_DEPTH", "_HAIR_COIL", "_HAIR_COIL_DEPTH")], True,
             "read from the photo's own strands; hair flow scores it"),
        Item("hair.hairline", "Hair", "Hairline over the forehead", True, True, [SR + "_HAIRLINE_EDGE", SR + "_HAIR_FEATHER", SR + "_HAIR_FRINGE"],
             True, "read; the pattern ends at it; the overlaps score checks hair standing over the forehead"),
        Item("hair.nape", "Hair", "Nape and behind the ears: hairline to collar", True, True,
             [SR + "_HAIR_NAPE_EDGE", SR + "_HAIR_NECK_BEHIND", SR + "_C7", SR + "_REF_BACK_SMOOTH"], True,
             "smooth with its curves; built from what covers it (the hair's reach round the back, the collar's rise, a bare nape's C7); "
             "the nape's bumps score it"),
        Item("hair.behind_ear", "Hair", "The gap between the hair and the back of the ear", True, True,
             [SR + "_REF_EAR_HAIR", BR + "_EAR_UNDER_HAIR", SR + "_REF_EAR_SINK"], True,
             "ears under hair aren't built; a seen ear stands off the head with its root on the skull; the ears' reach scores it"),
        Item("hair.parting", "Hair", "Parting", False, False),
        Item("hair.braids", "Hair", "Braids, locs", False, False),
    ]


def _worn() -> list:
    return [
        Item("wear.cap", "Headwear", "Cap or hat, its peak", True, True, [SR + "_HAT_PEAK"], True, "the overlaps score checks it over hair"),
        Item("wear.mask", "Headwear", "Face mask", True, True, [SR + "_MASK_SPAN"], True, "the overlaps score checks it over the face"),
        Item("wear.glasses", "Headwear", "Glasses, sunglasses", True, True, [SR + "_GLASSES_SPAN", SR + "_GLASSES_FRAME"], True,
             "the overlaps score checks them over the face"),
        Item("wear.earrings", "Headwear", "Earrings", True, True, [SR + "_EARRING_PROUD"], True,
             "an accessory below the ears beside the face; the overlaps score checks it over hair and neck"),
        Item("neck.neck", "Neck", "Neck", True, True, [SR + "_REF_BACK_SMOOTH"], True, "from the reference body's neck; the nape's bumps score it"),
        Item("neck.collar", "Neck", "Collar, round the back too", True, True,
             [SR + "_NECK_COLLAR", SR + "_NECK_COLLAR_BELOW", SR + "_NECK_COLLAR_RISE"], True, "the overlaps score checks it round the neck"),
        Item("neck.scarf", "Neck", "Scarf, roll neck", True, True, [SR + "_NECK_WEAR_MAX"], True, "the overlaps score checks it over clothes"),
        Item("neck.necklace", "Neck", "Necklace", False, True, [SR + "_NECK_LACE"], False, "shaped from the neck's band; not named on the photo yet"),
    ]


_GARMENTS = [
    ("shirt", "Shirt, blouse"), ("tshirt", "T-shirt, top"), ("jacket", "Jacket, blazer, suit"), ("coat", "Coat"),
    ("sweater", "Sweater, cardigan"), ("hoodie", "Hoodie (and its hood)"), ("dress", "Dress"), ("skirt", "Skirt"),
    ("trousers", "Trousers"), ("jeans", "Jeans"), ("shorts", "Shorts"), ("kimono", "Kimono, robe, wrap (and its sash)"),
    ("tie", "Tie, bow tie"), ("belt", "Belt"), ("lapels", "Lapels"), ("sleeves", "Sleeves and cuffs"), ("hems", "Hems"),
    ("buttons", "Buttons"), ("pockets", "Pockets"), ("folds", "Folds and drape"), ("shoes", "Shoes"), ("gloves", "Gloves"),
    ("bag", "Bag, straps"), ("watch", "Watch, bracelet"),
]


_STEP = [SR + "_CLOTH_LAYER", SR + "_CLOTH_STEP_SPAN"]
# garments the overlaps score covers (a sure pair with something it lies over or under)
_SCORED = {"shirt", "tshirt", "jacket", "suit", "coat", "sweater", "hoodie", "dress", "skirt", "trousers", "jeans", "shorts",
           "kimono", "tie", "belt", "lapels", "hems"}


def _clothes() -> list:
    out = []
    named = "named on the photo (CLIPSeg, where the body segmenter sees clothes)"
    for key, name in _GARMENTS:
        if key == "folds":
            out.append(Item("clothes.folds", "Clothes", name, True, True, [SR + "_REF_RELIEF_MAX", SR + "_REF_BACK_SMOOTH", SR + "_REF_SIDE_SMOOTH"],
                            True, "photo depth on surfaces facing the camera; bumps score the sides and back"))
        elif key in ("buttons", "pockets"):
            out.append(Item(f"clothes.{key}", "Clothes", name, False, False))
        elif key in ("lapels", "hems"):
            out.append(Item(f"clothes.{key}", "Clothes", name, True, True, _STEP, True,
                            "the jacket's edge stepping over the shirt / trousers; the overlaps score checks it"))
        elif key == "sleeves":
            out.append(Item("clothes.sleeves", "Clothes", name, True, True, _STEP, False,
                            "the sleeve stepping over the wrist; not scored (a hand can lie over clothes too)"))
        else:
            knobs_ = _STEP + ([SR + "_TIE_THICK"] if key == "tie" else []) + ([SR + "_BELT_THICK"] if key == "belt" else [])
            extra = [SR + "_REF_DRAPE", SR + "_REF_HANG"] if key in ("shirt", "jacket", "suit") else []
            scored = key in _SCORED
            out.append(Item(f"clothes.{key}", "Clothes", name, True, True, knobs_ + extra, scored,
                            named + "; stepped over what it lies on" + ("; the overlaps score checks it" if scored else
                                                                         "; nothing sure to check it against yet")))
    return out


def _overlaps() -> list:
    amb = "a front photo can't say which is in front here (both happen): not scored"
    return [
        Item("over.hair_ears", "Overlaps", "Hair over the ears", True, True, [BR + "_EAR_UNDER_HAIR", SR + "_REF_EAR_HAIR"], False, amb),
        Item("over.hair_face", "Overlaps", "Hair over the forehead and cheeks", True, True, [SR + "_HAIR_FRINGE", SR + "_HAIR_FACE_MEET"], False, amb),
        Item("over.hair_neck", "Overlaps", "Hair beside and behind the neck", True, True, [SR + "_HAIR_NECK_BEHIND", SR + "_HAIR_HANG"], False, amb),
        Item("over.hair_clothes", "Overlaps", "Hair over shoulders and clothes", True, True, [SR + "_HAIR_LAYER"], False, amb),
        Item("over.hat_hair", "Overlaps", "Cap or hat over hair", True, True, [SR + "_HAT_PEAK"], True, "the overlaps score checks it"),
        Item("over.glasses_face", "Overlaps", "Glasses over the face", True, True, [SR + "_GLASSES_SPAN", SR + "_GLASSES_FRAME"], True,
             "the overlaps score checks it"),
        Item("over.mask_face", "Overlaps", "Mask over the face", True, True, [SR + "_MASK_SPAN"], True, "the overlaps score checks it"),
        Item("over.collar_neck", "Overlaps", "Collar round the neck", True, True, [SR + "_NECK_COLLAR"] + _STEP, True,
             "garments in the band under the chin over the neck's skin"),
        Item("over.scarf_clothes", "Overlaps", "Scarf over clothes", True, True, [SR + "_NECK_WEAR_MAX"] + _STEP, True),
        Item("over.necklace", "Overlaps", "Necklace over skin or clothes", True, True, [SR + "_NECK_LACE"], False, "not named on the photo yet"),
        Item("over.jacket_shirt", "Overlaps", "Jacket over shirt", True, True, _STEP, True),
        Item("over.tie_shirt", "Overlaps", "Tie over shirt", True, True, _STEP + [SR + "_TIE_THICK"], True),
        Item("over.sweater_shirt", "Overlaps", "Sweater over shirt (collar out)", True, True, _STEP, True),
        Item("over.coat_all", "Overlaps", "Coat over everything", True, True, _STEP, True),
        Item("over.jacket_hem", "Overlaps", "Jacket's hem over trousers or skirt", True, True, _STEP, True),
        Item("over.dress_legs", "Overlaps", "Dress or skirt over the legs", False, False, note="the legs under a skirt: not yet"),
        Item("over.belt", "Overlaps", "Belt over clothes", True, True, _STEP + [SR + "_BELT_THICK"], True),
        Item("over.sleeve_hand", "Overlaps", "Sleeve over the wrist", True, True, _STEP, False, amb),
        Item("over.hand_clothes", "Overlaps", "Hand or arm in front of the body", True, True, [SR + "_REF_HAND_SHRINK"], False,
             "crossed arms and hands on the body, from the posed reference; not scored yet"),
        Item("over.strap", "Overlaps", "Bag strap over clothes", True, True, _STEP, False, "named and stepped; not scored yet"),
        Item("over.earrings", "Overlaps", "Earrings by the hair", True, True, [SR + "_EARRING_PROUD"], True, "the overlaps score checks it"),
    ]


# ---------------------------------------------------------------- 206 bones
# role: "measure"  its length or width is read from the photo (pose, face or
#                  hand points at its ends) and fits the reference body
#       "landmark" it shows under the skin as a shape of its own (a ridge, a
#                  knob) where the skin is bare
#       "shape"    it shapes the body as part of the reference body / face
#                  landmarks, with no point of its own in the photo
#       "inside"   no shape of its own on a figure this size (inside the
#                  skull, the ear, the wrist or foot as a whole)

_BONES = []


def _b(name, region, role, rig, sides=False, n=1, note=""):
    names = [name] if n == 1 else [f"{name} {i}" for i in range(1, n + 1)]
    for nm in names:
        for side in ((" (left)", " (right)") if sides else ("",)):
            _BONES.append((nm + side, region, role, rig, note))


# skull: 8 cranial, 14 facial
_b("Frontal", "Skull", "landmark", "head", note="the brow ridge and forehead")
_b("Parietal", "Skull", "shape", "head", sides=True, note="the top of the skull")
_b("Temporal", "Skull", "landmark", "head", sides=True, note="the temples; the mastoid behind the ear")
_b("Occipital", "Skull", "landmark", "head", note="the bump at the back of the head, above the nape")
_b("Sphenoid", "Skull", "inside", "head")
_b("Ethmoid", "Skull", "inside", "head")
_b("Nasal", "Skull", "measure", "head", sides=True, note="the nose bridge (face landmarks)")
_b("Maxilla", "Skull", "shape", "head", sides=True, note="the upper jaw under the cheeks and lip")
_b("Zygomatic", "Skull", "landmark", "head", sides=True, note="the cheekbones")
_b("Mandible", "Skull", "measure", "head", note="the jaw: its width and angle (face landmarks)")
_b("Palatine", "Skull", "inside", "head", sides=True)
_b("Inferior nasal concha", "Skull", "inside", "head", sides=True)
_b("Lacrimal", "Skull", "inside", "head", sides=True)
_b("Vomer", "Skull", "inside", "head")
# ear ossicles, hyoid
for o in ("Malleus", "Incus", "Stapes"):
    _b(o, "Ear", "inside", "head", sides=True)
_b("Hyoid", "Neck", "inside", "neck01", note="under the chin; no shape of its own at this size")
# spine
for i in range(1, 7):
    _b(f"Cervical C{i}", "Spine", "shape", "neck01-03", note="the neck's length (reference neck)")
_b("Cervical C7", "Spine", "landmark", "neck01", note="the knob at the base of the nape")
for i in range(1, 13):
    _b(f"Thoracic T{i}", "Spine", "shape", "spine02-05", note="the back's length and curve")
for i in range(1, 6):
    _b(f"Lumbar L{i}", "Spine", "shape", "spine01", note="the lower back's curve")
_b("Sacrum", "Spine", "shape", "root", note="the lower back above the hips")
_b("Coccyx", "Spine", "inside", "root")
# chest
_b("Sternum", "Chest", "landmark", "spine05", note="the notch at the neck's base and the chest's centre line")
for i in range(1, 13):
    _b(f"Rib {i}", "Chest", "shape", "spine03-05", sides=True, note="the rib cage's shape")
# shoulder girdle
_b("Clavicle", "Shoulder", "measure", "clavicle", sides=True, note="the shoulders' width; the collarbone ridge where bare")
_b("Scapula", "Shoulder", "shape", "shoulder01", sides=True, note="the shoulder blades under the back")
# arms and hands
_b("Humerus", "Arm", "measure", "upperarm01-02", sides=True, note="shoulder to elbow")
_b("Radius", "Arm", "measure", "lowerarm01-02", sides=True, note="elbow to wrist, thumb side")
_b("Ulna", "Arm", "measure", "lowerarm01-02", sides=True, note="elbow to wrist; the elbow's point and the wrist's knob")
for c in ("Scaphoid", "Lunate", "Triquetrum", "Pisiform", "Trapezium", "Trapezoid", "Capitate", "Hamate"):
    _b(c, "Wrist", "inside", "wrist", sides=True, note="the wrist as a whole")
for i in range(1, 6):
    _b(f"Metacarpal {i}", "Hand", "measure", f"metacarpal{i - 1}" if i > 1 else "finger1-1", sides=True,
       note="the palm; the knuckles where bare")
for f_ in range(1, 6):
    parts = ("proximal", "distal") if f_ == 1 else ("proximal", "middle", "distal")
    for ph in parts:
        _b(f"Finger {f_} {ph} phalanx", "Hand", "measure", f"finger{f_}", sides=True)
# pelvis and legs
_b("Hip bone", "Pelvis", "measure", "pelvis", sides=True, note="the hips' width; the hip crest where bare")
_b("Femur", "Leg", "measure", "upperleg01-02", sides=True, note="hip to knee")
_b("Patella", "Leg", "landmark", "lowerleg01", sides=True, note="the kneecap")
_b("Tibia", "Leg", "measure", "lowerleg01-02", sides=True, note="knee to ankle; the shin and the inner ankle")
_b("Fibula", "Leg", "landmark", "lowerleg01-02", sides=True, note="the outer ankle")
for t in ("Calcaneus", "Talus", "Navicular", "Cuboid", "Medial cuneiform", "Intermediate cuneiform", "Lateral cuneiform"):
    _b(t, "Ankle and foot", "landmark" if t == "Calcaneus" else "inside", "foot", sides=True,
       note="the heel" if t == "Calcaneus" else "the foot as a whole")
for i in range(1, 6):
    _b(f"Metatarsal {i}", "Ankle and foot", "measure", "foot", sides=True, note="the foot's length (heel to toes)")
for t_ in range(1, 6):
    parts = ("proximal", "distal") if t_ == 1 else ("proximal", "middle", "distal")
    for ph in parts:
        _b(f"Toe {t_} {ph} phalanx", "Ankle and foot", "inside", f"toe{t_}", sides=True, note="the toes, usually in shoes")
assert len(_BONES) == 206, len(_BONES)

# what's in the code for each role (updated as it lands)
_ROLE_STATUS = {
    "measure": dict(read=False, sculpt=False, critic=False, note_add="length not yet measured from the photo"),
    "landmark": dict(read=NA, sculpt=False, critic=False, note_add="ridge/knob where bare: not yet"),
    "landmark_done": dict(read=True, sculpt=True, critic=True,
                          note_add="shaped where the photo shows bare skin there, placed from the posed body's joints (their miss scores it)"),
    "face_landmark": dict(read=True, sculpt=True, critic=True, note_add="in the face's landmark shape; the face's points score it"),
    "shape": dict(read=NA, sculpt=True, critic=True, note_add="from the reference body (bumps and depth checks score it)"),
    "inside": dict(read=NA, sculpt=NA, critic=NA, note_add="nothing of its own to shape at this size"),
}
# bones measured now: by the face landmarks (nasal, mandible), the pose's
# widths (hip bone), or the reference fit's bone lengths (body_reference._STRETCH)
_MEASURED_NOW = {"Nasal", "Mandible", "Clavicle", "Hip bone", "Humerus", "Radius", "Ulna", "Femur", "Tibia"}
_FOOT = ("Metatarsal", "Calcaneus")
_HAND = ("Metacarpal", "Finger")


def _bones() -> list:
    out = []
    for i, (name, region, role, rig, note) in enumerate(_BONES):
        base_ = name.split(" (")[0]
        if role == "landmark" and base_ in ("Frontal", "Zygomatic", "Temporal"):
            role = "face_landmark"
        elif role == "landmark" and base_ == "Occipital":
            role = "shape"  # the reference head's own (under hair on most photos)
        elif role == "landmark" and base_ == "Calcaneus":
            role = "measure"  # the heel: a point of the photo's the fit matches
        elif role == "landmark" and base_ in ("Cervical C7", "Sternum", "Patella", "Fibula"):
            role = "landmark_done"
        st = dict(_ROLE_STATUS[role])
        add = st.pop("note_add")
        base = name.split(" (")[0]
        if role == "measure" and base.startswith(_FOOT):
            st.update(read=True, sculpt=True, critic=True)
            add = "measured as the foot's size (heel and big toe in the photo, when the feet show)"
        elif role == "measure" and (base in _MEASURED_NOW or base.startswith(_HAND)):
            st.update(read=True, sculpt=True, critic=True)
            add = ("measured as the hand's size (the hand's points against the reference's)" if base.startswith(_HAND) else
                   "its length fitted to the photo's points; the fit's joint miss scores it" if base in
                   ("Humerus", "Radius", "Ulna", "Femur", "Tibia", "Clavicle") else "measured now (face or pose points)")
            if base in ("Femur", "Tibia"):
                add += " (when the legs are in the photo)"
        elif base.startswith("Cervical C") and base != "Cervical C7":
            st.update(read=True, sculpt=True, critic=True)
            add = "the neck's length fitted to the photo (face to shoulders); the fit's joint miss scores it"
        out.append(Item(f"bone.{i:03d}", f"Bones: {region}", name, st["read"], st["sculpt"], [], st["critic"],
                        f"{role}: {note + '; ' if note else ''}{add}", rig))
    return out


def items() -> list:
    return _face() + _head() + _worn() + _clothes() + _overlaps() + _bones()
