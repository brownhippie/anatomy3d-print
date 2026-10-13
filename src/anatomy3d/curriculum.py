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
    return [
        Item("face.shape", "Face", "Face shape: jaw, cheeks, chin, forehead", True, True,
             [SR + "_FACE_RELIEF_FADE_FRAC", SR + "_FACE_RELIEF_SMOOTH_FRAC", SR + "_FACE_RELIEF_WITH_DEPTH"], True,
             "468 face landmarks; the Lab's face-point error and face-embedding match score it"),
        Item("face.nose", "Face", "Nose: bridge, tip, wings", True, True, [], False,
             "landmark shape only; per-part score and its own numbers to come"),
        Item("face.nostrils", "Face", "Nostrils", False, False),
        Item("face.eyes", "Face", "Eyes: sockets, lids, eyeballs", True, True, [], False, "sockets from landmarks; lids, eyeballs to come"),
        Item("face.eyes_open", "Face", "Eyes open or closed", False, False),
        Item("face.irises", "Face", "Irises and pupils", False, False),
        Item("face.brows", "Face", "Eyebrows (the hair)", False, False, note="the brow ridge is in the face shape; the brows' hair is not"),
        Item("face.lips", "Face", "Lips", True, True, [], False, "landmark shape only"),
        Item("face.mouth_open", "Face", "Mouth open (its opening)", False, False),
        Item("face.teeth", "Face", "Teeth", False, False),
        Item("face.tongue", "Face", "Tongue", False, False),
        Item("face.scars", "Face", "Scars", False, False),
        Item("face.beard", "Face", "Beard", False, False),
        Item("face.moustache", "Face", "Moustache", False, False),
        Item("face.stubble", "Face", "Stubble", False, False),
        Item("face.creases", "Face", "Creases: smile lines, forehead lines", True, True, [SR + "_REF_RELIEF_MAX"], False,
             "from photo depth where the face faces the camera"),
        Item("face.ears", "Face", "Ears: size, place, swept back, under hair", True, True,
             [SR + "_REF_EAR_HAIR", SR + "_REF_EAR_SINK", SR + "_REF_EAR_THICK", BR + "_EAR_DROP", BR + "_EAR_KINK", BR + "_EAR_BEND",
              BR + "_EAR_UNDER_HAIR"], False, "3D ears measured from the photo"),
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
             False, "read; the pattern ends at it; a score of its own to come"),
        Item("hair.nape", "Hair", "Nape and behind the ears: hairline to collar", False, False,
             [SR + "_HAIR_NAPE_EDGE", SR + "_HAIR_NECK_BEHIND"], False,
             "smooth with its curves (nape hollow, the muscle behind the ear); built from what covers it: hair, collar, hood, scarf"),
        Item("hair.parting", "Hair", "Parting", False, False),
        Item("hair.braids", "Hair", "Braids, locs", False, False),
    ]


def _worn() -> list:
    return [
        Item("wear.cap", "Headwear", "Cap or hat, its peak", True, True, [SR + "_HAT_PEAK"], False),
        Item("wear.mask", "Headwear", "Face mask", True, True, [SR + "_MASK_SPAN"], False),
        Item("wear.glasses", "Headwear", "Glasses, sunglasses", True, True, [SR + "_GLASSES_SPAN", SR + "_GLASSES_FRAME"], False),
        Item("wear.earrings", "Headwear", "Earrings", False, False),
        Item("neck.neck", "Neck", "Neck", True, True, [], True, "from the reference body's neck"),
        Item("neck.collar", "Neck", "Collar, round the back too", True, True,
             [SR + "_NECK_COLLAR", SR + "_NECK_COLLAR_BELOW", SR + "_NECK_COLLAR_RISE"], False),
        Item("neck.scarf", "Neck", "Scarf, roll neck", True, True, [SR + "_NECK_WEAR_MAX"], False),
        Item("neck.necklace", "Neck", "Necklace", True, True, [SR + "_NECK_LACE"], False),
    ]


_GARMENTS = [
    ("shirt", "Shirt, blouse"), ("tshirt", "T-shirt, top"), ("jacket", "Jacket, blazer, suit"), ("coat", "Coat"),
    ("sweater", "Sweater, cardigan"), ("hoodie", "Hoodie (and its hood)"), ("dress", "Dress"), ("skirt", "Skirt"),
    ("trousers", "Trousers"), ("jeans", "Jeans"), ("shorts", "Shorts"), ("kimono", "Kimono, robe, wrap (and its sash)"),
    ("tie", "Tie, bow tie"), ("belt", "Belt"), ("lapels", "Lapels"), ("sleeves", "Sleeves and cuffs"), ("hems", "Hems"),
    ("buttons", "Buttons"), ("pockets", "Pockets"), ("folds", "Folds and drape"), ("shoes", "Shoes"), ("gloves", "Gloves"),
    ("bag", "Bag, straps"), ("watch", "Watch, bracelet"),
]


def _clothes() -> list:
    out = []
    for key, name in _GARMENTS:
        if key == "folds":
            out.append(Item("clothes.folds", "Clothes", name, True, True, [SR + "_REF_RELIEF_MAX", SR + "_REF_BACK_SMOOTH", SR + "_REF_SIDE_SMOOTH"],
                            True, "photo depth on surfaces facing the camera; bumps score the sides and back"))
        elif key in ("shirt", "jacket"):
            out.append(Item(f"clothes.{key}", "Clothes", name, False, True, [SR + "_REF_DRAPE", SR + "_REF_HANG"], False,
                            "a man's chest draped as cloth; the garment itself not yet named"))
        else:
            out.append(Item(f"clothes.{key}", "Clothes", name, False, False))
    return out


def _overlaps() -> list:
    return [
        Item("over.hair_ears", "Overlaps", "Hair over the ears", True, True, [BR + "_EAR_UNDER_HAIR", SR + "_REF_EAR_HAIR"], False),
        Item("over.hair_face", "Overlaps", "Hair over the forehead and cheeks", True, True, [SR + "_HAIR_FRINGE", SR + "_HAIR_FACE_MEET"], False),
        Item("over.hair_neck", "Overlaps", "Hair beside and behind the neck", True, True, [SR + "_HAIR_NECK_BEHIND", SR + "_HAIR_HANG"], False),
        Item("over.hair_clothes", "Overlaps", "Hair over shoulders and clothes", True, True, [SR + "_HAIR_LAYER"], False),
        Item("over.hat_hair", "Overlaps", "Cap or hat over hair", True, True, [SR + "_HAT_PEAK"], False),
        Item("over.glasses_face", "Overlaps", "Glasses over the face", True, True, [SR + "_GLASSES_SPAN", SR + "_GLASSES_FRAME"], False),
        Item("over.mask_face", "Overlaps", "Mask over the face", True, True, [SR + "_MASK_SPAN"], False),
        Item("over.collar_neck", "Overlaps", "Collar round the neck", True, True, [SR + "_NECK_COLLAR"], False),
        Item("over.scarf_clothes", "Overlaps", "Scarf over clothes", True, True, [SR + "_NECK_WEAR_MAX"], False),
        Item("over.necklace", "Overlaps", "Necklace over skin or clothes", True, True, [SR + "_NECK_LACE"], False),
        Item("over.jacket_shirt", "Overlaps", "Jacket over shirt", False, False),
        Item("over.tie_shirt", "Overlaps", "Tie over shirt", False, False),
        Item("over.sweater_shirt", "Overlaps", "Sweater over shirt (collar out)", False, False),
        Item("over.coat_all", "Overlaps", "Coat over everything", False, False),
        Item("over.dress_legs", "Overlaps", "Dress or skirt over the legs", False, False),
        Item("over.belt", "Overlaps", "Belt over clothes", False, False),
        Item("over.sleeve_hand", "Overlaps", "Sleeve over the wrist", False, False),
        Item("over.hand_clothes", "Overlaps", "Hand or arm in front of the body", True, True, [SR + "_REF_HAND_SHRINK"], False,
             "crossed arms and hands on the body, from the posed reference"),
        Item("over.strap", "Overlaps", "Bag strap over clothes", False, False),
        Item("over.earrings", "Overlaps", "Earrings by the hair", False, False),
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
    "shape": dict(read=NA, sculpt=True, critic=True, note_add="from the reference body (bumps and depth checks score it)"),
    "inside": dict(read=NA, sculpt=NA, critic=NA, note_add="nothing of its own to shape at this size"),
}
# bones the face landmarks already measure and shape
_MEASURED_NOW = {"Nasal", "Mandible", "Clavicle", "Hip bone"}


def _bones() -> list:
    out = []
    for i, (name, region, role, rig, note) in enumerate(_BONES):
        st = dict(_ROLE_STATUS[role])
        add = st.pop("note_add")
        base = name.split(" (")[0]
        if role == "measure" and base in _MEASURED_NOW:
            st.update(read=True, sculpt=True, critic=True)
            add = "measured now (face or pose points); fits the reference"
        out.append(Item(f"bone.{i:03d}", f"Bones: {region}", name, st["read"], st["sculpt"], [], st["critic"],
                        f"{role}: {note + '; ' if note else ''}{add}", rig))
    return out


def items() -> list:
    return _face() + _head() + _worn() + _clothes() + _overlaps() + _bones()
