MARKER_GROUPS = {
    # ---------------------------------------------------------
    # PERFECT RIGID BODIES
    # ---------------------------------------------------------
    # Head
    "RFHD": ["LFHD", "LBHD", "RBHD"],
    "LFHD": ["RFHD", "LBHD", "RBHD"],
    "LBHD": ["RFHD", "LFHD", "RBHD"],
    "RBHD": ["RFHD", "LFHD", "LBHD"],

    # Pelvis
    "RPSI": ["LPSI", "RASI", "LASI"],
    "LPSI": ["RPSI", "RASI", "LASI"],
    "RASI": ["LASI", "RPSI", "LPSI"],
    "LASI": ["RASI", "RPSI", "LPSI"],

    # ---------------------------------------------------------
    # MOSTLY RIGID BODIES
    # ---------------------------------------------------------
    # Torso
    "CLAV": ["STRN", "C7", "T10"],
    "C7": ["T10", "CLAV", "STRN"],
    "T10": ["C7", "RBAK", "STRN"],
    "RBAK": ["T10", "C7", "STRN"],
    "STRN": ["CLAV", "C7", "T10"],

    # ---------------------------------------------------------
    # LIMB SEGMENTS (Grouped to stay on the same bone side)
    # ---------------------------------------------------------
    # Right Arm (Humerus vs Radius/Ulna)
    "RSHO": ["RUPA", "CLAV", "C7"],  # Keep on upper torso/arm
    "RUPA": ["RSHO", "RELB", "RFRM"],  # Crosses elbow
    "RELB": ["RUPA", "RSHO", "RFRM"],  # Crosses elbow
    "RWRA": ["RWRB", "RFRM", "RELB"],  # Keep on lower arm
    "RWRB": ["RWRA", "RFRM", "RELB"],  # Keep on lower arm
    "RFIN": ["RWRA", "RWRB", "RFRM"],  # Keep on hand/lower arm
    "RFRM": ["RWRA", "RWRB", "RELB"],

    # Left Arm
    "LSHO": ["LUPA", "CLAV", "C7"],
    "LUPA": ["LSHO", "LELB", "LFRM"],
    "LELB": ["LUPA", "LSHO", "LFRM"],
    "LWRA": ["LWRB", "LFRM", "LELB"],
    "LWRB": ["LWRA", "LFRM", "LELB"],
    "LFIN": ["LWRA", "LWRB", "LFRM"],
    "LFRM": ["LWRA", "LWRB", "LELB"],

    # Right Leg (Femur vs Tibia/Fibula)
    "RTHI": ["RASI", "RKNE", "RTIB"],  # Crosses hip/knee
    "RKNE": ["RTHI", "RTIB", "RANK"],  # Crosses knee
    "RTIB": ["RKNE", "RANK", "RHEE"],  # Keep on lower leg
    "RANK": ["RTIB", "RHEE", "RTOE"],  # Keep on foot/lower leg
    "RTOE": ["RANK", "RHEE", "RTIB"],  # Keep on foot
    "RHEE": ["RANK", "RTOE", "RTIB"],  # Keep on foot

    # Left Leg
    "LTHI": ["LASI", "LKNE", "LTIB"],
    "LKNE": ["LTHI", "LTIB", "LANK"],
    "LTIB": ["LKNE", "LANK", "LHEE"],
    "LANK": ["LTIB", "LHEE", "LTOE"],
    "LTOE": ["LANK", "LHEE", "LTIB"],
    "LHEE": ["LANK", "LTOE", "LTIB"]
}
