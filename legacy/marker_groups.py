MARKER_GROUPS = {
    # Head
    "RFHD": ["LFHD", "LBHD", "RBHD"],  #
    "LFHD": ["RFHD", "LBHD", "RBHD"],  #
    "LBHD": ["RFHD", "LFHD", "RBHD"],  #
    "RBHD": ["RFHD", "LFHD", "LBHD"],  #
    # Right arm
    "RSHO": ["RUPA", "RELB", "CLAV"],  #
    "RUPA": ["RELB", "RSHO", "CLAV"],  #
    "RELB": ["RUPA", "RSHO", "RWRA"],  #
    "RWRA": ["RWRB", "RELB", "RUPA"],  #
    "RWRB": ["RWRA", "RELB", "RUPA"],  #
    "RFIN": ["RWRA", "RWRB", "RFRM"],  #
    "RFRM": ["RELB", "RWRA", "RWRB"],  # na schemacie jest RFRA
    # Left arm
    "LSHO": ["LUPA", "LELB", "CLAV"],  #
    "LUPA": ["LELB", "LSHO", "CLAV"],  #
    "LELB": ["LUPA", "LSHO", "LWRA"],  #
    "LWRA": ["LWRB", "LELB", "LUPA"],  #
    "LWRB": ["LWRA", "LELB", "LUPA"],  #
    "LFIN": ["LWRA", "LWRB", "LFRM"],  #
    "LFRM": ["LELB", "LWRA", "LWRB"],  # na schemacie jest LFRA
    # Torso
    "CLAV": ["RSHO", "LSHO", "STRN"],  #
    "C7": ["LSHO", "T10", "RSHO"],  #
    "T10": ["C7", "LPSI", "RPSI"],  #
    "RBAK": ["T10", "C7", "RSHO"],  #
    "STRN": ["RSHO", "LSHO", "CLAV"],  #
    # Hips
    "RPSI": ["LPSI", "T10", "RASI"],  #
    "LPSI": ["RPSI", "T10", "LASI"],  #
    "RASI": ["LASI", "STRN", "RPSI"],  #
    "LASI": ["RASI", "STRN", "LPSI"],  #
    # Right leg
    "RTHI": ["RASI", "RKNE", "RTIB"],  #
    "RKNE": ["RTHI", "RTIB", "RANK"],  #
    "RTIB": ["RKNE", "RANK", "RTOE"],  #
    "RANK": ["RTIB", "RTOE", "RHEE"],  #
    "RTOE": ["RANK", "RTIB", "RHEE"],  #
    "RHEE": ["RANK", "RTOE", "RTIB"],  #
    # Left leg
    "LTHI": ["LASI", "LKNE", "LTIB"],  #
    "LKNE": ["LTHI", "LTIB", "LANK"],  #
    "LTIB": ["LKNE", "LANK", "LTOE"],  #
    "LANK": ["LTOE", "LTIB", "LHEE"],  #
    "LTOE": ["LANK", "LTIB", "LHEE"],  #
    "LHEE": ["LANK", "LTOE", "LTIB"]  #
}
