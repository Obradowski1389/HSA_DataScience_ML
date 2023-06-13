# Mapiranje boja iz legende u odgovarajuće klase
color_classes = {
    # (boja): klasa
    (255, 255, 255): 10,   # No data            
    (255, 255, 0): 0,       # Cultivated land
    (5, 73, 7): 1,          # Forest
    (255, 165, 0): 2,       # Grassland
    (128, 96, 0): 3,        # Shurb
    (6, 154, 243): 4,       # Water
    (149, 208, 252): 5,     # Wet
    (134, 111, 162): 6,     # Thundra           # Ne postoji
    (220, 20, 60): 7,       # Artificial
    (166, 166, 166): 8,     # BareLand
    (0, 0, 0): 9,           # Ice               # Ne postoji
}

color_classes_rev = {
    # (boja): klasa
    10:(255, 255, 255),   # No data            
    0:(255, 255, 0),       # Cultivated land
    1:(5, 73, 7),          # Forest
    2:(255, 165, 0),       # Grassland
    3:(128, 96, 0),        # Shurb
    4:(6, 154, 243),       # Water
    5:(149, 208, 252),     # Wet
    6:(134, 111, 162),     # Thundra           # Ne postoji
    7:(220, 20, 60),       # Artificial
    8:(166, 166, 166),     # BareLand
    9:(0, 0, 0),           # Ice               # Ne postoji
}