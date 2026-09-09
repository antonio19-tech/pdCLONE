from enum import Enum, IntEnum


class PlotType(Enum):
    XY = "xy"
    TRAJECTORY = "trajectory"
    PRESSURE_OVER_TIME = "pressure_over_time"


class PaHaWTasks(IntEnum):
    def __new__(cls, value, description):
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj.description = description
        return obj

    ARCHIMEDES_SPIRAL = (0, "Draw an Archimedes spiral")
    CURSIVE_LETTER_L = (1, "Write in cursive the letter 'l'")
    BIGRAM_LE = (2, "Write the bigram 'le'")
    TRIGRAM_LES = (3, "Write the trigram 'les'")
    WORD_LEKTORKA = (4, "Write in cursive the word 'lektorka' (female teacher)")
    WORD_POROVNAT = (5, "Write the word 'porovnat' (to compare)")
    WORD_NEPOPADNOUT = (6, "Write the word 'nepopadnout' (to not catch)")
    SENTENCE_TRAMVAJ_DNES_UZ_NEPOJEDE = (7, "Write in cursive the sentence 'Tramvaj dnes už nepojede'")
    
    
class PaHaWFeatures(Enum):
    def __new__(cls, value, description):
        obj = object.__new__(cls)
        obj._value_ = value  # the main value
        obj.description = description  # human-readable description
        return obj

    Y_COORDINATE = ("Y coordinate", "Vertical position on the screen")
    X_COORDINATE = ("X coordinate", "Horizontal position on the screen")
    TIME_STAMP   = ("time stamp", "Timestamp of the sample")
    BUTTON_STATE = ("button state", "Whether the button is pressed")
    AZIMUTH      = ("azimuth", "Angle of stylus rotation")
    ALTITUDE     = ("altitude", "Stylus tilt from vertical")
    PRESSURE     = ("pressure", "Pressure applied on the surface")
    
    
ids_to_exclude = ['00061', '00080', '00089']