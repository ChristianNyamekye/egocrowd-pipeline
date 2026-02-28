"""EgoCrowd: Crowdsourced egocentric manipulation data for robot learning."""

__version__ = "0.1.0"
__author__ = "Christian Nyamekye"

from egocrowd.parse import parse_r3d
from egocrowd.retarget import spatial_trajectory
from egocrowd.export import to_lerobot_hdf5, to_rlds_json, to_raw_json
