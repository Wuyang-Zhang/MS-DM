"""Convert raw test point annotations from text files to MATLAB files."""

from pathlib import Path

import numpy as np
from scipy.io import savemat


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_TEST_ROOT = PROJECT_ROOT / "data" / "raw_test"
SPECIES = ("whitefly", "fruit_fly")


def convert_annotations(species_dir):
    """Convert every ``annotations/*.txt`` file below a species directory."""
    annotation_dir = species_dir / "annotations"
    output_dir = species_dir / "mats"
    output_dir.mkdir(parents=True, exist_ok=True)

    for annotation_path in sorted(annotation_dir.glob("*.txt")):
        points = []
        for line in annotation_path.read_text(encoding="utf-8").splitlines():
            values = line.split()
            if len(values) >= 2:
                points.append((float(values[0]), float(values[1])))

        point_array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        image_info = np.empty((1, 1), dtype=object)
        image_info[0, 0] = np.asarray([[point_array]], dtype=object)
        output_path = output_dir / (annotation_path.stem + ".mat")
        savemat(str(output_path), {"image_info": image_info})
        print("Saved {}".format(output_path))


def main():
    for species in SPECIES:
        convert_annotations(RAW_TEST_ROOT / species)


if __name__ == "__main__":
    main()
