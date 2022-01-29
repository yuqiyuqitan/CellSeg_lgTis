import os

import pandas as pd

from src import cvutils


class CVConfig:

    # MODEL_DIRECTORY - (string) path to save logs to
    root = os.path.dirname(os.path.realpath(__file__))
    MODEL_DIRECTORY = os.path.join(root, "modelFiles")
    MODEL_PATH = os.path.join(root, "src", "modelFiles", "final_weights.h5")

    # target - (string) path to directory containing image folder and channels file
    # growth_pixels - (int) number of pixels from which to grow out from the nucleus to define a cell boundary.  Change based on tissue types (default value is 1).
    def __init__(self, target, growth_pixels):

        ###

        # paths

        # CHANNEL_PATH - (string) path to your channels file (usually named channelnames.txt). Only used for tif images with more than 3 channels, or 4D TIF images.
        self.CHANNEL_PATH = os.path.join(target, "channelnames.txt")

        self.output_path_name = target + "_out_gp_" + str(growth_pixels)
        self.DIRECTORY_PATH = os.path.join(target, "bestFocus/fullSizeMontage")
        self.IMAGEJ_OUTPUT_PATH = os.path.join(self.output_path_name, "imagej_files")
        self.QUANTIFICATION_OUTPUT_PATH = os.path.join(
            self.output_path_name, "quantifications"
        )
        self.VISUAL_OUTPUT_PATH = os.path.join(self.output_path_name, "visual_output")
        self.PROGRESS_TABLE_PATH = os.path.join(
            self.output_path_name, "progress_table.txt"
        )

        ###

        # additional configuration

        # IS_CODEX_OUTPUT - CODEX output files have special filenames that allow outputs to contain more metadata about absolute positions, regs, and other things. Set this parameter to False if not using the filename convention, otherwise set it to True. Follow the naming convention in the install/run page on the CellVision website to output this metadata for non-CODEX images.
        self.IS_CODEX_OUTPUT = True
        # NUCLEAR_CHANNEL_NAME - (string) name of nuclear stain (corresponding to channelnames.txt).  Case sensitive.  Only used for tif images with more than 3 channels, or 4D TIF images.
        self.NUCLEAR_CHANNEL_NAME = "Hoechst1"
        self.USE_GROWTH = True
        # BOOST - (double or 'auto') multiplier with which to boost the pixels of the nuclear stain before inference.  Choose 'auto' to try to infer the best boost to use based off of AUTOBOOST_PERCENTILE (default value is 'auto')
        self.GROWTH_METHOD = "Sequential"
        self.BOOST = "auto"
        # AUTOBOOST_REFERENCE_IMAGE - (string) If autoboosting, then set this to the image's filename to choose which image to autoboost off of (generally choose a non-empty image).  If image is not found or if it is empty, then the program uses first loaded image to autoboost. Parameter not used if BOOST is not set to 'auto', but gets metadata from selected image.
        self.AUTOBOOST_REFERENCE_IMAGE = "reg001_montage.tif"
        self.FILENAME_ENDS_TO_EXCLUDE = "xxx.tif"
        # OVERLAP (int) - amount of pixels overlap with which to run the stitching algorithm. Must be divisible by 2 and should be greater than expected average cell diameter in pixels (default value is 80).
        self.OVERLAP = 80
        # THRESHOLD - (int) minimum size (in pixels) of kept segmented instances. Objects smaller than THRESHOLD are not included in final segmentation output (default value is 20).
        self.THRESHOLD = 20
        # INCREASE_FACTOR - (double) Amount with which to boost the size of the image. Default is 2.5x, decided by visual inspection after training on the Kaggle dataset (default value is 2.5).
        self.INCREASE_FACTOR = 3
        # AUTOBOOST_PERCENTILE - (double) The percentile value with which to saturate to (default value is 99.98).
        self.AUTOBOOST_PERCENTILE = 99.98
        self.OUTPUT_METHOD = "all"

        ###

        try:
            os.makedirs(self.IMAGEJ_OUTPUT_PATH)
            os.makedirs(self.QUANTIFICATION_OUTPUT_PATH)
            os.makedirs(self.VISUAL_OUTPUT_PATH)
        except FileExistsError:
            print("Directory already exists")

        self.CHANNEL_NAMES = pd.read_csv(
            self.CHANNEL_PATH, sep="\t", header=None
        ).values[:, 0]

        self.GROWTH_PIXELS = growth_pixels

        VALID_IMAGE_EXTENSIONS = ("tif", "jpg", "png")
        self.FILENAMES = [
            f
            for f in os.listdir(self.DIRECTORY_PATH)
            if f.endswith(VALID_IMAGE_EXTENSIONS)
            and not f.startswith(".")
            and not f.endswith(self.FILENAME_ENDS_TO_EXCLUDE)
        ]

        VALID_IMAGE_EXTENSIONS = ("tif", "jpg", "png")
        self.FILENAMES = [
            f
            for f in os.listdir(self.DIRECTORY_PATH)
            if f.endswith(VALID_IMAGE_EXTENSIONS)
            and not f.startswith(".")
            and not f.endswith(self.FILENAME_ENDS_TO_EXCLUDE)
        ]

        if len(self.FILENAMES) < 1:
            raise NameError(
                "No image files found.  Make sure you are pointing to the right directory"
            )

        reference_image_path = os.path.join(self.DIRECTORY_PATH, self.FILENAMES[0])

        if self.AUTOBOOST_REFERENCE_IMAGE != "" and self.BOOST == "auto":
            if self.AUTOBOOST_REFERENCE_IMAGE in self.FILENAMES:
                self.FILENAMES.remove(self.AUTOBOOST_REFERENCE_IMAGE)
                self.FILENAMES.insert(0, self.AUTOBOOST_REFERENCE_IMAGE)
                print(
                    "Using autoboost reference image with filename",
                    self.AUTOBOOST_REFERENCE_IMAGE,
                )
            else:
                print(
                    "AUTOBOOST_REFERENCE_IMAGE does not exist.  Check your config file - image filename must match exactly."
                )
                print("Defaulting to first image reference...")

        (
            self.N_DIMS,
            self.EXT,
            self.DTYPE,
            self.SHAPE,
            self.READ_METHOD,
        ) = cvutils.meta_from_image(reference_image_path)
        self.PROGRESS_TABLE = []

        if os.path.exists(self.PROGRESS_TABLE_PATH):
            self.PROGRESS_TABLE = [
                line.rstrip("\n") for line in open(self.PROGRESS_TABLE_PATH)
            ]
