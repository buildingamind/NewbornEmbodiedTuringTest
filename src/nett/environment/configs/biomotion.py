
# TODO: refactor `../configs.py` to use files within `src/nett/environment/configs/*`

import csv
from nett.environment.configs import NETTConfig;

class Biomotion(NETTConfig):
    # is the Args block right? Not totally sure...
    """
    NETT configuration for Biological Motion Recognition experiment

    Args:
        imprinting_condition (str | list[str], optional): Motion type. Defaults to ["ChickBiologicalMotion", "InvertedBiologicalMotion"].
        test_condition (str | list[str], optional): Test condition. Defaults to ["Rest", "Inverted", "Random", "Rigid", "Cat", "Scrambled", "Color", "Stationary", "White"].
        target_video (str | list[str], optional): Motion Type video. Defaults to ["biomo.webm", "Inverted.webm"]. 
        nontarget_video (str | list[str], optional): Condition video. 
        left_monitor (str | list[str], optional): Video-to-monitor assignment. 
        right_monitor (str | list[str], optional): Video-to-monitor assignment. 
    """

    def __init__(self,
                    imprinting_condition: str | list[str] = ["ChickBiologicalMotion", "InvertedBiologicalMotion"],
                    test_condition: str | list[str] = ["Rest", "Inverted", "Random", "Rigid", "Cat", "Scrambled", "Color", "Stationary",  "White"],
                    # target_video: str | list[str] = ["biomo.webm", "Inverted.webm"],
                    # nontarget_video: str | list[str] = ["Rest", "Inverted", "Random", "Rigid", "Cat", "Scrambled", "Color", "Stationary",  "White"],
                    # left_monitor: str | list[str] = [],
                    # right_monitor: str | list[str] = []
                ) -> None:
        """Constructor method
        """
        super().__init__(
                    param_defaults = self.defaults,
                    imprinting_condition = imprinting_condition,
                    test_condition = test_condition,
                    # target_video = target_video,
                    # nontarget_video = nontarget_video,
                    # left_monitor = left_monitor,
                    # right_monitor = right_monitor
                )

    @property
    # TODO: move this row-counting method for DesignSheet_<ExperimentName> into more appropriate, shared placement within environment module 
    def num_conditions(self) -> int:
        """
        Get the number of conditions for the configuration.

        Returns:
            int: The number of conditions.
        """
        # TODO: pull the name of the .csv programmatically from the user-provided sheet 
        design_file = csv.reader(open("./biomotion.csv")) 
        # '4' is hard-coded for now to remove 4 lines from DesignSheet w/o data; programmatic implementation should happen
        row_count = len(list(design_file)) - 4 

        return row_count
