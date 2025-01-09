# ***************************************************************** #
#                  THIS FILE MODELS THE PROBLEM                     #
# ***************************************************************** #

from enum import Enum


# actions from a predefined feeding profile
class FeedAction(Enum):
    HIGH_DECREASE=-0.1
    DECREASE=-0.05
    NEUTRAL=0
    INCREASE=0.05
    HIGH_INCREASE=0.1


class Bioprocess():

    def __init__(self):
        pass


    def reset():
        pass

    def perform_action(self, action:FeedAction):
        pass

    def model():
        # we can add the model here to propagate states
        pass