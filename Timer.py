from time import time
from math import floor


class Timer:
    """Simple Timer object which prints elapsed time since its creation"""

    def __init__(self, name):
        self.name = name
        self.__start_time = None
        self.__end_time = None
        self.start()

    def start(self):
        self.__start_time = time()

    def stop(self):
        self.__end_time = time()
        self.__get_elapsed__()

    def __get_elapsed__(self):
        """function to return correctly formatted string according to time units"""
        elapsed = (self.__end_time - self.__start_time)
        unit = "seconds"
        if elapsed >= 3600:
            unit = "minutes"
            hours = elapsed / 3600
            minutes = hours % 60
            hours = floor(hours)
            print("{} took {} hours and {:.2f} {} to complete".format(self.name, hours, minutes, unit))
        elif elapsed >= 60:
            minutes = floor(elapsed / 60)
            seconds = elapsed % 60
            print("{} took {} minutes and {:.2f} {} to complete".format(self.name, minutes, seconds, unit))
        else:
            print("{} took {:.2f} {} to complete".format(self.name, elapsed, unit))
