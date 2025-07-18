#!/usr/bin/env python3
"""
Author:
    Lachlan Mares, lachlan.mares@adelaide.edu.au

License:
    GPL-3.0

Description:

"""

class RGBLEDColours:
    def __init__(self):
        self.colours = {"black": [0, 0, 0],
                        "white": [255, 255, 255],
                        "red": [255, 0, 0],
                        "lime": [0, 128, 0],
                        "blue": [0, 0, 255],
                        "yellow": [255, 255, 0],
                        "cyan": [0, 255, 255],
                        "magenta": [255, 0, 255],
                        "silver": [192, 192, 192],
                        "grey": [128, 128, 128],
                        "maroon": [128, 0, 0],
                        "olive": [128, 128, 0],
                        "green": [0, 255, 0],
                        "purple": [128, 0, 128],
                        "teal": [0, 128, 128],
                        "navy": [0, 0, 128],
                        "orange": [255, 165, 0],
                        "sea_green": [46, 139, 87],
                        "deep_sky_blue": [0, 191, 255],
                        "hot_pink": [255, 105, 180],
                        "brown": [139, 69, 19],
                        "dark_violet": [148, 0, 211],
                        }
    def get_available_colours(self) -> list:
        return list(self.colours.keys())

    def get_colour(self, colour: str) -> (list, str):
        colour_lower = colour.lower()
        return (self.colours[colour_lower], colour_lower) if colour_lower in self.colours.keys() else (self.colours['black'], 'black')


def main():
    rgbc = RGBLEDColours()

    print(rgbc.get_available_colours())
    print(rgbc.get_colour(colour='green'))
    print(rgbc.get_colour(colour='none'))


if __name__ == "__main__":
    main()