import random
from itertools import combinations, product, chain

from shapely import LineString, Point, Polygon

from machine import MACHINE
import pandas as pd
import os


class Auto_fight():
    def __init__(self):
        self.score = [0, 0]
        self.drawn_lines = []
        self.board_size = 7
        self.num_dots = 0
        self.whole_points = []
        self.location = []
        self.triangles = []

        self.turn = None
        self.interval = None
        self.offset = None
        self.machine = {"RED": MACHINE(), "BLUE": MACHINE()}
        self.get_score = False
        self.set_new_board()
        self.initialize_turn()


    def set_new_board(self):
        self.num_dots = random.randrange(5, 21)
        self.score = {"RED": 0, "BLUE": 0}  # USER, MACHINE
        self.drawn_lines = []  # Drawn Lines
        self.whole_points = []
        self.location = []
        self.triangles = []
        self.turn = None
        self.initialize_turn()
        for x in range(self.board_size):
            for y in range(self.board_size):
                self.whole_points.append((x, y))
        self.whole_points = random.sample(self.whole_points, self.num_dots)

    def machine_go(self, color):
        self.machine[color].score = self.score
        self.machine[color].drawn_lines = self.drawn_lines
        self.machine[color].whole_points = self.whole_points
        self.machine[color].location = self.location
        self.machine[color].triangles = self.triangles

        line = self.machine[color].find_best_selection()
        line = self.organize_points(line)

        if self.check_availability(color, line):
            print(color, " draw ", line)
            self.drawn_lines.append(line)

            self.check_triangle(line)
            self.change_turn()

            if self.check_endgame():
                if self.score["RED"] == self.score["BLUE"]:
                    print("무승부")
                    # 무승부
                else:
                    if self.score["RED"] > self.score["BLUE"]:
                        # 선공우승
                        print("선공우승")
                    else:
                        print("후공우승")
                        # 후공우승

                for i in range(7):
                    for j in range(7):
                        if [i, j] in self.whole_points:
                            print(1, end="")
                        else:
                            print(0, end="")
                    print()
                return "restart"

    def check_availability(self, turn, line):
        line_string = LineString(line)
        # Must be one of the whole points
        condition1 = (line[0] in self.whole_points) and (line[1] in self.whole_points)

        # Must not skip a dot
        condition2 = True
        for point in self.whole_points:
            if point == line[0] or point == line[1]:
                continue
            else:
                if bool(line_string.intersection(Point(point))):
                    condition2 = False

        # Must not cross another line
        condition3 = True
        for l in self.drawn_lines:
            if len(list({line[0], line[1], l[0], l[1]})) == 3:
                continue
            elif bool(line_string.intersection(LineString(l))):
                condition3 = False

        # Must be a new line
        condition4 = (line not in self.drawn_lines)

        # Must be own turn
        condition5 = (self.turn == turn)

        if condition1 and condition2 and condition3 and condition4 and condition5:
            return True
        else:
            return False

    def check_endgame(self):
        remain_to_draw = [[point1, point2] for (point1, point2) in list(combinations(self.whole_points, 2))
                          if self.check_availability(self.turn, [point1, point2])]
        return False if remain_to_draw else True

    def change_turn(self):
        turn = self.turn
        if turn == "RED":
            self.turn = "BLUE"
        elif turn == "BLUE":
            self.turn = "RED"

    def organize_points(self, point_list):
        point_list.sort(key=lambda x: (x[0], x[1]))
        return point_list

    def check_triangle(self, line):
        self.get_score = False

        point1 = line[0]
        point2 = line[1]

        point1_connected = []
        point2_connected = []

        for l in self.drawn_lines:
            if l == line:  # 자기 자신 제외
                continue
            if point1 in l:
                point1_connected.append(l)
            if point2 in l:
                point2_connected.append(l)

        if point1_connected and point2_connected:  # 최소한 2점 모두 다른 선분과 연결되어 있어야 함
            for line1, line2 in product(point1_connected, point2_connected):

                # Check if it is a triangle & Skip the triangle has occupied
                triangle = self.organize_points(list(set(chain(*[line, line1, line2]))))
                if len(triangle) != 3 or triangle in self.triangles:
                    continue

                empty = True
                for point in self.whole_points:
                    if point in triangle:
                        continue
                    if bool(Polygon(triangle).intersection(Point(point))):
                        empty = False

                if empty:
                    self.triangles.append(triangle)
                    self.score[self.turn] += 1
                    self.get_score = True

    def initialize_turn(self):
        self.turn = "RED"


if "__main__" == __name__:
    a = Auto_fight()
    flag = "go"
    turn = "RED"
    while flag != "restart":
        flag = a.machine_go(color=turn)
        if turn == "RED":
            turn = "BLUE"
        else:
            turn = "RED"
