import random
from itertools import combinations
from shapely.geometry import LineString, Point

class Legacy_M():
    class Node:
        def __init__(self, score, added_lines):
            self.score = score
            self.added_lines = added_lines
            self.children = []

    def __init__(self, score=[0, 0], drawn_lines=[], whole_lines=[], whole_points=[], location=[]):
        self.id = "MACHINE"
        self.score = [0, 0]
        self.drawn_lines = []
        self.board_size = 7
        self.num_dots = 0
        self.whole_points = []

        self.available = []
        self.lastState = []
        self.lastDrawn = []

        self.location = []
        self.triangles = []
        self.isRule = True
        self.isHeurisitic = True
        self.isMinMax = True

    def evaluation(self):
        # 이미 그려져있는 선분 중 2개씩 선택
        pair_list = list(combinations(self.drawn_lines, 2))
        for pair in pair_list:
            # 두 선분의 점의 수를 세고 4개가 아니면, 즉 각각 떨어져 있는 선분이 아니면 진행하지 않음
            dot_cnt = list(set([pair[0][0], pair[0][1], pair[1][0], pair[1][1]]))
            if len(dot_cnt) == 4:
                # 4개의 점으로 그릴 수 있는 선분 6개 중 이미 그려진 2개의 선분 제외 후 4개의 선분을 전부 그릴 수 있는지 판단
                lines = [[pair[0][0], pair[1][0]], [pair[0][0], pair[1][1]], [pair[0][1], pair[1][0]],
                         [pair[0][1], pair[1][1]]]
                # 4개의 선분 중 하나라도 그릴 수 없으면 falg가 False가 되고 다음 후보로 진행
                flag = True
                for i in lines:
                    if self.check_availability(i) == False:
                        flag = False
                    # print("i :", i, "flag :", flag)
                # flag가 True일 시 네 선분 다 그릴 수 있는 선분임. 이중 하나 리턴
                if flag:
                    return list(lines[0])
        # 가능한 선이 없을 시 -1 리턴
        return -1


    def check_stealing_situation_inOpponentTurn(self, line):
        connected_lines = [l for l in self.drawn_lines if set(line) & set(l)]

        if len(connected_lines) < 1:
            return False

        isDangerous = False

        for connected_line in connected_lines:

            # 그냥 all_points_set_withDuplicate
            all_points_set_withDuplicate = list([connected_line[0], connected_line[1], line[0], line[1]])
            # 중복 제거를 위해 set으로 함
            all_points_set_nonDuplicate = set([connected_line[0], connected_line[1], line[0], line[1]])

            # 중복된 값 찾기 (tuple을 넘겨줄 것임)
            overlapping_point = \
            [item for item in all_points_set_nonDuplicate if all_points_set_withDuplicate.count(item) > 1][0]

            if not overlapping_point:
                return False

            all_points_set_nonDuplicate.discard(overlapping_point)
            non_overlapping_points = list(all_points_set_nonDuplicate)

            if is_triangle(overlapping_point, non_overlapping_points[0],
                           non_overlapping_points[1]) and self.check_availability(
                    [non_overlapping_points[0], non_overlapping_points[1]]):
                for point in self.whole_points:
                    if point == overlapping_point or point == non_overlapping_points[0] or point == \
                            non_overlapping_points[1]:
                        continue
                    elif inner_point_usingInStealChecking(overlapping_point, non_overlapping_points[0],
                                                          non_overlapping_points[1], point):
                        break
                    else:
                        isDangerous = True
            else:
                continue

        if isDangerous == False:
            return False
        else:
            return True

    def check_valid_line(self, line):
        if not self.check_availability(line):
           return False
        if self.check_stealing_situation_inOpponentTurn(line) == True:
            return False
        return True

    def find_best_selection(self):

        if self.isRule:
            available_skipworst = [[point1, point2] for (point1, point2) in list(combinations(self.whole_points, 2)) if
                                   self.check_valid_line([point1, point2])]
            available_all = [[point1, point2] for (point1, point2) in list(combinations(self.whole_points, 2)) if
                             self.check_availability([point1, point2])]
            candidate_line = self.check_triangle(available_skipworst)
            if len(candidate_line) == 0:
                candidate_line = self.check_triangle(available_all)
            if len(candidate_line) != 0:
                return random.choice(candidate_line)  # 그들 중에 random 선택
            elif len(available_skipworst) != 0:
                return random.choice(available_skipworst)
            else:
                eval_line = self.evaluation()
                if eval_line != -1:
                    return eval_line
                return random.choice(available_all)
    def check_triangle(self, available):
        avail_triangle = []
        candiate_triangle = []
        prev_triangle = list(combinations(self.drawn_lines, 2))
        for lines in prev_triangle[:]:
            dots_three = list(set([lines[0][0], lines[0][1], lines[1][0], lines[1][1]]))
            if len(dots_three) != 3:
                prev_triangle.remove(lines)
        for lines in prev_triangle:
            dots_three = list(set([lines[0][0], lines[0][1], lines[1][0], lines[1][1]]))
            bitFlag = 0
            if [dots_three[0], dots_three[1]] in available:
                bitFlag = bitFlag | (1 << 0)
            if [dots_three[0], dots_three[2]] in available:
                bitFlag = bitFlag | (1 << 1)
            if [dots_three[1], dots_three[2]] in available:
                bitFlag = bitFlag | (1 << 2)
            if [dots_three[1], dots_three[0]] in available:
                bitFlag = bitFlag | (1 << 3)
            if [dots_three[2], dots_three[0]] in available:
                bitFlag = bitFlag | (1 << 4)
            if [dots_three[2], dots_three[1]] in available:
                bitFlag = bitFlag | (1 << 5)
            if bitFlag != 0:
                flag = True
                for p in self.whole_points:
                    if inner_point(dots_three[0], dots_three[1], dots_three[2], p):
                        flag = False
                if flag:
                    if bitFlag & (1 << 0):
                        candiate_triangle.append([dots_three[0], dots_three[1]])
                    elif bitFlag & (1 << 1):
                        candiate_triangle.append([dots_three[0], dots_three[2]])
                    elif bitFlag & (1 << 2):
                        candiate_triangle.append([dots_three[1], dots_three[2]])
                    elif bitFlag & (1 << 3):
                        candiate_triangle.append([dots_three[1], dots_three[0]])
                    elif bitFlag & (1 << 4):
                        candiate_triangle.append([dots_three[2], dots_three[0]])
                    elif bitFlag & (1 << 5):
                        candiate_triangle.append([dots_three[1], dots_three[2]])
        return candiate_triangle

    def check_availability(self, line):
        line_string = LineString(line)
        condition1 = (line[0] in self.whole_points) and (line[1] in self.whole_points)
        condition2 = True
        for point in self.whole_points:
            if point == line[0] or point == line[1]:
                continue
            else:
                if bool(line_string.intersection(Point(point))):
                    condition2 = False
        condition3 = True
        for l in self.drawn_lines:
            if len(list(set([line[0], line[1], l[0], l[1]]))) == 3:
                continue
            elif bool(line_string.intersection(LineString(l))):
                condition3 = False
        condition4 = (line not in self.drawn_lines)
        if condition1 and condition2 and condition3 and condition4:
            return True
        else:
            return False


def inner_point(point1, point2, point3, point):
    try:
        a = ((point2[1] - point3[1]) * (point[0] - point3[0]) + (point3[0] - point2[0]) * (point[1] - point3[1])) / (
                    (point2[1] - point3[1]) * (point1[0] - point3[0]) + (point3[0] - point2[0]) * (
                        point1[1] - point3[1]))
        b = ((point3[1] - point1[1]) * (point[0] - point3[0]) + (point1[0] - point3[0]) * (point[1] - point3[1])) / (
                    (point2[1] - point3[1]) * (point1[0] - point3[0]) + (point3[0] - point2[0]) * (
                        point1[1] - point3[1]))
    except:
        return False
    c = 1 - a - b
    if a > 0 and b > 0 and c > 0:
        return True
    else:
        return False


def inner_point_usingInStealChecking(point1, point2, point3, point):
    try:
        a = ((point2[1] - point3[1]) * (point[0] - point3[0]) + (point3[0] - point2[0]) * (point[1] - point3[1])) / (
                    (point2[1] - point3[1]) * (point1[0] - point3[0]) + (point3[0] - point2[0]) * (
                        point1[1] - point3[1]))
        b = ((point3[1] - point1[1]) * (point[0] - point3[0]) + (point1[0] - point3[0]) * (point[1] - point3[1])) / (
                    (point2[1] - point3[1]) * (point1[0] - point3[0]) + (point3[0] - point2[0]) * (
                        point1[1] - point3[1]))
    except:
        return False
    c = 1 - a - b

    if 0 <= a <= 1 and 0 <= b <= 1 and 0 <= c <= 1:
        return True
    else:
        return False

def is_triangle(p1, p2, p3):
    try:
        slope1 = (p1[1] - p2[1]) / (p1[0] - p2[0])
        slope2 = (p3[1] - p2[1]) / (p3[0] - p2[0])
        return slope1 != slope2
    except ZeroDivisionError:
        return True
    except:
        return False