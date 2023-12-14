from collections import Counter

from shapely import Polygon
from shapely.geometry import LineString, Point
from itertools import combinations, product
import matplotlib.pyplot as plt

debug = True


def generate_available(drawn_lines, whole_points):
    available = []
    for point1, point2 in combinations(whole_points, 2):
        line_string = LineString([point1, point2])
        if sorted((point1, point2)) in drawn_lines:
            continue
        flag = False  # 교차 체크
        for l in drawn_lines:
            if len({point1, point2, l[0], l[1]}) == 4:
                if bool(line_string.intersection(LineString(l))):
                    flag = True
                    break
        if flag:
            continue
        for point in whole_points:
            if point == point1 or point == point2:
                continue
            else:
                if bool(line_string.intersection(Point(point))):
                    flag = True
        if flag:
            continue
        available.append([point1, point2])
    return available


def inner_point(point1, point2, point3, point):
    try:
        a = ((point2[1] - point3[1]) * (point[0] - point3[0]) + (point3[0] - point2[0]) * (point[1] - point3[1])) / (
                (point2[1] - point3[1]) * (point1[0] - point3[0]) + (point3[0] - point2[0]) * (point1[1] - point3[1]))
        b = ((point3[1] - point1[1]) * (point[0] - point3[0]) + (point1[0] - point3[0]) * (point[1] - point3[1])) / (
                (point2[1] - point3[1]) * (point1[0] - point3[0]) + (point3[0] - point2[0]) * (point1[1] - point3[1]))
    except:
        return False

    c = 1 - a - b
    if a > 0 and b > 0 and c > 0:
        return True
    else:
        return False


def available_update(available, lastDrawn):
    available = available.copy()
    lastDrawn_sorted_list = sorted(lastDrawn)
    available.remove(lastDrawn_sorted_list)
    line_string = LineString(lastDrawn_sorted_list)

    for l in available:
        if len(list({lastDrawn_sorted_list[0], lastDrawn_sorted_list[1], l[0], l[1]})) == 3:
            continue

        elif bool(line_string.intersection(LineString(l))):
            available.remove(l)
    return available


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


def check_triangle(line, whole_line, whole_points):
    point1, point2 = line
    l1 = []
    l2 = []
    third_point = []
    for l in whole_line:
        # assert type(l) is list
        if l == line:
            continue
        if point1 in l:
            l1.append(l.copy())
        if point2 in l:
            l2.append(l.copy())
    for line1, line2 in product(l1, l2):
        for element in line1:
            if element in line2:
                innerpoint = True
                for point in whole_points:
                    if point in [point1, point2, element]:
                        continue
                    if inner_point(point1, point2, element, point):
                        innerpoint = False
                if innerpoint:
                    third_point.append(element)
    return len(third_point)


def get_score_line(drawn_lines, whole_points, available):
    candiate_triangle = []
    for line1, line2 in combinations(drawn_lines, 2):
        dots_three = sorted(list({line1[0], line1[1], line2[0], line2[1]}))
        if len(dots_three) != 3:
            continue
        flag = False
        for p in whole_points:
            if inner_point(dots_three[0], dots_three[1], dots_three[2], p):
                flag = True
                break
        if flag:
            continue
        point1 = [item for item in dots_three if item not in line1][0]
        point2 = [item for item in dots_three if item not in line2][0]
        if sorted([point1,point2]) not in available:
            continue
        candiate_triangle.append(sorted([point1, point2]))

    unique_items = []
    counter_square = []

    for item in candiate_triangle:
        if item in unique_items:
            counter_square.append(item)
        else:
            unique_items.append(item)
    return counter_square, candiate_triangle


def get_lines_between_lines(whole_line, whole_points, available_line):
    # assert type(whole_line) is list, "get_lines_between_lines 의 whole_line 가 리스트가 아님"
    res = []
    for line1, line2 in list(combinations(whole_line, 2)):
        # 두 선분의 점의 수를 세고 4개가 아니면, 즉 각각 떨어져 있는 선분이 아니면 진행하지 않음
        for point in line1:
            if point in line2:
                break
        square = Polygon([line1[0], line1[1], line2[0], line2[1]])
        flag = True
        for point in whole_points:
            if square.intersection(Point(point)) and point not in [line1[0], line1[1], line2[0], line2[1]]:
                flag = False
                break
        if flag:
            for line in list(combinations([line1[0], line1[1], line2[0], line2[1]], 2)):
                if line not in res and line in available_line:
                    res.append(list(line))
            try:
                res.remove(list(line1))
            except:
                pass
            try:
                res.remove(list(line2))
            except:
                pass
    return res


def get_lines_in_triangle(whole_line, whole_points, available_line):
    alone_points = whole_points.copy()
    for point1, point2 in whole_line:
        try:
            alone_points.remove(point1)
        except:
            pass
        try:
            alone_points.remove(point2)
        except:
            pass
    target_line = []
    for point1, point2 in available_line:
        if point1 in alone_points or point2 in alone_points:
            if point1 in alone_points and point2 in alone_points:
                continue
            else:
                target_line.append([point1, point2])
    return target_line


def evaluation(whole_line, whole_points, available_line):
    good_lines = get_lines_between_lines(whole_line, whole_points, available_line)
    bad_lines = get_lines_in_triangle(whole_line, whole_points, available_line)
    score = (len(good_lines) - len(bad_lines)) / (len(available_line) + 1)
    return score


def showmap(drawn_lines, whole_points):
    for line in drawn_lines:
        multiline = LineString(line)
        x, y = multiline.xy
        plt.plot(x, y, 'o-')
    for point in whole_points:
        plt.scatter(point[0], point[1], color='black', marker='o')
    plt.title('MADE BY LEESEOKJIN')
    plt.show()
