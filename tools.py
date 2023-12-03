from collections import Counter

from shapely import Polygon
from shapely.geometry import LineString, Point
from itertools import combinations, product
import matplotlib.pyplot as plt

debug = True


# [[1,1],[1,0]].sort -> [1,0],[1,1]
# whole_point가 정렬되어있다면 return 하는 선분에서도 정렬됨을 보장함
def generate_available(drawn_lines, whole_points):
    assert [sorted(line) for line in drawn_lines] == drawn_lines, "drawn_line 정렬 안 되어있음"
    assert sorted(whole_points) == whole_points, "whole_points 정렬이 안되어 있음"
    # tuple_drawn_lines = set(tuple(sorted(line)) for line in drawn_lines)
    available = []
    # 존재하는 선인지 체크
    for point1, point2 in combinations(whole_points, 2):
        line_string = LineString([point1, point2])
        if tuple(sorted((point1, point2))) in drawn_lines:
            continue
        flag = False  # 교차 체크
        for l in drawn_lines:
            if len({point1, point2, l[0], l[1]}) == 4:
                if bool(line_string.intersection(LineString(l))):
                    flag = True
                    break
        if flag:
            continue
        flag = False
        for point in whole_points:
            if point == point1 or point == point2:
                continue
            else:
                if bool(line_string.intersection(Point(point))):
                    flag = True
        if flag:
            continue
        available.append([point1, point2])
    assert sorted(available) == available, "available 정렬이 안되어 있음"
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
    lastDrawn_sortedTuple = tuple(lastDrawn_sorted_list)
    available.remove(lastDrawn_sortedTuple)
    line_string = LineString(lastDrawn_sortedTuple)

    for l in available:
        if len(list({lastDrawn_sortedTuple[0], lastDrawn_sortedTuple[1], l[0], l[1]})) == 3:
            continue

        elif bool(line_string.intersection(LineString(l))):
            available.remove(l)
    return available


# inner_point 그대로 가져다 쓰면, check_triangle에서 오류를 범할 수 있으므로, 등호를 추가한 버전은 따로 만들었습니다
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


# 3개의 꼭짓점 정보를 받아서 이것이 삼각형이 되는지 판명하는 함수
# 이 함수는, 3개의 점이 일직선상에 있는지도 판명한다 (일직선상에 있으면 false 반환)
def is_triangle(p1, p2, p3):
    try:
        slope1 = (p1[1] - p2[1]) / (p1[0] - p2[0])
        slope2 = (p3[1] - p2[1]) / (p3[0] - p2[0])

        return slope1 != slope2
    except ZeroDivisionError:
        # Handle the case where the denominator is zero (vertical line)
        return True
    except:
        # Handle other exceptions, e.g., if two points are the same
        return False


# 선위에 짐이 존재하는경우 라인은 어베일러블에서 거르고 홀 라인에서 제외할것
# 삼각형 만들수 있는 갯수를 INT 타입으로 0,1,2로 반환한다.
def check_triangle(line, whole_line, whole_points):
    point1, point2 = line
    l1 = []
    l2 = []
    third_point = []
    for l in whole_line:
        assert type(l) is list
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
                    if debug:  # 디버깅 중이라면
                        showmap(whole_line, whole_points)
                    result_line = whole_line + [line]
                    if debug:
                        showmap(result_line, whole_points)
    return len(third_point)


def get_score_line(drawn_lines, whole_points, available):
    assert available == [sorted(line) for line in available], "available 정렬안됨"
    assert drawn_lines == [sorted(line) for line in drawn_lines], "self.drawn_lines 정렬안됨"
    candiate_triangle = []
    #  라인은 리스트일 것 # 정렬된 상태를 유지할 것
    for line1, line2 in combinations(drawn_lines, 2):
        dots_three = sorted(list({line1[0], line1[1], line2[0], line2[1]}))
        if len(dots_three) != 3:
            continue
        # 안에 점이 있는가?
        flag = False
        for p in whole_points:
            if inner_point(dots_three[0], dots_three[1], dots_three[2], p):
                flag = True
                break
        if flag:
            continue
        point1 = [item for item in dots_three if item not in line1][0]
        point2 = [item for item in dots_three if item not in line2][0]
        candiate_triangle.append(sorted([point1, point2]))
    square = [item for item, count in Counter(candiate_triangle).items() if count > 1]
    return square, candiate_triangle

# 선분과 선분 사이의 선분 중 이로울 확률이 높은 선분 리스트 반환  #각 노드에서 evaluation을 하기 전에 한 번 실행해 주어야 함.
# 이 리스트에 해당하는 선분들은 가점을 부여할 예정
def get_lines_between_lines(whole_line, whole_points, available_line):
    assert type(whole_line) is list, "get_lines_between_lines 의 whole_line 가 리스트가 아님"
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


# 삼각형 내에 점이 하나 있을 때-> 한점은 홀로있고 나머지는 점은 다른선분과 연결되어있는 선의 목록들
# 이 리스트에 해당하는 선분들은 페널티 점수를 부여할 예정
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


# 에러가 발생했을시 available_line 이 업데이트가 되었는지 확인할 것
def evaluation(whole_line, whole_points, available_line):
    assert available_line == generate_available(whole_line, whole_points), "evaluation에서 available_line 입력이상"
    good_lines = get_lines_between_lines(whole_line, whole_points, available_line)
    bad_lines = get_lines_in_triangle(whole_line, whole_points, available_line)
    # print(len(good_lines),good_lines)
    # print(len(bad_lines),bad_lines)
    # print(len(available_line),available_line)
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
