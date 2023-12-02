import shapely
from shapely import Polygon
from shapely.geometry import LineString, Point
from itertools import product, chain, combinations
import matplotlib.pyplot as plt

debug = True

def generate_available(drawn_lines, whole_points):
    tuple_drawn_lines = []
    for line in drawn_lines:
        # tuple_line = tuple(line)
        sorted_line_list = sorted(line)
        tuple_line = tuple(sorted_line_list)
        tuple_drawn_lines.append(tuple_line)
    available = []

    # 존재하는 선인지 체크
    for point1, point2 in combinations(whole_points,2):
        line_string = LineString([point1, point2])
        if (point1, point2) in tuple_drawn_lines:
            continue
        flag = False # 교차 체크
        for l in tuple_drawn_lines:
            if len(list({tuple(point1), tuple(point2), tuple(l[0]), tuple(l[1])})) == 4:
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
                    flag= True
        if flag:
            continue

        #여기도 정렬 필요
        available_line_add_list = sorted((point1, point2))
        returnToTuple = tuple(available_line_add_list)
        available.append(returnToTuple)
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
                    if point in [point1,point2,element]:
                        continue
                    if inner_point(point1,point2,element,point):
                        innerpoint = False
                if innerpoint:
                    third_point.append(element)
                    if debug:  # 디버깅 중이라면
                        showmap(whole_line,whole_points)
                    result_line = whole_line + [line]
                    if debug:
                        showmap(result_line,whole_points)
    return len(third_point)



def is_point_inside_half_plane(start, end, point, apex):
    return (end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (point[0] - start[0]) >= 0


#선분과 선분 사이의 선분 중 이로울 확률이 높은 선분 리스트 반환
    #각 노드에서 evaluation을 하기 전에 한 번 실행해 주어야 함.
def get_lines_between_lines(whole_line,whole_points,available_line):
    assert type(whole_line) is list, "get_lines_between_lines 의 whole_line 가 리스트가 아님"
    res = []
    for line1, line2 in list(combinations(whole_line, 2)):
        #두 선분의 점의 수를 세고 4개가 아니면, 즉 각각 떨어져 있는 선분이 아니면 진행하지 않음
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
            for line in list(combinations([line1[0], line1[1], line2[0], line2[1]],2)):
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


#삼각형 내에 점이 하나 있을 때-> 한점은 홀로있고 나머지는 점은 다른선분과 연결되어있는 선의 목록들
# 이 리스트에 해당하는 선분들은 네거티브 점수를 부여할 예정
def get_lines_in_triangle(whole_line,whole_points,available_line):
    alone_points = whole_points.copy()
    for point1,point2 in whole_line:
        try:
            alone_points.remove(point1)
        except:
            pass
        try:
            alone_points.remove(point2)
        except:
            pass
    target_line = []
    for point1,point2 in available_line:
        if point1 in alone_points or point2 in alone_points:
            if point1 in alone_points and point2 in alone_points:
                continue
            else:
                target_line.append([point1,point2])
    return target_line

#두 선분 리스트를 각 노드에서 미리 구한 뒤 evaluation 함수에 매개변수로 넣어주어야 함
#lines_bt : get_lines_between_lines()으로 구한 선분 리스트
#lines_tri : get_lines_triangle()으로 구한 선분 리스트
#각 선분 리스트 내에 line이 존재하는지 비교 후 반환
def evaluation(whole_line, whole_points, available_line):
    good_lines = get_lines_between_lines(whole_line, whole_points, available_line)
    bad_lines = get_lines_in_triangle(whole_line, whole_points, available_line)
    print(len(good_lines),good_lines)
    print(len(bad_lines),bad_lines)
    print(len(available_line),available_line)
    score = (len(good_lines) - len(bad_lines))/(len(available_line)+1)
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