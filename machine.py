import random
from itertools import combinations
from shapely.geometry import LineString, Point

class MACHINE():
    """
        [ MACHINE ]
        MinMax Algorithm을 통해 수를 선택하는 객체.
        - 모든 Machine Turn마다 변수들이 업데이트 됨

        ** To Do **
        MinMax Algorithm을 이용하여 최적의 수를 찾는 알고리즘 생성
           - class 내에 함수를 추가할 수 있음
           - 최종 결과는 find_best_selection을 통해 Line 형태로 도출
               * Line: [(x1, y1), (x2, y2)] -> MACHINE class에서는 x값이 작은 점이 항상 왼쪽에 위치할 필요는 없음 (System이 organize 함)
    """
    
    #유효한 선분인지 검사하는 함수 check_valid_line
    def check_valid_line(self, line):
        #line segment가 삼각형 안에 있거나 선분 위에 있는 경우를 탐지할 것입니다
        def is_point_inside_triangle(p, a, b, c):
            def sign(p1, p2, p3):
                return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

            d1 = sign(p, a, b)
            d2 = sign(p, b, c)
            d3 = sign(p, c, a)

            has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
            has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

            return not (has_neg and has_pos)

        #chain을 사용하는 대신에 sum(self.triangles,[]) 이런식으로 표현함으로서 chain과 같은 역할을 한다
        triangle_points = list(set(sum(self.triangles, [])))
        a, b, c = triangle_points[:2], triangle_points[2:4], triangle_points[4:]

        if not (is_point_inside_triangle(line[0], a, b, c) and is_point_inside_triangle(line[1], a, b, c)):
            return False

        for point in self.whole_points:
            if point in line:
                continue
            if is_point_inside_triangle(point, line[0], line[1], a) or \
               is_point_inside_triangle(point, line[0], line[1], b) or \
               is_point_inside_triangle(point, line[0], line[1], c):
                return False

        return True
    

    def __init__(self, score=[0, 0], drawn_lines=[], whole_lines=[], whole_points=[], location=[]):
        self.id = "MACHINE"
        self.score = [0, 0] # USER, MACHINE
        self.drawn_lines = [] # Drawn Lines
        self.board_size = 7 # 7 x 7 Matrix
        self.num_dots = 0
        self.whole_points = []
        self.location = []
        self.triangles = [] # [(a, b), (c, d), (e, f)]

    def find_best_selection(self):
        available = [[point1, point2] for (point1, point2) in list(combinations(self.whole_points, 2)) if self.check_availability([point1, point2])]
        return random.choice(available)
    
    #기본적인 line condition을 checking하는 함수입니다
    def check_availability(self, line):
        line_string = LineString(line)

        # Must be one of the whole points
        condition1 = (line[0] in self.whole_points) and (line[1] in self.whole_points)
        
        # Must not skip a dot
        condition2 = True
        for point in self.whole_points:
            if point==line[0] or point==line[1]:
                continue
            else:
                if bool(line_string.intersection(Point(point))):
                    condition2 = False

        # Must not cross another line
        condition3 = True
        for l in self.drawn_lines:
            if len(list(set([line[0], line[1], l[0], l[1]]))) == 3:
                continue
            elif bool(line_string.intersection(LineString(l))):
                condition3 = False

        # Must be a new line
        condition4 = (line not in self.drawn_lines)

        if condition1 and condition2 and condition3 and condition4:
            return True
        else:
            return False    

    
