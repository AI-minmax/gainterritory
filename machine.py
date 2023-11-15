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
    
    #유효한 선분인지 검사하는 함수 check_valid_line() (+상대방에게 steal 당하지 않도록 하는 최소한의 알고리즘 적용) 
    def check_valid_line(self, line):
        #check_availability()로 1차 검사
        #check_availability()로 선분을 그을 수 없는 예외 사항들을 모두 처리할 수 있다고 판단되었습니다. (+check_availability 약간 수정)
        if not self.check_availability(line):
            return False
        
        #그을려고 하는 선이 기존에 존재하는 삼각형과 함께 삼각형을 생성하는지 확인해봐야 합니다.
        #만약 그을려고 하는 선이 기존에 존재하는 삼각형과 함께 삼각형을 생성하게 되버린다면, False를 반환해야 할 것입니다
        new_triangle = LineString(line).intersection(LineString(self.triangle_points))
        if not new_triangle.is_empty:
            return False
        
        #선이 두 개의 기존 선에 연결되어 잠재적인 삼각형이 만들어 지는지 확인해봐야 합니다.
        #만약 그을려고 하는 선이 기존에 존재하는 삼각형과 함께 삼각형을 생성하게 되버린다면, False를 반환해야 할 것입니다
        connected_lines = [l for l in self.drawn_lines if set(line) & set(l)]
        if len(connected_lines) == 2:
            triangle = LineString(connected_lines[0] + connected_lines[1]).intersection(LineString(line))
            if not triangle.is_empty:
                return False
    

    def __init__(self, score=[0, 0], drawn_lines=[], whole_lines=[], whole_points=[], location=[]):
        self.id = "MACHINE"
        self.score = [0, 0] # USER, MACHINE
        self.drawn_lines = [] # Drawn Lines
        self.board_size = 7 # 7 x 7 Matrix
        self.num_dots = 0
        self.whole_points = []
        self.location = []
        self.triangles = [] # [(a, b), (c, d), (e, f)]

    #해당 함수에서 check_availablity() 대신, check_valid_line() 사용
    def find_best_selection(self):
        available = [[point1, point2] for (point1, point2) in list(combinations(self.whole_points, 2)) if self.check_valid_line([point1, point2])]
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

        #삼각형의 3개의 선분 중 2개의 선분이 연결되게 되면, steal 당할 확률이 높아지겠죠. (원래 함수에서 추가 부분)
        #그런 상황을 방지하기 위해 애초에 3개의 삼각형 선분들 중 2개의 선분이 연결되면 False(unavailable 의미) 반환
        connected_lines = [l for l in self.drawn_lines if set(line) & set(l)]
        if len(connected_lines) == 2:
            return False

        if condition1 and condition2 and condition3 and condition4:
            return True
        else:
            return False    

    
