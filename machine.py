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

    def __init__(self, score=[0, 0], drawn_lines=[], whole_lines=[], whole_points=[], location=[]):
        self.id = "MACHINE"
        self.score = [0, 0]  # USER, MACHINE
        self.drawn_lines = []  # Drawn Lines
        self.board_size = 7  # 7 x 7 Matrix
        self.num_dots = 0
        self.whole_points = []
        self.location = []
        self.triangles = []  # [(a, b), (c, d), (e, f)]

    #삼각형을 이루는 선분 3개 중 2개가 이미 그어져 있는 경우를 만드는 상황을 판별하기 위한 함수에요
    #해당 함수에서 반환하는 connected_lines의 length가 1 이상이면 다음 턴에 상대방에게 삼각형을 뺏겨요. (상대방이 하나만 더 그으면 되거든요)
    def must_stealed_point(self, line):
        connected_lines = [l for l in self.drawn_lines if set(line) & set(l)]
        return len(connected_lines)

    # 유효한 선분인지 검사하는 함수 check_valid_line() (+상대방에게 steal 당하지 않도록 하는 최소한의 알고리즘 적용)
    def check_valid_line(self, line):
        # check_availability()로 1차 검사
        # check_availability()로 선분을 그을 수 없는 예외 사항들을 모두 처리할 수 있다고 판단되었습니다.
        if not self.check_availability(line):
            return False
        
        #지금 그을려고 하는 선분을 통해서, must_stealed_point() 함수의 return 값의 길이가 1 이상일 경우
        #즉 그리려고 하는 선분을 통해 삼각형을 이루는 선분 3개 중 2개가 이미 그려지게 만들면
        #이때는 그으면 안되겠죠. 다음 턴에 상대방이 steal할 테니까요.
        if self.must_stealed_point(line) >= 1:
            return False

        #아무런 거 없디면 걍 True를 반환시키죠. check_triangle()에서도 빈 list를 반환하면 random 선택하도록
        return True

    # available은 최악의 상황이 아니면 모두 집어넣고 싶으므로, check_valid_line() 호출
    def find_best_selection(self):
        #최악의 상황은 면할 수 있는 check_valid_line()을 통해서 일단 available_skipWorst list 구성
        available_skipWorst = [[point1, point2] for (point1, point2) in list(combinations(self.whole_points, 2)) if
                     self.check_valid_line([point1, point2])]
        
        #선분이 하나도 연결되어 있지 않은 점이 한개도 남아있지 않은 경우 candidate_line을 available_all에서 뽑아야 할 것이므로.. available_all list 하나 더 생성
        available_all = [[point1, point2] for (point1, point2) in list(combinations(self.whole_points, 2)) if
                     self.check_availability([point1, point2])]
        
        #check_triangle에서 얻은 삼각형을 그릴 수 있는 candidate_line들의 list를 저장합니다.
        #즉, 하나만 더 그으면 삼각형을 만들 수 있는 바로 그 선분들의 list가 candidate_line에 저장됩니다.
        #available_skipWorst에서 하나만 더 그으면 삼각형을 만들 수 있는 list를 못 찾았다면, available_all에서 찾아야 합니다.
        candidate_line = self.check_triangle(available_skipWorst)
        if len(candidate_line) == 0:
            candidate_line = self.check_triangle(available_all)
        
        if len(candidate_line) != 0: #candidate_line 리스트가 비어있지 않다면, 즉 하나만 더 그으면 삼각형이 될 수 있는 상황이 있다면
            #print("하나만 더 그으면 삼각형 획득!, 아래 리스트는 해당 상황에서의 candidate line")
            #print(candidate_line)
            return random.choice(candidate_line) #그들 중에 random 선택
        elif len(available_skipWorst) != 0: #candidate_line 리스트는 비어 있는데, available_skipWorst 리스트는 비어있지 않다면
            #print("그런 선분은 없지만, 최악의 상황은 면할 수 있는 방도는 있음!")
            return random.choice(available_skipWorst)
        else: #available_part 리스트도 비어있다면, 그냥 check_availabilty()로 가능한 모든 선분들 중 random 선택해야 할 것입니다.
            #print("걍 랜덤임!")
            return random.choice(available_all)
        
    #삼각형을 구성할 수 있는 line 집합을 return 해주는 함수 
    #즉, 삼각형의 선분 3개 중 2개의 선분이 그어져 있을 때, 한개만 더 그으면 삼각형 되는데, 그 한개만 더 그으면 되는 선분들의 list를 돌려줌
    def check_triangle(self, available):
        avail_triangle = []
        candiate_triangle = []
        prev_triangle = list(combinations(self.drawn_lines, 2))
        
        for lines in prev_triangle:
            dots_three = list(set([lines[0][0], lines[0][1], lines[1][0], lines[1][1]]))
            if len(dots_three) == 3:
                if [dots_three[0], dots_three[1]] in available or [dots_three[0], dots_three[2]] in available or [
                    dots_three[1], dots_three[2]] in available or [dots_three[1], dots_three[0]] in available or [
                    dots_three[2], dots_three[0]] in available or [dots_three[2], dots_three[1]] in available:
                    flag = True
                    for p in self.whole_points:
                        if inner_point(dots_three[0], dots_three[1], dots_three[2], p):
                            flag = False
                    if flag:
                        if [dots_three[0], dots_three[1]] in available:
                            candiate_triangle.append([dots_three[0], dots_three[1]])
                        elif [dots_three[0], dots_three[2]] in available:
                            candiate_triangle.append([dots_three[0], dots_three[2]])
                        elif [dots_three[1], dots_three[2]] in available:
                            candiate_triangle.append([dots_three[1], dots_three[2]])
                        elif [dots_three[1], dots_three[0]] in available:
                            candiate_triangle.append([dots_three[1], dots_three[0]])
                        elif [dots_three[2], dots_three[0]] in available:
                            candiate_triangle.append([dots_three[2], dots_three[0]])
                        elif [dots_three[2], dots_three[1]] in available:
                            candiate_triangle.append([dots_three[1], dots_three[2]])
        return candiate_triangle

    def check_availability(self, line):
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


def inner_point(point1, point2, point3, point):
    a = ((point2[1] - point3[1]) * (point[0] - point3[0]) + (point3[0] - point2[0]) * (point[1] - point3[1])) / (
                (point2[1] - point3[1]) * (point1[0] - point3[0]) + (point3[0] - point2[0]) * (point1[1] - point3[1]))
    b = ((point3[1] - point1[1]) * (point[0] - point3[0]) + (point1[0] - point3[0]) * (point[1] - point3[1])) / (
                (point2[1] - point3[1]) * (point1[0] - point3[0]) + (point3[0] - point2[0]) * (point1[1] - point3[1]))
    c = 1 - a - b
    if a > 0 and b > 0 and c > 0:
        return True
    else:
        return False
