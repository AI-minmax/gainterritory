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
        self.score = [0, 0] # USER, MACHINE
        self.drawn_lines = [] # Drawn Lines
        self.board_size = 7 # 7 x 7 Matrix
        self.num_dots = 0
        self.whole_points = []
        self.location = []
        self.triangles = [] # [(a, b), (c, d), (e, f)]

    def find_best_selection(self):
        available = [[point1, point2] for (point1, point2) in list(combinations(self.whole_points, 2)) if self.check_availability([point1, point2])]
        candiate_line=self.check_triangle(available)
        #print(available)
        #print(candiate_line)
        if len(candiate_line)!=0:
            return random.choice(candiate_line)
        else:
            return random.choice(available)

    def check_triangle(self, available):
        avail_triangle=[]
        candiate_triangle=[]
        prev_triangle=list(combinations(self.drawn_lines,2))
        #print("prev :",prev_triangle)
        for lines in prev_triangle:
            dots_three=list(set([lines[0][0],lines[0][1],lines[1][0],lines[1][1]]))
            if len(dots_three)==3:
                #print("dots_three :",dots_three)
                #avail_triangle.append([lines[0][0],lines[0][1],lines[1][0],lines[1][1]])
                if [dots_three[0],dots_three[1]] in available or [dots_three[0],dots_three[2]] in available or [dots_three[1],dots_three[2]] in available or [dots_three[1],dots_three[0]] in available or [dots_three[2],dots_three[0]] in available or [dots_three[2],dots_three[1]] in available:
                    flag=True
                    for p in self.whole_points:
                        if inner_point(dots_three[0],dots_three[1],dots_three[2],p):
                            flag=False
                    if flag:
                        if [dots_three[0],dots_three[1]] in available:
                            candiate_triangle.append([dots_three[0],dots_three[1]])
                        elif [dots_three[0],dots_three[2]] in available:
                            candiate_triangle.append([dots_three[0],dots_three[2]])
                        elif [dots_three[1],dots_three[2]] in available:
                            candiate_triangle.append([dots_three[1],dots_three[2]])
                        elif [dots_three[1],dots_three[0]] in available:
                            candiate_triangle.append([dots_three[1],dots_three[0]])
                        elif [dots_three[2],dots_three[0]] in available:
                            candiate_triangle.append([dots_three[2],dots_three[0]])
                        elif [dots_three[2],dots_three[1]] in available:
                            candiate_triangle.append([dots_three[1],dots_three[2]])
        return candiate_triangle
    
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
        
def inner_point(point1, point2, point3, point):
    a=((point2[1]-point3[1])*(point[0]-point3[0])+(point3[0]-point2[0])*(point[1]-point3[1]))/((point2[1]-point3[1])*(point1[0]-point3[0])+(point3[0]-point2[0])*(point1[1]-point3[1]))
    b=((point3[1]-point1[1])*(point[0]-point3[0])+(point1[0]-point3[0])*(point[1]-point3[1]))/((point2[1]-point3[1])*(point1[0]-point3[0])+(point3[0]-point2[0])*(point1[1]-point3[1]))
    c=1-a-b
    if a>0 and b>0 and c>0:
        return True
    else:
        return False

    
