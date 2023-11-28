import random
from itertools import combinations
from shapely.geometry import LineString, Point

import matplotlib as plt

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

    #내가 득점하면 +1, 상대 득점 -1 (score)
    #현재 상황에서 추가되는 lines: added_lines
    #player에 대한 정보는 따로 필요 없을 것으로 판단'
    #added_lines에는.. available line 추가 / 부모 단에서 available 검증해서 자식에게 보낼 것임
    #민맥스는 후반에 쓸 것인데, avaialble 숫자가 적고, whole_points 다 검사하면.. 속도 너무 느림
    class Node:
        def __init__(self, score, added_lines):
            self.score = score
            self.added_lines = added_lines
            self.children = []
    def __init__(self, score=[0, 0], drawn_lines=[], whole_lines=[], whole_points=[], location=[]):
        self.id = "MACHINE"
        self.score = [0, 0]  # USER, MACHINE
        self.drawn_lines = []  # Drawn Lines
        self.board_size = 7  # 7 x 7 Matrix
        self.num_dots = 0
        self.whole_points = []

        self.available = []
        self.lastState = []
        self.lastDrawn = []

        self.location = []
        self.triangles = []  # [(a, b), (c, d), (e, f)]
        self.isRule = True
        self.isHeurisitic = True
        self.isMinMax = True


    #상태 업데이트
    #self.available: 
    def available_update(self):
        self.lastDrawn = self.drawn_lines - self.lastState
        self.available = self.available - (self.lastDrawn)
        self.intersection_check()

    def intersection_check(self):
        line = self.last_drawn
        line_string = LineString(self.last_drawn)
        for l in self.available:
            if len(list(set([line[0], line[1], l[0], l[1]]))) == 3:
                continue
            elif bool(line_string.intersection(LineString(l))):
                self.available.remove(l) #available 리스트 요소 삭제 및 update

    #선분과 선분 사이의 선분 중 이로울 확률이 높은 선분 리스트 반환
    #각 노드에서 evaluation을 하기 전에 한 번 실행해 주어야 함.
    def get_lines_between_lines(self, node):
        #전체 선분 병합 후 중복 제거
        all_line = node.added_lines + self.drawn_lines
        all_line = list(set(map(tuple, all_line)))
        all_line = [list(i) for i in all_line]
        res = []
        pair_list = list(combinations(all_line, 2))
        for pair in pair_list:
            #두 선분의 점의 수를 세고 4개가 아니면, 즉 각각 떨어져 있는 선분이 아니면 진행하지 않음
            dots = list(set([pair[0][0], pair[0][1], pair[1][0], pair[1][1]]))
            comb_dots = list(combinations(dots, 3))
            if len(dots) == 4:
                #4개의 점으로 그릴 수 있는 선분 6개 중 이미 그려진 2개의 선분 제외 후 4개의 선분을 전부 그릴 수 있는지 판단
                lines = [[pair[0][0], pair[1][0]], [pair[0][0], pair[1][1]], [pair[0][1], pair[1][0]], [pair[0][1], pair[1][1]]]
                #4개의 선분 중 하나라도 그릴 수 없으면 flag가 False가 되고 다음 후보로 진행
                flag = True
                for i in lines:
                    if self.check_availability_node(node, i) == False:
                        flag = False
                #만들게 될 삼각형 사이에 점이 있는지 확인. 하나라도 있으면 flag는 False
                for dot in self.whole_points:
                    for check in comb_dots:
                        if inner_point(check[0], check[1], check[2], dot):
                            flag = False
                            break
                    if flag == False:
                        break
                #flag가 True일 시 네 선분 다 이로울 확률이 높은 선분. res에 추가
                if flag:
                    lines.sort()
                    res += lines
        #중복 제거 후 리스트 반환
        res = list(set(map(tuple, res)))
        res = [list(i) for i in res]
        return res
    
    #삼각형 내에 점이 하나 있을 때, 처음으로 그 점에 그리는 선분들을 반환
    def get_lines_in_triangle(self, node):
        all_line = node.added_lines + self.drawn_lines
        all_line = list(set(map(tuple, all_line)))
        all_line = [list(i) for i in all_line]
        res = []
        #선분 하나와 점 하나를 통해 삼각형이 이미 만들어져 있는지, 삼각형 내에 점이 하나 있는지, 점이 하나 있을 때 선분이 하나도 없는지 비교 후 모두 만족하면 세 선분을 res에 추가
        for line in all_line:
            for dot in self.whole_points:
                if dot in line:
                    continue
                if ([dot, line[0]] not in all_line and [line[0], dot] not in all_line) or ([dot, line[1]] not in all_line and [line[1], dot] not in all_line):
                    continue
                cnt = 0
                only_dot = None
                for inner_dot in self.whole_points:
                    if inner_dot==dot or inner_dot in line:
                        continue
                    if inner_point(dot, line[0], line[1], inner_dot):
                        cnt += 1
                        only_dot = inner_dot
                if cnt != 1:
                    continue
                flag = False
                dots = line + [dot]
                lines = []
                for i in dots:
                    tmp = [only_dot, i]
                    tmp.sort()
                    lines += [tmp]
                    if tmp in all_line:
                        flag = True
                        break
                if not flag:
                    res += lines
        res = list(set(map(tuple, res)))
        res = [list(i) for i in res]
        return res
    
    #두 선분 리스트를 각 노드에서 미리 구한 뒤 evaluation 함수에 매개변수로 넣어주어야 함
    #lines_bt : get_lines_between_lines()으로 구한 선분 리스트
    #lines_tri : get_lines_triangle()으로 구한 선분 리스트
    #각 선분 리스트 내에 line이 존재하는지 비교 후 반환
    def evaluation(self, node, lines_bt, lines_tri, line):
        #이미 존재하는 선분이면 -1
        line.sort()
        if line in self.drawn_lines or line in node.added_lines:
            return -1
        #선분 사이 선분 중 이로운 선분들 리스트에 선분이 있을 때
        if line in lines_bt:
            return 1
        #삼각형 안에 점이 하나 있을 때 최초로 그리는 선분이면
        if line in lines_tri:
            return 1
        #이외는 -1 반환
        return -1



    #알파베타 컷오프는 병렬 안됨 / 모듈화 필요
    #상대가 한 번에 2개의 삼각형 득점하는 경우: -무한대 / 우리가 한 번에 2개의 삼각형 득점: +무한대
    #한번에 2개의 삼각형을 얻는 것이 무한대의 이점이라고 우선 가정
    #더 이상 트리를 확장하지 않는 것으로 탐색해야 하는 범위를 축소
    #자료구조는 무엇으로? ->
    # def min_max(self):
        

    #삼각형을 이루는 선분 3개 중 2개가 이미 그어져 있는 경우를 만드는 상황을 판별하기 위한 함수에요
    #해당 함수에서 반환하는 connected_lines의 length가 1 이상이면 다음 턴에 상대방에게 삼각형을 뺏겨요. (상대방이 하나만 더 그으면 되거든요)
    #avaiablae에서 한 선분을 뽑는다면, 선분에 양쪽 끝에 점이 있지 / 그 점에 연결된 선분들을 check를 해서 양쪽 점이 공통으로 가지고 있는 점이 일치하는 점이 있는지 확인 / 그 선분을 추가했을 때, check_triangle() 해서 True 반환하면, must_stealed_point()에서 True 반환
    def check_stealing_situation_inOpponentTurn(self, line):
        connected_lines = [l for l in self.drawn_lines if set(line) & set(l)]

        #애초에 점이 겹치는 게 없으면 훔칠 기회가 없다
        if len(connected_lines) < 1:
            return False

        #위에서 1보다 크거나 같게 나오면 삼각형의 3개의 선분 중 2개의 선분이 연결되는 경우이므로 중복되는 점(두 선분이 겹치는 점)을 제외한 점들의 값을 알아내야 한다
        #그 2개의 점의 좌표는 상대방 턴에 그을 선분의 양 끝 점의 좌표이다(steal 할 수 있는 경우라고 볼 수 있지)
        #그런데, 그 선분을 긋고, 삼각형 안을 확인해보니 점이 1개 이상 있다고 하면, 그러면 steal 못하니까, 걍 그어도 상관 없다!
        #connected_lines는 len이 1 이상일 수 있으니까, for문을 돌린다
        for connected_line in connected_lines:

            #그냥 all_points_set_withDuplicate
            all_points_set_withDuplicate = list([connected_line[0], connected_line[1], line[0], line[1]])
            #중복 제거를 위해 set으로 함
            all_points_set_nonDuplicate = set([connected_line[0], connected_line[1], line[0], line[1]])
            
            #중복된 값 찾기
            overlapping_point = [item for item in all_points_set_nonDuplicate if all_points_set_withDuplicate.count(item) > 1]
            
            if not overlapping_point:
                # Handle the case when overlapping_point is None (아마 초반에만 이거에 걸러질거임)
                return False
            
            #겹치는 point를 제외한 점들의 좌표(2개가 되겠죠)를 따로 저장한다
            non_overlapping_points = list(all_points_set_nonDuplicate - {tuple(overlapping_point)})
            
            #만약에 non_overlapping_points와 overlapping_point를 이용해서 만들어진 삼각형 내부에 self.whole_points에 있는 점이 1개 이상 있는게 판명되면 False return
            for point in self.whole_points:
                #inner_point 함수로 3개의 점을 기준으로 이루어진 삼각형 내부에 whole_points에 포함되는 점이 있는 지 확인 가능
                if inner_point(overlapping_point, non_overlapping_points[0], non_overlapping_points[1], point):
                    return False

        return True


    # 유효한 선분인지 검사하는 함수 check_valid_line() (+상대방에게 steal 당하지 않도록 하는 최소한의 알고리즘 적용)
    def check_valid_line(self, line):
        # check_availability()로 1차 검사
        # check_availability()로 선분을 그을 수 없는 예외 사항들을 모두 처리할 수 있다고 판단되었습니다.
        if not self.check_availability(line):
            return False
        
        #지금 그을려고 하는 선분을 통해서, check_stealing_situation_inOpponentTurn() 함수의 return 값의 길이가 1 이상일 경우
        #즉 그리려고 하는 선분을 통해 삼각형을 이루는 선분 3개 중 2개가 이미 그려지게 만들면
        #이때는 그으면 안되겠죠. 다음 턴에 상대방이 steal할 테니까요.
        if self.check_stealing_situation_inOpponentTurn(line) == True:
            return False

        #아무런 거 없디면 걍 True를 반환시키죠. check_triangle()에서도 빈 list를 반환하면 random 선택하도록
        return True

    # available은 최악의 상황이 아니면 모두 집어넣고 싶으므로, check_valid_line() 호출
    def find_best_selection(self):
        if self.isRule:
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
                #print("[available_skipworst]")
                #print(available_skipWorst)
                #print("그런 선분은 없지만, 최악의 상황은 면할 수 있는 방도는 있음!")
                return random.choice(available_skipWorst)
            else: #available_part 리스트도 비어있다면, 그냥 check_availabilty()로 가능한 모든 선분들 중 random 선택해야 할 것입니다.
                #print("걍 랜덤임!")

                #evaluation 함수 테스트용
                node = self.Node(None, self.drawn_lines) #노드 임의로 지정
                lines_bt = self.get_lines_between_lines(node)
                lines_tri = self.get_lines_in_triangle(node)
                for i in available_all:
                    if self.evaluation(node, lines_bt, lines_tri , i) > 0 :
                        return i

                #self.isRule = False
                return random.choice(available_all)
        
        # elif self.isHeurisitic:
        #     #휴리스틱을 사용할 것임
        
        # elif self.isMinMax:
        #     #minmax 알고리즘을 사용할 것임
        
        
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
        
    #heuristic()에서 노드용으로 사용할 함수 
    def check_availability_node(self, node, line):
        line_string = LineString(line)
        cur_lines = self.drawn_lines + node.added_lines
        
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
        for l in cur_lines:
            if len(list(set([line[0], line[1], l[0], l[1]]))) == 3:
                continue
            elif bool(line_string.intersection(LineString(l))):
                condition3 = False

        # Must be a new line
        condition4 = (line not in cur_lines)

        if condition1 and condition2 and condition3 and condition4:
            return True
        else:
            return False

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
