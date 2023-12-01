import random

from shapely import Polygon
from shapely.geometry import LineString, Point
from itertools import product, chain, combinations
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


    #평가함수. 민맥스트리에서 시간 내에 최적해를 구하지 못했을 때 사용. 당장 만들 수 있는 삼각형이 없다고 가정.
    #직선과 점을 연결하는 경우는 피한다. 직선과 직선을 연결하는 경우, 각 두 점이 서로 연결될 수 있는지, 즉 4개의 선분이 전부 가능한지 확인한다.
    def evaluation(self):
        #이미 그려져있는 선분 중 2개씩 선택
        pair_list = list(combinations(self.drawn_lines, 2))
        for pair in pair_list:
            #두 선분의 점의 수를 세고 4개가 아니면, 즉 각각 떨어져 있는 선분이 아니면 진행하지 않음
            dot_cnt = list(set([pair[0][0], pair[0][1], pair[1][0], pair[1][1]]))
            if len(dot_cnt) == 4:
                #4개의 점으로 그릴 수 있는 선분 6개 중 이미 그려진 2개의 선분 제외 후 4개의 선분을 전부 그릴 수 있는지 판단
                lines = [[pair[0][0], pair[1][0]], [pair[0][0], pair[1][1]], [pair[0][1], pair[1][0]], [pair[0][1], pair[1][1]]]
                #4개의 선분 중 하나라도 그릴 수 없으면 falg가 False가 되고 다음 후보로 진행
                flag = True
                for i in lines:
                    if self.check_availability(i) == False:
                        flag = False
                    #print("i :", i, "flag :", flag)
                #flag가 True일 시 네 선분 다 그릴 수 있는 선분임. 이중 하나 리턴
                if flag:
                    return list(lines[0])
        #가능한 선이 없을 시 -1 리턴
        return -1

    #알파베타 컷오프는 병렬 안됨 / 모듈화 필요
    #상대가 한 번에 2개의 삼각형 득점하는 경우: -무한대 / 우리가 한 번에 2개의 삼각형 득점: +무한대
    #한번에 2개의 삼각형을 얻는 것이 무한대의 이점이라고 우선 가정
    #더 이상 트리를 확장하지 않는 것으로 탐색해야 하는 범위를 축소
    #자료구조는 무엇으로? ->


    #삼각형을 이루는 선분 3개 중 2개가 이미 그어져 있는 경우를 만드는 상황을 판별하기 위한 함수에요
    #해당 함수에서 반환하는 connected_lines의 length가 1 이상이면 다음 턴에 상대방에게 삼각형을 뺏겨요. (상대방이 하나만 더 그으면 되거든요)
    #available에서 한 선분을 뽑는다면, 선분에 양쪽 끝에 점이 있지 / 그 점에 연결된 선분들을 check를 해서 양쪽 점이 공통으로 가지고 있는 점이 일치하는 점이 있는지 확인 / 그 선분을 추가했을 때, check_triangle() 해서 True 반환하면, must_stealed_point()에서 True 반환
    def check_stealing_situation_inOpponentTurn(self, line):
        connected_lines = [l for l in self.drawn_lines if set(line) & set(l)]

        #애초에 점이 겹치는 게 없으면 훔칠 기회가 없다
        if len(connected_lines) < 1:
            return False

        #삼각형을 뺏길 수 있는 상황이 1번이라도 연출되면 아래 isDangerous 값은 True로 바뀔 것이다
        isDangerous = False

        #위에서 1보다 크거나 같게 나오면 삼각형의 3개의 선분 중 2개의 선분이 연결되는 경우이므로 중복되는 점(두 선분이 겹치는 점)을 제외한 점들의 값을 알아내야 한다
        #그 2개의 점의 좌표는 상대방 턴에 그을 선분의 양 끝 점의 좌표이다(steal 할 수 있는 경우라고 볼 수 있지)
        #그런데, 그 선분을 긋고, 삼각형 안을 확인해보니 점이 1개 이상 있다고 하면, 그러면 steal 못하니까, 걍 그어도 상관 없다!
        #connected_lines는 len이 1 이상일 수 있으니까, for문을 돌린다
        for connected_line in connected_lines:

            #그냥 all_points_set_withDuplicate
            all_points_set_withDuplicate = list([connected_line[0], connected_line[1], line[0], line[1]])
            #중복 제거를 위해 set으로 함
            all_points_set_nonDuplicate = set([connected_line[0], connected_line[1], line[0], line[1]])
            
            #중복된 값 찾기 (tuple을 넘겨줄 것임)
            overlapping_point = [item for item in all_points_set_nonDuplicate if all_points_set_withDuplicate.count(item) > 1][0]
            
            if not overlapping_point:
                # Handle the case when overlapping_point is None (아마 초반에만 이거에 걸러질거임)
                return False
            
            #겹치는 point를 제외한 점들의 좌표(2개가 되겠죠)를 따로 저장한다 (overlapping_point는 1개만 나오는 것이 정상이므로, for문 돌릴 필요가 없습니다)
            all_points_set_nonDuplicate.discard(overlapping_point)
            non_overlapping_points = list(all_points_set_nonDuplicate)

            #3개의 점으로 삼각형을 만들 수 있는 경우에만 내부에 점이 있는지 판명하면 된다
            #3개의 점으로 삼각형을 안 만든다면 그냥 넘긴다(continue)
            #삼각형을 만드는 조건은 맞지만, self.check_availability를 통해, 상대방이 긋는 선이 유효하기까지 해야 아래의 조건들을 확인하는 게 의미가 있다
            if is_triangle(overlapping_point, non_overlapping_points[0], non_overlapping_points[1]) and self.check_availability([non_overlapping_points[0],non_overlapping_points[1]]):
                #만약에 non_overlapping_points와 overlapping_point를 이용해서 만들어진 삼각형 내부에 self.whole_points에 있는 점이 1개 이상 있는게 판명되면 그 경우는 삼각형으로 인정 안된다
                for point in self.whole_points:
                    #point가 그으려는(혹은 이미 그어진) 선분에 포함되는 3개의 점이라면 검사할 필요가 없다
                    if point == overlapping_point or point == non_overlapping_points[0] or point == non_overlapping_points[1]:
                        continue
                    #inner_point_usingInStealChecking 함수로 3개의 점을 기준으로 이루어진 삼각형 내부에 whole_points에 포함되는 점이 있는 지 확인 가능
                    #일직선 상에 있는 점도 판명 가능(예외처리를 위함)
                    #다음 connected_line과의 확인을 위해 continue
                    elif inner_point_usingInStealChecking(overlapping_point, non_overlapping_points[0], non_overlapping_points[1], point):
                        break
                    else: #inner_point에서 내부에 점이 없다는 것이 판명되면
                        isDangerous = True #실점할 수 있는 위험한 상황으로 판명
            else:
                continue

        #삼각형을 상대방에게 뺏기는 상황이 한번이라도 발생했다면, isDangerous는 True를 반환했을 것이다
        #그것을 확인하고, 값이 일치하면 False 반환, 아니면 최종적으로 True 반환
        if isDangerous == False:
            return False
        else:             
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
        if self.isHeurisitic:
            self.minmax()
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
                print("하나만 더 그으면 삼각형 획득!, 아래 리스트는 해당 상황에서의 candidate line")
                return random.choice(candidate_line) #그들 중에 random 선택
            elif len(available_skipWorst) != 0: #candidate_line 리스트는 비어 있는데, available_skipWorst 리스트는 비어있지 않다면
                print(available_skipWorst)
                print("그런 선분은 없지만, 최악의 상황은 면할 수 있는 방도는 있음!")
                return random.choice(available_skipWorst)
            else: #available_part 리스트도 비어있다면, 그냥 check_availabilty()로 가능한 모든 선분들 중 random 선택해야 할 것입니다.
                print("걍 랜덤임!")
                eval_line = self.evaluation()
                if eval_line != -1:
                    return eval_line
                
                # self.isRule = False
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
        for lines in prev_triangle[:]:
            dots_three = list({lines[0][0], lines[0][1], lines[1][0], lines[1][1]})
            if len(dots_three) != 3:
                prev_triangle.remove(lines)
        for lines in prev_triangle:
            dots_three = list({lines[0][0], lines[0][1], lines[1][0], lines[1][1]})
            bitFlag=0
            if [dots_three[0], dots_three[1]] in available:
                bitFlag=bitFlag|(1<<0)
            if [dots_three[0], dots_three[2]] in available:
                bitFlag=bitFlag|(1<<1)
            if [dots_three[1], dots_three[2]] in available:
                bitFlag=bitFlag|(1<<2)
            if [dots_three[1], dots_three[0]] in available:
                bitFlag=bitFlag|(1<<3)
            if [dots_three[2], dots_three[0]] in available:
                bitFlag=bitFlag|(1<<4)
            if [dots_three[2], dots_three[1]] in available:
                bitFlag=bitFlag|(1<<5)
            if bitFlag!=0:
                flag = True
                for p in self.whole_points:
                    if inner_point(dots_three[0], dots_three[1], dots_three[2], p):
                        flag = False
                if flag:
                    if bitFlag&(1<<0):
                        candiate_triangle.append([dots_three[0], dots_three[1]])
                    elif bitFlag&(1<<1):
                        candiate_triangle.append([dots_three[0], dots_three[2]])
                    elif bitFlag&(1<<2):
                        candiate_triangle.append([dots_three[1], dots_three[2]])
                    elif bitFlag&(1<<3):
                        candiate_triangle.append([dots_three[1], dots_three[0]])
                    elif bitFlag&(1<<4):
                        candiate_triangle.append([dots_three[2], dots_three[0]])
                    elif bitFlag&(1<<5):
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
            if len(list({line[0], line[1], l[0], l[1]})) == 3:
                continue
            elif bool(line_string.intersection(LineString(l))):
                condition3 = False

        # Must be a new line
        condition4 = (line not in self.drawn_lines)

        if condition1 and condition2 and condition3 and condition4:
            return True
        else:
            return False

def available_update(available, lastDrawn):
    available = available - lastDrawn
    line_string = LineString(lastDrawn)
    for l in available:
        if len(list({lastDrawn[0], lastDrawn[1], l[0], l[1]})) == 3:
            continue
        elif bool(line_string.intersection(LineString(l))):
            available.remove(l)
    return available

def inner_point(point1, point2, point3, point):
    try:
        a=((point2[1]-point3[1])*(point[0]-point3[0])+(point3[0]-point2[0])*(point[1]-point3[1]))/((point2[1]-point3[1])*(point1[0]-point3[0])+(point3[0]-point2[0])*(point1[1]-point3[1]))
        b=((point3[1]-point1[1])*(point[0]-point3[0])+(point1[0]-point3[0])*(point[1]-point3[1]))/((point2[1]-point3[1])*(point1[0]-point3[0])+(point3[0]-point2[0])*(point1[1]-point3[1]))
    except:
        return False
    c=1-a-b
    if a>0 and b>0 and c>0:
        return True
    else:
        return False

#inner_point 그대로 가져다 쓰면, check_triangle에서 오류를 범할 수 있으므로, 등호를 추가한 버전은 따로 만들었습니다
def inner_point_usingInStealChecking(point1, point2, point3, point):
    try:
        a=((point2[1]-point3[1])*(point[0]-point3[0])+(point3[0]-point2[0])*(point[1]-point3[1]))/((point2[1]-point3[1])*(point1[0]-point3[0])+(point3[0]-point2[0])*(point1[1]-point3[1]))
        b=((point3[1]-point1[1])*(point[0]-point3[0])+(point1[0]-point3[0])*(point[1]-point3[1]))/((point2[1]-point3[1])*(point1[0]-point3[0])+(point3[0]-point2[0])*(point1[1]-point3[1]))
    except:
        return False
    c=1-a-b

    if 0 <= a <= 1 and 0 <= b <= 1 and 0 <= c <= 1:
        return True
    else:
        return False
    
#3개의 꼭짓점 정보를 받아서 이것이 삼각형이 되는지 판명하는 함수
#이 함수는, 3개의 점이 일직선상에 있는지도 판명한다 (일직선상에 있으면 false 반환)
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
    point1 = line[0]
    point2 = line[1]
    point1_connected = []
    point2_connected = []
    for l in whole_line:
        if point1 in l:
            point1_connected.append(l)
        if point2 in l:
            point2_connected.append(l)
    third_point = []
    if point1_connected and point2_connected:  # 최소한 2점 모두 다른 선분과 연결되어 있어야 함
        for line1, line2 in product(point1_connected, point2_connected):
            if line1.remove(point1) == line2.remove(point2):
                target_point = line1.remove(point1)
                triangle = [point1,point2,target_point]
                empty = True
                for point in whole_points:
                    if point in triangle:
                        continue
                    if bool(Polygon(triangle).intersection(Point(point))):
                        empty = False
                if empty:
                    third_point.append(target_point)
    return len(third_point)

def evaluation_func():
    pass # 성환님 꺼 가져와서 만들것

def alpha_pruning():
    pass

def beta_pruning():

class Node():
    #먼저 초기화할 때, alpha, beta 값들을 추가하자.
    def __init__(self, added_line = None, parent = None, alpha=float('-inf'), beta=float('inf') ):
        # self.ab_value = 0 (알파베타 가지치기용 isOpponent에 따라서 알파값인지 베타값인지 결정)
        self.alpha = alpha
        self.beta = beta

        if parent is not None:
            self.added_line = added_line #추가한 line (이번 turn에 그릴 line)
            self.total_lines = parent.total_lines
            self.whole_points = parent.whole_points
            self.isOpponentTurn = not parent.isOpponentTurn
            if self.isOpponentTurn:
                self.score = parent.score + check_triangle(added_line,self.total_lines,self.whole_points)
            else:
                self.score = parent.score - check_triangle(added_line,self.total_lines,self.whole_points)
            self.available = available_update(parent.available, added_line)
        else: #루트노드
            self.score = 0
            self.isOpponentTurn = False
            

    def expand_node(self, depth_limit, alpha, beta):

        #-1 ~ +1 사이에 값으로 eval_func 값을 정규화한다고 가정한다고 +1, -1로 설정
        if self.isOpponentTurn:
            score = self.score + 1 #상대방 turn일 때에는 자신이 득점할 수 없다
        else:
            score = self.score - 1 #내 turn에는 실점할 수 없다

        if len(self.available) == 0: #게임 끝났다는 의미
            return self.score, self.added_line
        
        if depth_limit == 0:
            return evaluation_func, self.added_line # 평가함수 적용할 계획
        
        else:
            for l in self.available:
                child = Node(l, parent=self)
                child_score = child.expand_node(depth_limit - 1, self.alpha, self.beta)
                if self.isOpponentTurn: #상대방 turn일 때 (minimize-player가 되야 하는 경우에)
                    if score > child_score:
                        score = child_score
                        target_line = l
                        pruning = alpha_pruning() #True, false로 반환 (pruning이 가능할 경우에만 True 반환)
                        if pruning:
                            break

                        beta = min(beta, score)
                        if score <= alpha:
                            break

                else: #내 turn일 때 (maximize-player가 되야 하는 경우에)
                    if score < child_score:
                        score = child_score
                        target_line = l
                        pruning = beta_pruning() #True, false로 반환 (pruning이 가능할 경우에만 True 반환)
                        if pruning:
                            break

                        alpha = max(alpha, score)
                        if score >= beta:
                            break

        return self.score , target_line , pruning