import random
from tools import generate_available, inner_point, available_update, inner_point_usingInStealChecking, is_triangle, check_triangle, evaluation, showmap
from shapely.geometry import LineString
from itertools import combinations


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

    # 내가 득점하면 +1, 상대 득점 -1 (score)
    # 현재 상황에서 추가되는 lines: added_lines
    # player에 대한 정보는 따로 필요 없을 것으로 판단'
    # added_lines에는.. available line 추가 / 부모 단에서 available 검증해서 자식에게 보낼 것임
    # 민맥스는 후반에 쓸 것인데, avaialble 숫자가 적고, whole_points 다 검사하면.. 속도 너무 느림

    def __init__(self, score=[0, 0], drawn_lines=[], whole_lines=[], whole_points=[], location=[]):
        self.id = "MACHINE"
        self.score = [0, 0]  # USER, MACHINE
        self.drawn_lines = []  # Drawn Lines
        self.board_size = 7  # 7 x 7 Matrix
        self.num_dots = 0
        self.whole_points = []
        self.available = generate_available(self.drawn_lines,self.whole_points)
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
        for line in self.drawn_lines:
            if line not in self.lastState:
                self.lastDrawn = line
                break
        self.intersection_check()

    def intersection_check(self):
        line = self.last_drawn
        line_string = LineString(self.last_drawn)
        for l in self.available:
            if len(list({line[0], line[1], l[0], l[1]})) == 3:
                continue
            elif bool(line_string.intersection(LineString(l))):
                self.available.remove(l) #available 리스트 요소 삭제 및 update


    # 삼각형을 이루는 선분 3개 중 2개가 이미 그어져 있는 경우를 만드는 상황을 판별하기 위한 함수에요
    # 해당 함수에서 반환하는 connected_lines의 length가 1 이상이면 다음 턴에 상대방에게 삼각형을 뺏겨요. (상대방이 하나만 더 그으면 되거든요)
    # available에서 한 선분을 뽑는다면, 선분에 양쪽 끝에 점이 있지 / 그 점에 연결된 선분들을 check를 해서 양쪽 점이 공통으로 가지고 있는 점이 일치하는 점이 있는지 확인 / 그 선분을 추가했을 때, check_triangle() 해서 True 반환하면, must_stealed_point()에서 True 반환
    def check_stealing_situation_inOpponentTurn(self, line):
        connected_lines = [l for l in self.drawn_lines if set(line) & set(l)]

        # 애초에 점이 겹치는 게 없으면 훔칠 기회가 없다
        if len(connected_lines) < 1:
            return False

        # 삼각형을 뺏길 수 있는 상황이 1번이라도 연출되면 아래 isDangerous 값은 True로 바뀔 것이다
        isDangerous = False

        # 위에서 1보다 크거나 같게 나오면 삼각형의 3개의 선분 중 2개의 선분이 연결되는 경우이므로 중복되는 점(두 선분이 겹치는 점)을 제외한 점들의 값을 알아내야 한다
        # 그 2개의 점의 좌표는 상대방 턴에 그을 선분의 양 끝 점의 좌표이다(steal 할 수 있는 경우라고 볼 수 있지)
        # 그런데, 그 선분을 긋고, 삼각형 안을 확인해보니 점이 1개 이상 있다고 하면, 그러면 steal 못하니까, 걍 그어도 상관 없다!
        # connected_lines는 len이 1 이상일 수 있으니까, for문을 돌린다
        for connected_line in connected_lines:

            # 그냥 all_points_set_withDuplicate
            all_points_set_withDuplicate = list([connected_line[0], connected_line[1], line[0], line[1]])
            # 중복 제거를 위해 set으로 함
            all_points_set_nonDuplicate = set([connected_line[0], connected_line[1], line[0], line[1]])

            # 중복된 값 찾기 (tuple을 넘겨줄 것임)
            overlapping_point = \
            [item for item in all_points_set_nonDuplicate if all_points_set_withDuplicate.count(item) > 1][0]

            if not overlapping_point:
                # Handle the case when overlapping_point is None (아마 초반에만 이거에 걸러질거임)
                return False

            # 겹치는 point를 제외한 점들의 좌표(2개가 되겠죠)를 따로 저장한다 (overlapping_point는 1개만 나오는 것이 정상이므로, for문 돌릴 필요가 없습니다)
            all_points_set_nonDuplicate.discard(overlapping_point)
            non_overlapping_points = list(all_points_set_nonDuplicate)

            # 3개의 점으로 삼각형을 만들 수 있는 경우에만 내부에 점이 있는지 판명하면 된다
            # 3개의 점으로 삼각형을 안 만든다면 그냥 넘긴다(continue)
            # 삼각형을 만드는 조건은 맞지만, self.check_availability를 통해, 상대방이 긋는 선이 유효하기까지 해야 아래의 조건들을 확인하는 게 의미가 있다
            if (is_triangle(overlapping_point, non_overlapping_points[0],non_overlapping_points[1])
                    and [non_overlapping_points[0], non_overlapping_points[1]] in self.available):
                # 만약에 non_overlapping_points와 overlapping_point를 이용해서 만들어진 삼각형 내부에 self.whole_points에 있는 점이 1개 이상 있는게 판명되면 그 경우는 삼각형으로 인정 안된다
                for point in self.whole_points:
                    # point가 그으려는(혹은 이미 그어진) 선분에 포함되는 3개의 점이라면 검사할 필요가 없다
                    if point == overlapping_point or point == non_overlapping_points[0] or point == \
                            non_overlapping_points[1]:
                        continue
                    # inner_point_usingInStealChecking 함수로 3개의 점을 기준으로 이루어진 삼각형 내부에 whole_points에 포함되는 점이 있는 지 확인 가능
                    # 일직선 상에 있는 점도 판명 가능(예외처리를 위함)
                    # 다음 connected_line과의 확인을 위해 continue
                    elif inner_point_usingInStealChecking(overlapping_point, non_overlapping_points[0],
                                                          non_overlapping_points[1], point):
                        break
                    else:  # inner_point에서 내부에 점이 없다는 것이 판명되면
                        isDangerous = True  # 실점할 수 있는 위험한 상황으로 판명
            else:
                continue

        # 삼각형을 상대방에게 뺏기는 상황이 한번이라도 발생했다면, isDangerous는 True를 반환했을 것이다
        # 그것을 확인하고, 값이 일치하면 False 반환, 아니면 최종적으로 True 반환
        if isDangerous == False:
            return False
        else:
            return True

    # 유효한 선분인지 검사하는 함수 check_valid_line() (+상대방에게 steal 당하지 않도록 하는 최소한의 알고리즘 적용)
    def check_valid_line(self, line):
        # check_availability()로 1차 검사
        # check_availability()로 선분을 그을 수 없는 예외 사항들을 모두 처리할 수 있다고 판단되었습니다.
        if line not in self.available:
            return False

        # 지금 그을려고 하는 선분을 통해서, check_stealing_situation_inOpponentTurn() 함수의 return 값의 길이가 1 이상일 경우
        # 즉 그리려고 하는 선분을 통해 삼각형을 이루는 선분 3개 중 2개가 이미 그려지게 만들면
        # 이때는 그으면 안되겠죠. 다음 턴에 상대방이 steal할 테니까요.
        if self.check_stealing_situation_inOpponentTurn(line) == True:
            return False

        # 아무런 거 없디면 걍 True를 반환시키죠. check_triangle()에서도 빈 list를 반환하면 random 선택하도록
        return True

    # available은 최악의 상황이 아니면 모두 집어넣고 싶으므로, check_valid_line() 호출
    def find_best_selection(self):
        #맨 처음 시작하는 경우 whole_points가 안 들어오는 경우를 대비한 예외처리
        if len(self.available)+len(self.drawn_lines) < 2:
            self.available = generate_available(self.drawn_lines, self.whole_points)

        if self.isRule:
            # 최악의 상황은 면할 수 있는 check_valid_line()을 통해서 일단 available_skipWorst list 구성
            available_skipWorst = [[point1, point2] for (point1, point2) in list(combinations(self.whole_points, 2)) if
                                   self.check_valid_line([point1, point2])]

            # check_triangle에서 얻은 삼각형을 그릴 수 있는 candidate_line들의 list를 저장합니다.
            # 즉, 하나만 더 그으면 삼각형을 만들 수 있는 바로 그 선분들의 list가 candidate_line에 저장됩니다.
            # available_skipWorst에서 하나만 더 그으면 삼각형을 만들 수 있는 list를 못 찾았다면, available_all에서 찾아야 합니다.
            candidate_line = self.check_triangle(available_skipWorst)
            # 선분이 하나도 연결되어 있지 않은 점이 한개도 남아있지 않은 경우 candidate_line을 available에서 뽑아야 할 것
            showmap(self.drawn_lines, self.whole_points)
            if len(candidate_line) == 0:
                candidate_line = self.check_triangle(self.available)
            if len(candidate_line) != 0:  # candidate_line 리스트가 비어있지 않다면, 즉 하나만 더 그으면 삼각형이 될 수 있는 상황이 있다면
                print("하나만 더 그으면 삼각형 획득!, 아래 리스트는 해당 상황에서의 candidate line")
                return random.choice(candidate_line)  # 그들 중에 random 선택
            elif len(available_skipWorst) != 0:  # candidate_line 리스트는 비어 있는데, available_skipWorst 리스트는 비어있지 않다면
                print(available_skipWorst)
                print("그런 선분은 없지만, 최악의 상황은 면할 수 있는 방도는 있음!")
                return random.choice(available_skipWorst)
            else:  # available_part 리스트도 비어있다면, 그냥 check_availabilty()로 가능한 모든 선분들 중 random 선택해야 할 것입니다.
                print("걍 랜덤임!")
                root = Node()
                root.available = self.available
                root.total_lines = self.drawn_lines
                root.whole_points = self.whole_points
                child_score, _ = root.expand_node(3, '-inf', 'inf')
                print(child_score, _)
                return list(_)
                # eval_line = self.evaluation()
                # if eval_line != -1:
                #     return eval_line

                # # self.isRule = False
                # return random.choice(available_all)

        # elif self.isHeurisitic:
        #     #휴리스틱을 사용할 것임

        # elif self.isMinMax:
        #     #minmax 알고리즘을 사용할 것임

    # 삼각형을 구성할 수 있는 line 집합을 return 해주는 함수
    # 즉, 삼각형의 선분 3개 중 2개의 선분이 그어져 있을 때, 한개만 더 그으면 삼각형 되는데, 그 한개만 더 그으면 되는 선분들의 list를 돌려줌
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
    

    #heuristic()에서 노드용으로 사용할



class Node():
    # 먼저 초기화할 때, alpha, beta 값들을 추가하자.
    def __init__(self, added_line=None, parent=None, alpha=float('-inf'), beta=float('inf')):
        # self.ab_value = 0 (알파베타 가지치기용 isOpponent에 따라서 알파값인지 베타값인지 결정)
        self.alpha = alpha
        self.beta = beta

        if parent is not None:
            self.added_line = added_line  # 추가한 line (이번 turn에 그릴 line)
            self.total_lines = parent.total_lines
            self.whole_points = parent.whole_points
            self.isOpponentTurn = not parent.isOpponentTurn
            score = check_triangle(added_line, self.total_lines, self.whole_points)
            if (score != 0):
                print(score)
            if self.isOpponentTurn:
                self.score = parent.score + score
            else:
                self.score = parent.score - score
            self.available = available_update(parent.available, added_line)
        else:  # 루트노드
            self.added_line = None  # 추가한 line (이번 turn에 그릴 line)
            self.total_lines = None
            self.whole_points = None
            self.score = 0
            self.isOpponentTurn = False
            self.available = []

    def expand_node(self, depth_limit, alpha, beta):

        if len(self.available) == 0:  # 게임 끝났다는 의미
            return self.score, self.added_line
        
        score = float('-inf') if self.isOpponentTurn else float('inf')

        if depth_limit == 0: #노드를 만들면서 tree를 계속 확장하다가 depth-limit에 도달했을 때
            return evaluation(self.total_lines, self.whole_points, self.available), self.added_line  # 평가함수 적용할 계획 (score 대신에 evaluate()을 넣을 것임)

        target_line = None #target_line 비어있는 거 문제 해결

        for l in self.available: #available 리스트의 선분들을 하나씩 보면서 Search
            child = Node(l, parent=self)
            child_score, _ = child.expand_node(depth_limit - 1, alpha, beta)

            print(f"Line: {l}, Child Score: {child_score}, Current Score: {score}")

            if self.isOpponentTurn:  # 상대방 turn일 때 (minimize-player가 되야 하는 경우에)

                if child_score < score:  # child_score가 infinity가 아닐 때만 업데이트 한다.
                    score = child_score

                target_line = l

                try:
                    pruning = self.alpha_pruning(self, child)  # True, false로 반환 (pruning이 가능할 경우에만 True 반환)

                    if pruning:
                        break

                    beta = min(beta, score)
                    if score <= alpha:
                        break
                except:
                    pass


            else:  # 내 turn일 때 (maximize-player가 되야 하는 경우에)
                if child_score > score:  # child_score가 infinity가 아닐 때에만 업데이트
                    score = child_score

                target_line = l
                
                try:
                    pruning = self.beta_pruning(self, child)  # True, false로 반환 (pruning이 가능할 경우에만 True 반환)
                    if pruning:
                        break

                    alpha = max(alpha, score)
                    if score >= beta:
                        break
                except:
                    pass
        
        return (score, target_line)

    def alpha_pruning(self, parent, child):
        # if parent.alpha > child.alpha:
        #     return True
        # else:
        #     return False
        return parent.alpha >= child.alpha

    def beta_pruning(self, parent, child):
        # if parent.beta < child.beta:
        #     return True
        # else:
        #     return False
        return parent.beta <= child.beta







    
