import multiprocessing
import queue
import random
import time
from collections import Counter
from multiprocessing import Process

from legacy_machine import Legacy_M
from tools import generate_available, inner_point, available_update, inner_point_usingInStealChecking, is_triangle, \
    check_triangle, evaluation, showmap, get_score_line
from shapely.geometry import LineString
from itertools import combinations


class MACHINE:
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
        self.available = []
        self.lastState = []
        self.lastDrawn = []
        self.location = []
        self.triangles = []  # [(a, b), (c, d), (e, f)]
        self.isRule = True
        self.isMinMaxActivate = False
        self.isMinMax = False

    # 상태 업데이트
    def update_turn(self):
        self.drawn_lines = [sorted(line) for line in self.drawn_lines]
        # if len(self.drawn_lines) < 2:
        self.available = generate_available(self.drawn_lines, self.whole_points)



    # 삼각형을 이루는 선분 3개 중 2개가 이미 그어져 있는 경우를 만드는 상황을 판별하기 위한 함수에요
    # 해당 함수에서 반환하는 connected_lines의 length가 1 이상이면 다음 턴에 상대방에게 삼각형을 뺏겨요. (상대방이 하나만 더 그으면 되거든요)
    # available에서 한 선분을 뽑는다면, 선분에 양쪽 끝에 점이 있지 / 그 점에 연결된 선분들을 check를 해서 양쪽 점이 공통으로 가지고 있는 점이 일치하는 점이 있는지 확인 / 그 선분을 추가했을 때, check_triangle() 해서 True 반환하면, must_stealed_point()에서 True 반환
    def check_stealing_situation_inOpponentTurn(self, line):
        connected_lines = [l for l in self.drawn_lines if set(line) & set(l)]
        try:
            connected_lines.remove(line)
        except:
            pass
        if len(connected_lines) < 1:
            return False
        isDangerous = True
        for connected_line in connected_lines:
            all_points_set_withDuplicate = list([connected_line[0], connected_line[1], line[0], line[1]])
            all_points_set_nonDuplicate = set([connected_line[0], connected_line[1], line[0], line[1]])
            # assert len(all_points_set_nonDuplicate)==3, "all 어쩌구는 3이 아니다."
            overlapping_point = \
                [item for item in all_points_set_nonDuplicate if all_points_set_withDuplicate.count(item) > 1][0]
            if not overlapping_point:
                return False
            all_points_set_nonDuplicate.discard(overlapping_point)
            non_overlapping_points = list(all_points_set_nonDuplicate)
            sorted_point_list = sorted([non_overlapping_points[0], non_overlapping_points[1]])
            if (is_triangle(overlapping_point, non_overlapping_points[0], non_overlapping_points[1])) and (
                    sorted_point_list in self.available):
                for point in self.whole_points:
                    if point == overlapping_point or point == non_overlapping_points[0] or point == \
                            non_overlapping_points[1]:
                        continue
                    elif inner_point_usingInStealChecking(overlapping_point, non_overlapping_points[0],
                                                          non_overlapping_points[1], point):
                        isDangerous = False
                        break
                    else:
                        continue
            else:
                continue
        if isDangerous == False:
            return False
        else:
            return True

    # 유효한 선분인지 검사하는 함수 check_valid_line() (+상대방에게 steal 당하지 않도록 하는 최소한의 알고리즘 적용)
    def rule(self):
        square, triangle = get_score_line(self.drawn_lines, self.whole_points, self.available)
        if len(square) != 0:
            return random.choice(square), 2
        if len(triangle) != 0:
            return random.choice(triangle), 1
        guardpoint_line = [[point1, point2] for (point1, point2) in self.available if
                           not self.check_stealing_situation_inOpponentTurn([point1, point2])]
        if len(guardpoint_line) > 1:
            return random.choice(guardpoint_line), 0
        elif len(guardpoint_line) > 0:
            self.isMinMax = True
            return random.choice(guardpoint_line), 0
        self.isMinMax = True
        print("득점도 방어도 하지못함")
        return random.choice(self.available), -1

    def minmax(self, result_queue):
        root = Node()
        root.available = self.available.copy()
        root.total_lines = self.drawn_lines.copy()
        root.whole_points = self.whole_points.copy()
        child_score, _ = root.expand_node(2)
        #print(child_score, _)
        result_queue.put(list(_))  # 리턴대신
        # return list(_)

    def find_best_selection(self):
        result = self.find_best_selection2()
        self.drawn_lines.append(result)
        #self.update_turn()

        return result

    def use_lagacy(self):
        legacy =Legacy_M()
        legacy.score = self.score
        legacy.drawn_lines = self.drawn_lines
        legacy.whole_points = self.whole_points
        legacy.location = self.location
        legacy.triangles = self.triangles
        return legacy.find_best_selection()
        #result_queue.put(legacy.find_best_selection())

    # available은 최악의 상황이 아니면 모두 집어넣고 싶으므로, check_valid_line() 호출
    def find_best_selection2(self):
        # showmap(self.drawn_lines, self.whole_points)
        start_t = time.time()
        self.update_turn()
        # result_queue = multiprocessing.Queue()
        # lagacy_process = Process(target=self.use_lagacy, args=(result_queue,))
        # lagacy_process.start()
        # lagacy_process.join()
        # return result_queue.get()
        lagacy_result = self.use_lagacy()
        if type(lagacy_result[0]) == tuple:
            self.available = [lagacy_result]
        else:
            self.available = lagacy_result
        self.available = [sorted(line) for line in self.available]
        print(len(self.available))
        if len(self.available) > 10 or len(self.available)==1:
            return random.choice(self.available)
        self.isMinMax = True
        if self.isMinMax:  # 민맥스 트리 시작
            # result = self.minmax()
            # print(time.time()-start_t)
            # print(len(self.available))
            try:
                result_queue = multiprocessing.Queue()
                minmax_process = Process(target=self.minmax, args=(result_queue,))
                minmax_process.start()
                minmax_process.join(timeout=start_t-time.time()+30)
                return result_queue.get()
            except:
                return random.choice(self.available)

            # return result_queue.get()
        if self.isRule:  # get_score_line 은 리스트로 반환 + 선분도 정렬된 리스트로 존재
            rule_result, estimate = self.rule()
            if estimate == 2:  # 2점 득점할 수 있으면 바로 2점 득점
                if self.isMinMax:
                    minmax_process.terminate()
                return rule_result
        if self.isMinMax:
            if self.isMinMaxActivate:
                remain_time = int(start_t - time.time() + 55)
                minmax_process.join(timeout=remain_time)
                if result_queue.empty():
                    minmax_process.terminate()
                    return rule_result
                else:
                    return result_queue.get()
            else:
                self.isMinMaxActivate = True
                return rule_result
        else:
            return rule_result


class Node:
    # 먼저 초기화할 때, alpha, beta 값들을 추가하자.
    def __init__(self, added_line=None, parent=None, alpha=float('-inf'), beta=float('inf')):
        # self.ab_value = 0 (알파베타 가지치기용 isOpponent에 따라서 알파값인지 베타값인지 결정)
        self.alpha = alpha
        self.beta = beta

        if parent is not None:  # root 노드가 아닌 경우
            self.parent=parent
            self.added_line = added_line  # 추가한 line (이번 turn에 그릴 line)
            self.total_lines = parent.total_lines.copy()
            self.total_lines.append(self.added_line)
            self.whole_points = parent.whole_points
            self.isOpponentTurn = not parent.isOpponentTurn
            score = check_triangle(added_line, self.total_lines, self.whole_points)
            if self.isOpponentTurn:
                self.score = parent.score + score
            else:
                self.score = parent.score - score
            temp_list = self.total_lines.copy()
            temp_list.remove(added_line)
            self.available = generate_available(self.total_lines,self.whole_points)
        else:  # root 노드인 경우
            self.added_line = None  # 추가한 line (이번 turn에 그릴 line)
            self.total_lines = None
            self.whole_points = None
            self.score = 0
            self.isOpponentTurn = False
            self.available = []

    def expand_node(self, depth_limit):
        if len(self.available) == 0:  # 게임 끝났다는 의미
            return self.score, self.added_line
        score = float('inf') if self.isOpponentTurn else float('-inf')
        if depth_limit == 0:  # 노드를 만들면서 tree를 계속 확장하다가 depth-limit에 도달했을 때
            return evaluation(self.total_lines, self.whole_points,
                              self.available), self.added_line  # 평가함수 적용할 계획 (score 대신에 evaluate()을 넣을 것임)
        target_line = None  # target_line 비어있는 거 문제 해결
        cnt = len(self.available)
        for l in self.available:  # available 리스트의 선분들을 하나씩 보면서 Search
            cnt = cnt - 1
            child = Node(l, parent=self, alpha=self.alpha, beta=self.beta)
            child_score, _ = child.expand_node(depth_limit - 1)
            #print(
            #    f"Line: {l}, Child Score: {child_score}, Current Score: {score}, isOpponentTurn: {self.isOpponentTurn}")
            if self.isOpponentTurn:  # 상대방 turn일 때 (minimize-player가 되야 하는 경우에)
                if child_score < self.beta:  # child_score가 infinity가 아닐 때만 업데이트 한다.
                    self.beta = child_score
                    target_line = l
                try:

                    pruning = self.alpha_pruning(self.parent, child)  # True, false로 반환 (pruning이 가능할 경우에만 True 반환)
                    if pruning:
                        #print("alpha cut-off 발생!")
                        break
                    score = min(self.beta, score)
                    if score <= self.alpha:
                        #print("아무튼 alpha 쪽 발생!")
                        break
                except:
                    pass
            else:  # 내 turn일 때 (maximize-player가 되야 하는 경우에)
                if child_score > self.alpha:  # child_score가 infinity가 아닐 때에만 업데이트
                    self.alpha = child_score
                    target_line = l

                try:

                    pruning = self.beta_pruning(self.parent, child)  # True, false로 반환 (pruning이 가능할 경우에만 True 반환)
                    if pruning:
                        #print("beta cut-off 발생!")
                        break
                    score = max(self.alpha, score)

                    if score >= self.beta:
                        #print("아무튼 beta 머시기 발생!")
                        break
                except:
                    pass
        #print(cnt,"알파 베타")
        return (score, target_line)

    def alpha_pruning(self, parent, child):
        # if parent.alpha > child.alpha:
        #     return True
        # else:
        #     return False
        #print("alpha쪽",parent.alpha,child.alpha)
        return parent.alpha > child.alpha

    def beta_pruning(self, parent, child):
        # if parent.beta < child.beta:
        #     return True
        # else:
        #     return False
        #print("beta쪽", parent.alpha, child.alpha)
        return parent.beta < child.beta
