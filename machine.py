import multiprocessing
import random
import time
from multiprocessing import Process
from legacy_machine import Legacy_M
from tools import generate_available, inner_point_usingInStealChecking, is_triangle, \
    check_triangle, evaluation, get_score_line


class MACHINE:

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

    def update_turn(self):
        self.drawn_lines = [sorted(line) for line in self.drawn_lines]
        # if len(self.drawn_lines) < 2:
        self.available = generate_available(self.drawn_lines, self.whole_points)

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
        result_queue.put(list(_))

    def find_best_selection(self):
        result = self.find_best_selection2()
        self.drawn_lines.append(result)
        return result

    def use_lagacy(self):
        legacy = Legacy_M()
        legacy.score = self.score
        legacy.drawn_lines = self.drawn_lines
        legacy.whole_points = self.whole_points
        legacy.location = self.location
        legacy.triangles = self.triangles
        return legacy.find_best_selection()

    def find_best_selection2(self):
        start_t = time.time()
        self.update_turn()
        lagacy_result = self.use_lagacy()
        if type(lagacy_result[0]) == tuple:
            self.available = [lagacy_result]
        else:
            self.available = lagacy_result
        self.available = [sorted(line) for line in self.available]
        print(len(self.available))
        if len(self.available) > 10 or len(self.available) == 1:
            return random.choice(self.available)
        self.isMinMax = True
        if self.isMinMax:
            try:
                result_queue = multiprocessing.Queue()
                minmax_process = Process(target=self.minmax, args=(result_queue,))
                minmax_process.start()
                minmax_process.join(timeout=start_t - time.time() + 50)
                return result_queue.get()
            except:
                return random.choice(self.available)
        if self.isRule:
            rule_result, estimate = self.rule()
            if estimate == 2:
                if self.isMinMax:
                    pass

                return rule_result
        if self.isMinMax:
            if self.isMinMaxActivate:
                remain_time = int(start_t - time.time() + 55)
            else:
                self.isMinMaxActivate = True
                return rule_result
        else:
            return rule_result


class Node:
    def __init__(self, added_line=None, parent=None, alpha=float('-inf'), beta=float('inf')):
        self.alpha = alpha
        self.beta = beta

        if parent is not None:
            self.parent = parent
            self.added_line = added_line
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
            self.available = generate_available(self.total_lines, self.whole_points)
        else:
            self.added_line = None
            self.total_lines = None
            self.whole_points = None
            self.score = 0
            self.isOpponentTurn = False
            self.available = []

    def expand_node(self, depth_limit):
        if len(self.available) == 0:
            return self.score, self.added_line
        score = float('inf') if self.isOpponentTurn else float('-inf')
        if depth_limit == 0:
            return evaluation(self.total_lines, self.whole_points,
                              self.available), self.added_line
        target_line = None
        cnt = len(self.available)
        for l in self.available:
            cnt = cnt - 1
            child = Node(l, parent=self, alpha=self.alpha, beta=self.beta)
            child_score, _ = child.expand_node(depth_limit - 1)
            if self.isOpponentTurn:
                if child_score < self.beta:
                    self.beta = child_score
                    target_line = l
                try:

                    pruning = self.alpha_pruning(self.parent, child)
                    if pruning:
                        break
                    score = min(self.beta, score)
                    if score <= self.alpha:
                        break
                except:
                    pass
            else:
                if child_score > self.alpha:
                    self.alpha = child_score
                    target_line = l
                try:
                    pruning = self.beta_pruning(self.parent, child)
                    if pruning:
                        break
                    score = max(self.alpha, score)

                    if score >= self.beta:
                        break
                except:
                    pass
        return (score, target_line)

    def alpha_pruning(self, parent, child):
        return parent.alpha > child.alpha

    def beta_pruning(self, parent, child):
        return parent.beta < child.beta
