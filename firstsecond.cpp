#include <iostream>
#include <string>
#include <algorithm>
#include <memory.h>
#include <vector>
using namespace std;

#define FAST_IO ios_base::sync_with_stdio(false), cin.tie(NULL), cout.tie(NULL)
#define MS(tmp, num) memset(tmp, num, sizeof(tmp));
#define X first
#define Y second

typedef pair<int, int> pii;

int n, x, y;
vector<pii> dot;
vector<pair<pii, pii>> edge;

int ccw(pii a, pii b, pii c) {
	int s = a.X * b.Y + b.X * c.Y + c.X * a.Y;
	s -= a.Y * b.X + b.Y * c.X + c.Y * a.X;
	if (s > 0)return 1;
	else if (s == 0) return 0;
	else return -1;
}

bool onLine(int a, pii b) {
	pii p1 = edge[a].X;
	pii p2 = edge[a].Y;
	if (p1 == b || p2 == b)return false;
	int s = ccw(p1, p2, b);
	if (s != 0)return false;
	if (p1 < b && b < p2) {
		return true;
	}
	return false;
}

bool intersect(int a, int b) {
	pii p1 = edge[a].X;
	pii p2 = edge[a].Y;
	pii p3 = edge[b].X;
	pii p4 = edge[b].Y;

	int p1p2 = ccw(p1, p2, p3) * ccw(p1, p2, p4);
	int p3p4 = ccw(p3, p4, p1) * ccw(p3, p4, p2);
	return p1p2 < 0 && p3p4 < 0;
}

int main() {
	FAST_IO;

	cin >> n;
	for (int i = 0; i < n; i++) {
		cin >> x >> y;
		dot.push_back({ x, y });
	}
	sort(dot.begin(), dot.end());
	for (int i = 0; i < n - 1; i++) {
		for (int j = i + 1; j < n; j++) {
			edge.push_back({ dot[i], dot[j] });
		}
	}
	for (int i = 0; i < edge.size(); i++) {
		for (int j = 0; j < dot.size(); j++) {
			if (onLine(i, dot[j])) {
				edge.erase(edge.begin() + i);
				i--;
				break;
			}
		}
	}
	for (int i = 0; i < edge.size() - 1; i++) {
		for (int j = i + 1; j < edge.size(); j++) {
			if (intersect(i, j)) {
				edge.erase(edge.begin() + j);
				j--;
			}
		}
	}
	if (edge.size() % 2 == 1) {
		cout << "선공이 유리합니다\n";
	}
	else {
		cout << "후공이 유리합니다.\n";
	}
}
