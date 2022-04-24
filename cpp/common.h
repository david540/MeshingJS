#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <queue>
#include <stack>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <algorithm>


using namespace std;
#define FORm(i, m, n) for(int i = m; i < (int) n; i++)
#define FOR(i, n) FORm(i, 0, n)
template <typename T>
void prln(T t) { cout << t << "\n"; }
template <typename T, typename ...U>
void prln(T t, U ...u) { cout << t; prln(u...); }

struct vec3{
    double x, y, z;
    double& operator[](const int i) { return i==0 ? x : (1==i ? y : z); }
    double  operator[](const int i) const { return i==0 ? x : (1==i ? y : z); }
} ;
vec3 operator+(const vec3& lhs, const vec3& rhs) { vec3 r = lhs; FOR(i, 3) r[i] += rhs[i]; return r;}
vec3 operator-(const vec3& lhs, const vec3& rhs) { vec3 r = lhs; FOR(i, 3) r[i] -= rhs[i]; return r;}
double dot(const vec3& lhs, const vec3& rhs){ double r = 0; FOR(i, 3) r += lhs[i] * rhs[i]; return r;}
vec3 cross(const vec3 &v1, const vec3 &v2){ return {v1.y*v2.z - v1.z*v2.y, v1.z*v2.x - v1.x*v2.z, v1.x*v2.y - v1.y*v2.x}; }
double norm(const vec3& v){ return sqrt(dot(v, v));}
double norm2(const vec3& v){ return dot(v, v);}
vec3 operator*(const vec3& v, double c) {vec3 r; FOR(i, 3) r[i] = v[i] * c; return r;}
vec3 operator/(const vec3& v, double c) {vec3 r; FOR(i, 3) r[i] = v[i] / c; return r;}
vec3 normalized(const vec3 &v) { return v / norm(v); }

struct vec2{
    double x, y;
    double& operator[](const int i) { return i==0 ? x : y; }
    double  operator[](const int i) const { return i==0 ? x : y; }
} ;
vec2 operator+(const vec2& lhs, const vec2& rhs) { vec2 r = lhs; FOR(i, 2) r[i] += rhs[i]; return r;}
vec2 operator-(const vec2& lhs, const vec2& rhs) { vec2 r = lhs; FOR(i, 2) r[i] -= rhs[i]; return r;}
double dot(const vec2& lhs, const vec2& rhs){ double r = 0; FOR(i, 2) r += lhs[i] * rhs[i]; return r;}
//vec2 cross(const vec2 &v1, const vec2 &v2){ return {v1.y*v2.z - v1.z*v2.y, v1.z*v2.x - v1.x*v2.z, v1.x*v2.y - v1.y*v2.x}; }
double norm(const vec2& v){ return sqrt(dot(v, v));}
vec2 operator*(const vec2& v, double c) {vec2 r; FOR(i, 2) r[i] = v[i] * c; return r;}
vec2 operator/(const vec2& v, double c) {vec2 r; FOR(i, 2) r[i] = v[i] / c; return r;}
vec2 normalized(const vec2 &v) { return v / norm(v); }

struct TriMesh {
    vector<vec3> points{};
    vector<int> h2v{};
    int nf() const { return h2v.size() /3; }
    int nh() const { return h2v.size(); }
};

struct TriConnectivity { 
    TriConnectivity(const TriMesh& p_m);
    vec3 geom(const int h) const { return m.points[m.h2v[next(h)]] - m.points[m.h2v[h]]; }
    int facet(const int h) const { return h / 3; }
    int  from(const int h) const { return m.h2v[h]; }
    int    to(const int h) const { return m.h2v[next(h)]; }
    int  prev(const int h) const { return h - h % 3 + (h + 2) % 3; }
    int  next(const int h) const { return h - h % 3 + (h + 1) % 3; }
    int opp(const int h) const { return h2h[h] == -1 ? -1 : prev(h2h[h]); }
    bool is_boundary_vert(const int v) const { return nav(v2h[v]) == -1; }
    int nav(const int h) const  { return opp(prev(h)); }
    int pav(const int h) const { return h2h[h]; }
    void reset();
    const TriMesh& m;
    vector<int> v2h;
    vector<int> h2h;
};

TriConnectivity::TriConnectivity(const TriMesh& p_m) : m(p_m) { reset(); }

void TriConnectivity::reset() {
    vector<vector<pair<int, int>>> ring(m.points.size());
    v2h.assign(m.points.size(), -1);
    h2h.assign(m.h2v.size(), -1);
    FOR(h, m.h2v.size()) {
        v2h[m.h2v[h]] = h;
        ring[m.h2v[h]].push_back(make_pair(m.h2v[next(h)], h));
    }
    FOR(h, m.h2v.size()) {
        int id = -1;
        FOR(i, ring[m.h2v[next(h)]].size()) if (ring[m.h2v[next(h)]][i].first == m.h2v[h]) {
            id = i; break;
        }
        if (id == -1) v2h[m.h2v[h]] = h;
        else h2h[h] = next(ring[m.h2v[next(h)]][id].second);
    }
    FOR(v, m.points.size()) if (v2h[v] != -1 && pav(v2h[v]) == -1) {
        while (nav(v2h[v]) != -1) v2h[v] = nav(v2h[v]);
    }
}


vector<bool> compute_is_feature(const TriMesh& m, const TriConnectivity& tc, double thr_feature = M_PI / 4){
    auto normal = [&m, &tc] (int f) {
        return normalized(cross(tc.geom(3 * f), tc.geom(3 * f + 1)));
    };
    vector<bool> is_feature(m.nh());
    FOR(h, m.nh()) if (tc.opp(h) == -1 || acos(dot(normal(h/3), normal(tc.opp(h)/3))) > thr_feature) {
		is_feature[h] = true;
    }
    return is_feature;
}


struct DSU {
	vector<int> par, rnk, sze;
	int c;
	DSU(int n) : par(n + 1), rnk(n + 1, 0), sze(n + 1, 1), c(n) {
		for (int i = 1; i <= n; ++i) par[i] = i;
	}
	int find(int i) { return (par[i] == i ? i : (par[i] = find(par[i]))); }
	bool same(int i, int j) { return find(i) == find(j); }
	int get_size(int i) { return sze[find(i)]; }
	int count() { return c; }
	int merge(int i, int j) {
		if ((i = find(i)) == (j = find(j))) return -1;
		else --c;
		if (rnk[i] > rnk[j]) swap(i, j);
		par[i] = j;
		sze[j] += sze[i];
		if (rnk[i] == rnk[j]) rnk[j]++;
		return j;
	}
};