
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
#include "math.h"

#include "OpenNL_psm/OpenNL_psm.h"

#define M_PI       3.14159265358979323846   // pi

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
vec3 operator*(const vec3& v, double c) {vec3 r; FOR(i, 3) r[i] = v[i] * c; return r;}
vec3 operator/(const vec3& v, double c) {vec3 r; FOR(i, 3) r[i] = v[i] / c; return r;}
vec3 normalized(const vec3 &v) { return v / norm(v); }

struct TriMesh { // polygonal mesh interface
    vector<vec3> points{};
    vector<int> h2v{};
    int nf() const { return h2v.size() /3; }
    int nh() const { return h2v.size(); }
};

struct TriConnectivity { // half-edge-like connectivity interface
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

struct Lstq{
    int N; 
    int state;
    NLContext context;
    Lstq(int nN) : N(nN) {
        context = nlNewContext();
        nlSolverParameteri(NL_LEAST_SQUARES, NL_TRUE);
        nlSolverParameteri(NL_NB_VARIABLES, NLint(nN));
        nlBegin(NL_SYSTEM);
        state = 0;
    }
    void lock(int i, double val){
        if(state != 0) return;
        nlSetVariable(i, val); nlLockVariable(i);
    }
    void addLine(vector<pair<int, double>> co, double rhs){
        if(state == 0){ nlBegin(NL_MATRIX); state = 1; } else if(state != 1) return;
        nlBegin(NL_ROW);
        for(auto [i, w] : co) nlCoefficient(i, w);
        nlEnd(NL_ROW);
    }
    vector<double> solve() {
        vector<double> sol (N, 0) ;
        if(state != 1) { prln("called solve to early"); return sol;}
        nlEnd(NL_MATRIX);
        nlEnd(NL_SYSTEM);
        nlSolve();
        FOR(i, N) sol[i] = nlGetVariable(i);
        nlDeleteContext(context);
        return sol;
    }
};

vector<double> compute_FF_angles(const vector<vector<pair<int, double>>>& adj, const vector<pair<int, double>>& constraints){
    Lstq solver(2 * adj.size());
    for(auto [i, angle] : constraints) FOR(d, 2) solver.lock((i << 1)|d, d == 0 ? cos(4 * angle) : sin(4 * angle));
    FOR(i, adj.size()){
        for(auto [j, omega_0] : adj[i]){
            double rot[2][2] = { { cos(4 * omega_0), sin(4 * omega_0) }, { -sin(4 * omega_0), cos(4 * omega_0) } };
            FOR(d, 2) solver.addLine({make_pair((i << 1) | d, -1), make_pair((j << 1), rot[d][0]), make_pair((j << 1) | 1, rot[d][1])  }, 0);
        }
    }
    vector<double> res = solver.solve();
    vector<double> angles(adj.size());
    FOR(i, adj.size()) angles[i] = (1. / 4) * atan2(res[(i << 1) | 1], res[i << 1]);
    return angles;
}
vector<double> compute_FF_angles(const TriMesh& m, const TriConnectivity& tc, const vector<bool>& is_feature){
    auto vector_angle = [] (const vec3& v0, const vec3& v1) {
        return atan2(norm(cross(v0, v1)), dot(v0, v1));
    };
    auto edge_angle_in_ref = [&m, &tc, &vector_angle](int h) {
        if(h % 3 == 0) return 0.;
        double angle = vector_angle(tc.geom(h), tc.geom(h - h % 3));
        return h % 3 == 1 ? angle : 2. * M_PI - angle;
    };
    auto c_ij = [&m, &tc, &vector_angle, &edge_angle_in_ref] (int h) { 
        return M_PI - edge_angle_in_ref(tc.opp(h)) + edge_angle_in_ref(h); 
    };
    vector<vector<pair<int, double>>> adj(m.nf());
    vector<pair<int, double>> constraints;
    FOR(h, m.nh()) {
        if(tc.opp(h) != -1 && !is_feature[h]) adj[h/3].push_back(make_pair(tc.opp(h)/3, -c_ij(h)));
        else constraints.push_back(make_pair(h/3, edge_angle_in_ref(h)));
    }
    return compute_FF_angles(adj, constraints);
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

vector<double> computeFF(vector<int> const& ph2v, vector<double> const& ppoints){
    TriMesh m; m.h2v = ph2v;
    m.points.resize(ppoints.size()/3);
    FOR(i, ppoints.size()) m.points[i/3][i%3] = ppoints[i]; 
    TriConnectivity tc(m);

    FOR(h, m.nh()) if(tc.opp(h) == -1) {prln("There is boundary verts"); break;}

    vector<bool> is_feature(m.nh(), false);
    
    clock_t start = clock();
    vector<double> angles = compute_FF_angles(m, tc, compute_is_feature(m, tc));
    clock_t end = clock();

    //for(double a : angles) prln(a);
    double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
    cout << "Computation of ff took time : "  << time_taken << "\n";


    auto normal = [&m, &tc] (int f) { return normalized(cross(tc.geom(3 * f), tc.geom(3 * f + 1))); };
    vector<double> crosses;
    vector<double> barys;
    FOR(f, m.nf()) {
        vec3 x = normalized(tc.geom(3 * f)); 
        vec3 y = normalized(cross(normal(f), x));
        vec3 b = (m.points[m.h2v[3 * f]] + m.points[m.h2v[3 * f + 1]] + m.points[m.h2v[3 * f + 2]]) / 3;
        FOR(i, 2) {
            double a = angles[f] + i * M_PI / 2.;
            vec3 g = (x * cos(a) + y * sin(a));
            FOR(e, 3){
                crosses.push_back(g[e]);
                barys.push_back(b[e]);
            } 
        }
    }
    crosses.insert(crosses.end(), barys.begin(), barys.end());
    return crosses;
}

