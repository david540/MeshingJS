
#include "common.h"

#include "OpenNL_psm/OpenNL_psm.h"


using namespace std;



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
    cout << "Computation of ff took time : "  << time_taken*1000 << "\n";


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

