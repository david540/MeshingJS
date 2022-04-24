#pragma once

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
	auto get_normal_edge_angle = [&tc, &vector_angle] (int h){
		int f = h / 3, e = h % 3;
		double angle = vector_angle(tc.geom(h), tc.geom(3 * f));
		if (e == 2) angle = -angle;
		angle -= M_PI_2;
		while (angle <= -M_PI) angle += 2 * M_PI;
		while (angle > M_PI) angle -= 2 * M_PI;
		return angle;
	};
    vector<vector<pair<int, double>>> adj(m.nf());
    vector<pair<int, double>> constraints;
    FOR(h, m.nh()) {
        if(tc.opp(h) != -1 && !is_feature[h]) adj[h/3].push_back(make_pair(tc.opp(h)/3, -c_ij(h)));
        else constraints.push_back(make_pair(h/3, get_normal_edge_angle(h)));//edge_angle_in_ref(h)));
    }
    return compute_FF_angles(adj, constraints);
}

pair<vector<int>, vector<int>> compute_j_v(const TriMesh& m, const TriConnectivity& tc, std::vector<vec2>& angles) {
	auto vector_angle = [](vec3 v0, vec3 v1) { return atan2(norm(cross(v0, v1)), dot(v0, v1)); };
	auto edge_angle_in_ref = [&m, &tc, &vector_angle](int h) {
        if(h % 3 == 0) return 0.;
        double angle = vector_angle(tc.geom(h), tc.geom(h - h % 3));
        return h % 3 == 1 ? angle : 2. * M_PI - angle;
    };
    auto c_ij = [&m, &tc, &vector_angle, &edge_angle_in_ref] (int h) { 
        return M_PI - edge_angle_in_ref(tc.opp(h)) + edge_angle_in_ref(h); 
    };
	vector<int> LS(m.nh(), 0);
	FOR(h, m.nh()) if (tc.opp(h) != -1) {
		int f = h / 3, nei = tc.opp(h) / 3;
		LS[h] = -std::round((c_ij(h) + angles[nei][0] - angles[f][0]) / M_PI_2);
	}
	vector<int> valence(m.points.size(), 4);
	FOR(v, m.points.size()) if(tc.v2h[v] != -1 && !tc.is_boundary_vert(v)) {
		int h = tc.v2h[v];
		int sum_LS = 0;
		do{
			sum_LS += LS[h];
			h = tc.pav(h);
		}while(h != tc.v2h[v] && h != -1);
		valence[v] += sum_LS;
		while(valence[v] >= 8) valence[v] -=4;
		while(valence[v] <= 0) valence[v] += 4;
	}
	return make_pair(LS, valence);
}

void smooth_nonortho(const TriMesh& m, const TriConnectivity& tc, vector<vec2>& angles, const vector<bool>& is_feature, const vector<int>& LS) {//, FacetAttribute<vec2>& out_angles) {
	auto vector_angle = [](vec3 v0, vec3 v1) { return atan2(norm(cross(v0, v1)), dot(v0, v1)); };
	auto corner_angle = [&m, &tc, &vector_angle] (int h) { return vector_angle(m.points[m.h2v[tc.next(h)]] - m.points[m.h2v[h]], m.points[m.h2v[tc.prev(h)]] - m.points[m.h2v[h]]); };
	auto edge_angle_in_ref = [&m, &tc, &vector_angle](int h) {
        if(h % 3 == 0) return 0.;
        double angle = vector_angle(tc.geom(h), tc.geom(h - h % 3));
        return h % 3 == 1 ? angle : 2. * M_PI - angle;
    };
    auto c_ij = [&m, &tc, &vector_angle, &edge_angle_in_ref] (int h) { 
        return M_PI - edge_angle_in_ref(tc.opp(h)) + edge_angle_in_ref(h); 
    };
	auto facet_normal = [&tc] (int f) { return normalized(cross(tc.geom(3 * f), tc.geom(3 * f + 1))); };
	auto get_flagging = [&tc, &facet_normal, &angles] (int h){
		int f = h/3;
		vec3 x = normalized(tc.geom(h - h%3));
		vec3 y = normalized(cross(facet_normal(f), x));
		return abs(dot(normalized(tc.geom(h)), (x * cos(angles[f][0]) + y * sin(angles[f][0])))) < abs(dot(normalized(tc.geom(h)), (x * cos(angles[f][1]) + y *  sin(angles[f][1])))) ? 0 : 1;
	};
	vector<double> omega(m.nh(), 0);
	vector<double> omega_plat(m.nh());
	vector<double> total_omega(m.nh(), 0);
	vector<bool> corner_cad(m.nf(), false);

	FOR(h, m.nh()) if (is_feature[h]) {
		int i = 1, cur = h, n = 0; double angle_sum = 0;
		while (!is_feature[tc.prev(cur)]) { angle_sum += corner_angle(cur); n++; cur = tc.nav(cur); }
		angle_sum += corner_angle(cur);
		for (; i < 20 && angle_sum > i * M_PI_2 + M_PI_4 + (1 - i % 2) * M_PI_4 / 2.; i++); 
		double diffangle = i * M_PI_2 - angle_sum;
		for (cur = tc.nav(h); cur != -1 && !is_feature[cur]; ) {
			total_omega[cur] = diffangle / n;
			if (i % 2 == 0) {
				omega_plat[cur] = 2 * diffangle / n;
				omega_plat[tc.opp(cur)] = -omega_plat[cur];
			}
			else corner_cad[cur/3] = true;
			cur = tc.nav(cur);
		}
	}

	DSU ds(m.nf());
	FOR(h, m.nh()) if (!is_feature[h] && tc.opp(h) != -1) {
		ds.merge(h / 3, tc.opp(h) / 3);
	}
	FOR(h, m.nh()) {
		if(corner_cad[h/3]) corner_cad[ds.find(h/3)] = true;
		if (!is_feature[h]) omega[h] = - c_ij(h);//omega_plat[h] - c_ij(h);
	}
	auto context = nlNewContext();
	{
		nlSolverParameteri(NL_LEAST_SQUARES, NL_TRUE);
		nlSolverParameteri(NL_NB_VARIABLES, NLint(2 * m.nf()));
		nlBegin(NL_SYSTEM);
		FOR(f, m.nf()) FOR(e, 3) if (is_feature[3 * f + e]) {
			int d = get_flagging(3 * f + e);
			double angle = angles[f][d];
			nlSetVariable(2 * f + d, angle);
			nlLockVariable(2 * f + d);
			if (!corner_cad[ds.find(f)]) {
				nlSetVariable(2 * f + (d + 1) % 2, angles[f][(d + 1) & 1]); 
				nlLockVariable(2 * f + (d + 1) % 2);
				corner_cad[ds.find(f)] = true;
			}
		} 
		nlBegin(NL_MATRIX);
		FOR(f, m.nf()) FOR(e, 3) if (!is_feature[3 * f + e]) FOR(d, 2) {
			int nei = tc.opp(3 * f + e) / 3;
			int nei_lay = d + LS[3 * f + e];//+ 1000) & 3;//(d + LS[3 * f + e] + 1000) & 3;
			nei_lay = (nei_lay + 1000) & 3;
			int nei_dim = (d +  LS[3 * f + e] + 1000) & 1;
			nlBegin(NL_ROW);
			nlCoefficient(2 * f + d, -1);
			nlCoefficient(2 * nei + nei_dim, 1);

			double rhs = (nei_dim - d -  LS[3 * f + e]) * M_PI_2 + (nei_lay >= 2 ? M_PI : 0);
			int ff_jump = std::round(rhs / (2 * M_PI));
			nlRightHandSide(2 * M_PI * ff_jump + omega[3 * f + e]);
			nlEnd(NL_ROW);
		}
		nlEnd(NL_MATRIX);
		nlEnd(NL_SYSTEM);
		nlSolve();
		FOR(f, m.nf()) FOR(d, 2) {
			angles[f][d] = nlGetVariable(2 * f + d); //+ M_PI / 2;
		}
		nlDeleteContext(context);
	}
}

void rectify_topo(const TriMesh& m, const TriConnectivity& tc, const vector<int>& in_valences, vector<int>& out_valences, vector<int>& LS, const vector<bool>& is_feature){
	FOR(v, m.points.size()){		
		if(!tc.is_boundary_vert(v) && in_valences[v] != 4 && out_valences[v] != in_valences[v]){
			while(out_valences[v] != in_valences[v]){
				int sign = out_valences[v] > in_valences[v] ? 1 : -1;
				vector<int> prev(m.nh(), -1);
				queue<tuple<int, int, int>> q;
				{
					int h = tc.v2h[v];
					do{
						q.push(make_tuple(h, h, h));
						h = tc.pav(h);
					}while(h != tc.v2h[v]);
				}
				int end_h = -1;
				while(!q.empty()){
					auto [h, p, root] = q.front(); q.pop();
					if(prev[h] != -1) continue;

					if(in_valences[m.h2v[h]] != 4) prln(in_valences[m.h2v[h]]);
					if(in_valences[m.h2v[h]] != out_valences[m.h2v[h]] &&  (in_valences[m.h2v[h]] > out_valences[m.h2v[h]] ? 1 : -1) == sign){
						prln("FOUND ", h, " ", tc.v2h[v], " ", in_valences[m.h2v[h]], " ",  out_valences[m.h2v[h]]);
						prev[h] = p;
						end_h = h;//prev[h];
						break;
					}
					int st_h = h;
					while(tc.nav(st_h) != h && tc.nav(st_h) != -1 && !is_feature[tc.nav(st_h)]) st_h = tc.nav(st_h);
					int cur_h = st_h;
					do {
						if(prev[cur_h] != -1) prln("BUG 32020");
						prev[cur_h] = p;
						if(prev[tc.next(cur_h)] == -1){
							q.push(make_tuple(tc.next(cur_h), cur_h, root));
						}
						if(is_feature[cur_h]) break;
						cur_h = tc.pav(cur_h);
					}while(cur_h != -1 && cur_h != st_h);
				}
				if(end_h == -1){
 					end_h = tc.next(tc.v2h[v]);
					prev[tc.next(tc.v2h[v])] = tc.v2h[v];
				}
				out_valences[m.h2v[end_h]] += sign;
				out_valences[v] -= sign;
				while(m.h2v[end_h] != v){
					if(prev[end_h] == -1) prln("BUG 3023");
					end_h = prev[end_h];
					LS[end_h] -= sign;
					LS[tc.opp(end_h)] += sign;
				}
			}
		}
	}
}

pair<vector<vec2>, vector<int>> compute_FF_angles_with_input_topo(const TriMesh& m, const TriConnectivity& tc, const vector<int>& in_valences){
    clock_t start = clock();
    vector<bool> is_feature = compute_is_feature(m, tc);
    vector<double> angles = compute_FF_angles(m, tc, is_feature);
    vector<vec2> alpha(angles.size()); FOR(i, angles.size()) alpha[i] = {angles[i], angles[i] + M_PI_2};
    auto [LS, valences] = compute_j_v(m, tc, alpha);
	
	rectify_topo(m, tc, in_valences, valences, LS, is_feature);
	smooth_nonortho(m, tc, alpha, is_feature, LS);
    clock_t end = clock();

    //for(double a : angles) prln(a);
    double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
    cout << "Computation of ff took time : "  << time_taken*1000 << "\n";
	return make_pair(alpha, valences);
}

vector<double> computeFF(const vector<int>& ph2v, const vector<double>& ppoints, const vector<int>& in_valences){
    TriMesh m; m.h2v = ph2v;
    m.points.resize(ppoints.size()/3);
    FOR(i, ppoints.size()) m.points[i/3][i%3] = ppoints[i]; 
    TriConnectivity tc(m);

    FOR(h, m.nh()) if(tc.opp(h) == -1) {prln("There is boundary verts"); break;}

    auto [alpha, valences] = compute_FF_angles_with_input_topo(m, tc, in_valences);

    auto normal = [&m, &tc] (int f) { return normalized(cross(tc.geom(3 * f), tc.geom(3 * f + 1))); };
    vector<double> out_v; 
    int infos[2] = {(int)m.h2v.size(),(int) m.points.size()};
    out_v.push_back(m.h2v.size());
    out_v.push_back(m.points.size());
	
    vector<double> crosses;
    vector<double> barys;
    FOR(f, m.nf()) {
        vec3 x = normalized(tc.geom(3 * f)); 
        vec3 y = normalized(cross(normal(f), x));
        vec3 b = (m.points[m.h2v[3 * f]] + m.points[m.h2v[3 * f + 1]] + m.points[m.h2v[3 * f + 2]]) / 3;
        
        FOR(i, 2) {
            double a = alpha[f][i];//angles[f] + i * M_PI / 2.;
            vec3 g = (x * cos(a) + y * sin(a));
            FOR(e, 3) crosses.push_back(g[e]);
            FOR(e, 3) barys.push_back(b[e]);
        }
    }
    out_v.insert(out_v.end(), crosses.begin(), crosses.end());
    out_v.insert(out_v.end(), barys.begin(), barys.end());
    out_v.reserve(out_v.size() + valences.size());
    for(int v : valences) out_v.push_back(v);
    return out_v;
}

