#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <queue>
#include <stack>
#include <array>
#include <chrono>

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

inline void file_must_no_be_at_end(std::ifstream& f, const std::string& r = "bad") { if (f.eof()) { f.close(); throw std::runtime_error("ERROR 32323");} }

inline static bool string_start(const std::string& string, const std::string& start_of_string) {
    size_t start = 0;
    FOR(i, string.size()) if (string[i] != ' ' && string[i] != '\t') {
        start = (size_t)i;
        break;
    }
    std::string copy_without_space(string.begin() + start, string.end());
    if (copy_without_space.size() < start_of_string.size()) return false;
    return (std::string(copy_without_space.begin(), copy_without_space.begin() + (long int)start_of_string.size()) == start_of_string);
}

void read_medit_format(const std::string& filename, std::vector<vec3>& verts_, std::vector<int>& tris_) {
    std::ifstream in;
    in.open(filename, std::ifstream::in);
    if (in.fail())
        throw std::runtime_error("Failed to open " + filename);

    std::string firstline;

    while (!in.eof()) {
        std::getline(in, firstline);
        if (string_start(firstline, "Vertices")) {
            std::string line;
            int nb_of_vertices = 0;
            {
                file_must_no_be_at_end(in, "parsing vertices");
                std::getline(in, line);
                std::istringstream iss(line.c_str());
                iss >> nb_of_vertices;
            }
            verts_.resize(nb_of_vertices);
            FOR(v, nb_of_vertices) {
                file_must_no_be_at_end(in, "parsing vertices");
                std::getline(in, line);
                std::istringstream iss(line.c_str());
                FOR(i, 3)  iss >> verts_[v][i];
            }
        }
        if (string_start(firstline, "Triangles")) {
            std::string line;
            int nb_of_tri = 0;
            {
                file_must_no_be_at_end(in, "parsing Triangles");
                std::getline(in, line);
                std::istringstream iss(line.c_str());
                iss >> nb_of_tri;
            }
            tris_.resize(3 * nb_of_tri);
            FOR(t, nb_of_tri) {
                file_must_no_be_at_end(in, "parsing Triangles");
                std::getline(in, line);
                std::istringstream iss(line.c_str());
                FOR(i, 3) {
                    int a = 0;
                    iss >> a;
                    tris_[3 * t + i] = a - 1;
                }
            }
        }
    }
}

struct Lstq{
    Lstq(int N){}
    void lock(int i, double val){}
    void addLine(vector<pair<int, double>> co, double rhs){}
    vector<double> solve() {return vector<double>();}
};

vector<double> compute_FF_angles(const vector<vector<pair<int, double>>>& adj, const vector<pair<int, double>>& constraints){
    Lstq solver(2 * adj.size());
    for(auto [i, val] : constraints){
        solver.lock(i, val);
    }
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
vector<double> compute_FF_angles(const TriMesh& m, const TriConnectivity& tc){
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
        if(tc.opp(h) != -1) adj[h/3].push_back(make_pair(tc.opp(h), c_ij(h)));
        else constraints.push_back(make_pair(h/3, edge_angle_in_ref(h)));
    }
    return compute_FF_angles(adj, constraints);
}

int main(int argc, char** argv) {

    TriMesh m;
    read_medit_format(argv[1], m.points, m.h2v);
    TriConnectivity tc(m);

    int ave_val = 0;
    int ct_v = 0;
    FOR(v, m.points.size()) if(tc.v2h[v] != -1) {
        int h = tc.v2h[v];
        ct_v ++;
        do{
            ave_val++;
            h = tc.h2h[h];
            //prln(fec.v2h[v], " : ", h);
        }while(h != -1 && h != tc.v2h[v]);
    }
    FOR(i, 3) FOR(h, m.h2v.size()) if(tc.opp(h) == -1){
       
    }

    clock_t start = clock();
    vector<double> angles = compute_FF_angles(m, tc);
    clock_t end = clock();
    double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
    cout << "Computation of ff took time : "  << time_taken << "\n";
    prln(ave_val, " ", m.points.size(), " ", m.h2v.size(), " ", ct_v);

}
