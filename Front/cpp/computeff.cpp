#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <queue>
#include <stack>
#include <array>
#include <chrono>

using namespace std;

using vi = vector<int>;
#define FORm(i, m, n) for(int i = m; i < (int) n; i++)
#define FOR(i, n) FORm(i, 0, n)
template <typename T>
void prln(T t) { cout << t << "\n"; }
template <typename T, typename ...U>
void prln(T t, U ...u) { cout << t; prln(u...); }

struct pair_hash {
    size_t operator() (const std::pair<int, int>& pair) const {
        return ((size_t)(pair.first) << 32) | pair.second;
    }
};

struct TriMesh { // polygonal mesh interface
    vector<array<double, 3>> points{};
    vi h2v{};
};
struct TriConnectivity { // half-edge-like connectivity interface
    TriConnectivity(const TriMesh& p_m);
    array<double, 3> geom(const int h) const { return m.points[m.h2v[h]]; }
    int facet(const int h) const { return h / 3; }
    int  from(const int h) const { return m.h2v[h]; }
    int    to(const int h) const { return m.h2v[next(h)]; }
    int  prev(const int h) const { return h - h % 3 + (h + 2) % 3; }
    int  next(const int h) const { return h - h % 3 + (h + 1) % 3; }
    int opp(const int h) { return h2h[h] == -1 ? -1 : prev(h2h[h]); }
    bool is_boundary_vert(const int v) { return nav(v2h[v]) == -1; }
    int nav(const int h) { return opp(prev(h)); }
    int pav(const int h) { return h2h[h]; }
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

inline void file_must_no_be_at_end(std::ifstream& f, const std::string& reason = " should not") {
    if (f.eof()) {
        f.close();
        throw std::runtime_error("File ended to soon while " + reason);
    }
}

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

void read_medit_format(const std::string& filename, std::vector<array<double, 3>>& verts_, std::vector<int>& tris_) {
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

int main(int argc, char** argv) {

    TriMesh m;
    read_medit_format(argv[1], m.points, m.h2v);
    TriConnectivity fec(m);

    int ave_val = 0;
    int ct_v = 0;
    FOR(v, m.points.size()) if(fec.v2h[v] != -1) {
        int h = fec.v2h[v];
        ct_v ++;
        do{
            ave_val++;
            h = fec.h2h[h];
            //prln(fec.v2h[v], " : ", h);
        }while(h != -1 && h != fec.v2h[v]);
    }
    FOR(i, 3) FOR(h, m.h2v.size()) if(fec.opp(h) == -1){
       
    }

    prln(ave_val, " ", m.points.size(), " ", m.h2v.size(), " ", ct_v);

}
