#include "computeff.h"
#include "compute_param.h"
#include "common.h"

void read_wavefront_obj(const std::string filename, TriMesh& m) {
        std::vector<vec3> VN;
        std::vector<vec2> VT;
        std::vector<std::vector<int>> VTID;
        std::vector<std::vector<int>> VNID;

        std::ifstream in;
        in.open (filename, std::ifstream::in);
        if (in.fail())
            throw std::runtime_error("Failed to open " + filename);
        std::string line;
        while (!in.eof()) {
            std::getline(in, line);
            std::istringstream iss(line.c_str());
            std::string type_token;
            iss >> type_token;

            if (type_token=="v") {
                vec3 v;
                for (int i=0;i<3;i++) iss >> v[i];
                //m.points.data->push_back(v);
                m.points.push_back(v);
            } else if (type_token=="vn") {
                vec3 v;
                for (int i=0;i<3;i++) iss >> v[i];
                //VN.push_back(v);
            } else if (type_token=="vt") {
                vec2 v;
                for (int i=0;i<2;i++) iss >> v[i];
               // VT.push_back(v);
            } else if (type_token=="f") {
                std::vector<int> vid;
                //std::vector<int> vnid;
                //std::vector<int> vtid;
                int tmp;
                while (1) { // in wavefront obj all indices start at 1, not zero
                    while (!iss.eof() && !std::isdigit(iss.peek())) iss.get(); // skip (esp. trailing) white space
                    if (iss.eof()) break;
                    iss >> tmp;
                    m.h2v.push_back(tmp - 1);
                    //vid.push_back(tmp-1);
                    if (iss.peek() == '/') {
                        iss.get();
                        if (iss.peek() == '/') {
                            iss.get();
                            iss >> tmp;
                            //vnid.push_back(tmp-1);
                        } else {
                            iss >> tmp;
                            //vtid.push_back(tmp-1);
                            if (iss.peek() == '/') {
                                iss.get();
                                iss >> tmp;
                                //vnid.push_back(tmp-1);
                            }
                        }
                    }
                }
                //VTID.push_back(vtid);
                //VNID.push_back(vnid);

                //int off_f = m.create_facets(1, vid.size());
                //for (int i=0; i<static_cast<int>(vid.size()); i++)
                //    m.vert(off_f, i) = vid[i];
            }
        }

        in.close();
//      std::cerr << "#v: " << m.nverts() << " #f: "  << m.nfacets() << std::endl;
    }

int main(){
    TriMesh m;
    read_wavefront_obj("max.obj", m);
    TriConnectivity tc(m);
    vector<double> vp;
    //for(const vec3& p : m.points) FOR(e, 3) vp.push_back(p[e]);
    FOR(v, m.points.size()) if(tc.v2h[v] != -1) FOR(e, 3) vp.push_back(m.points[v][e]);
    FOR(h, m.nh()) if(tc.opp(h) == -1) {prln("There is boundary verts"); break;}
    prln(m.points.size(), " verts, ", m.nf(), " facets");
    clock_t start = clock();
    compute_param(m.h2v, vp);
    clock_t end = clock();
     double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
    cout << "Computation of ff took time : "  << time_taken*1000 << "\n";
}